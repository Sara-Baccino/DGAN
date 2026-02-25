import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ==================================================================
# TIME ENCODER
# ==================================================================
class TimeEncoder(nn.Module):
    """
    Genera [t_norm, delta_t] da z_temporal tramite MLP leggero,
    separato dal GRU principale per evitare dipendenze circolari.

    Perché t_norm E delta_t insieme?
      t_norm:  contesto globale ("dove siamo nella traiettoria").
               ALP a mese 10 vs mese 240 sono fisiologicamente diversi
               anche con lo stesso Dt.
      delta_t: velocità locale ("quanto tempo dall'ultima visita").
               Due visite con Dt=6 a inizio vs fine follow-up hanno
               dinamiche diverse che t_norm da solo non cattura.
    """

    def __init__(self, z_temporal_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_temporal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, z_temporal: torch.Tensor):
        """
        z_temporal: [B, T, z_dim]
        Returns: t_norm [B,T], delta_t [B,T], t_feat [B,T,2]
        """
        delta_raw = F.softplus(self.mlp(z_temporal).squeeze(-1))  # [B,T] >0
        t_cumul   = torch.cumsum(delta_raw, dim=1)                 # [B,T]
        t_max     = t_cumul[:, -1:].clamp(min=1e-8)
        t_norm    = t_cumul  / t_max                               # in [0,1]
        delta_t   = delta_raw / t_max                              # normalizzato
        t_feat    = torch.stack([t_norm, delta_t], dim=-1)         # [B,T,2]
        return t_norm, delta_t, t_feat


# ==================================================================
# HIERARCHICAL GENERATOR
# ==================================================================
class HierarchicalGenerator(nn.Module):
    """
    Generatore DoppelGANger-style:
      - TimeEncoder separato: MLP su z_temporal -> [t_norm, delta_t]
      - GRU principale: z_temporal + static_embed + [t_norm, delta_t]
      - cummax hard constraint per variabili irreversibili
      - Sigmoid sull'head continua: output in [0,1] coerente con min-max scaling
    """

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim:   int,
        z_temporal_dim: int,
        hidden_dim:     int,
        gru_layers:     int,
        dropout:        float
    ):
        super().__init__()
        self.data_config    = data_config
        self.preprocessor   = preprocessor
        self.max_len        = data_config.max_len
        self.hidden_dim     = hidden_dim
        self.z_temporal_dim = z_temporal_dim

        # Time encoder separato
        self.time_encoder = TimeEncoder(z_temporal_dim, hidden_dim)

        # Static pathway
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # GRU: z_temporal + static_embed + [t_norm, delta_t]
        self.gru = nn.GRU(
            input_size=z_temporal_dim + hidden_dim + 2,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.visit_head = nn.Linear(hidden_dim, 1)

        # Head continua con Sigmoid: output in [0,1]
        self.temporal_cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, data_config.n_temp_cont),
            nn.Sigmoid(),
        )

        # Heads categoriche temporali
        self.temporal_cat_heads = nn.ModuleDict()
        self.temporal_cat_irrev = {}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.irreversible:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, 1)
                self.temporal_cat_irrev[var.name] = True
            else:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, len(var.mapping))
                self.temporal_cat_irrev[var.name] = False

        # Heads statiche
        self.static_cont_head = None
        if data_config.n_static_cont > 0:
            self.static_cont_head = nn.Sequential(
                nn.Linear(hidden_dim, data_config.n_static_cont),
                nn.Sigmoid(),
            )

        self.static_cat_heads = nn.ModuleDict()
        for v in data_config.static_cat:
            if v.name in preprocessor.embedding_configs:
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, preprocessor.embedding_configs[v.name]
                )
            else:
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

    # ------------------------------------------------------------------
    @staticmethod
    def _cummax_irreversible(hazard: torch.Tensor) -> torch.Tensor:
        """
        Hard constraint monotonicità 0->1.
        state[t] = max(hazard[0..t]) -- differenziabile via torch.cummax.
        """
        states, _ = torch.cummax(hazard, dim=1)
        return states

    # ------------------------------------------------------------------
    def forward(
        self,
        z_static:        torch.Tensor,
        z_temporal:      torch.Tensor,
        temperature:     float                  = 1.0,
        teacher_forcing: bool                   = False,
        real_irr:        Optional[torch.Tensor] = None,
        fixed_visits:    Optional[int]          = None,
    ) -> Dict:

        B, T, _ = z_temporal.shape
        device  = z_temporal.device

        # 1. Tempi (MLP separato, no circolarita)
        t_norm, delta_t, t_feat = self.time_encoder(z_temporal)   # [B,T], [B,T], [B,T,2]

        # 2. Static embedding
        static_h     = self.fc_static(z_static)                   # [B, H]
        static_h_rep = static_h.unsqueeze(1).expand(-1, T, -1)    # [B, T, H]

        # 3. GRU con z_temporal + static + [t_norm, delta_t]
        gru_in   = torch.cat([z_temporal, static_h_rep, t_feat], dim=-1)
        h_seq, _ = self.gru(gru_in)                                # [B, T, H]

        # 4. Static outputs
        static_cont      = None
        static_cat       = {}
        static_cat_embed = {}
        embed_cfg        = self.preprocessor.embedding_configs

        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(static_h)

        for name, head in self.static_cat_heads.items():
            out = head(static_h)
            if name in embed_cfg:
                static_cat_embed[name] = out.contiguous()
            else:
                static_cat[name] = F.gumbel_softmax(out, tau=temperature, hard=True, dim=-1)

        # 5. Visit mask (monotonicita: se una visita e assente, le successive lo sono)
        visit_prob = torch.sigmoid(self.visit_head(h_seq).squeeze(-1))
        raw_mask   = torch.bernoulli(visit_prob)
        raw_mask[:, 0] = 1.0

        mask_list = [raw_mask[:, 0]]
        current_m = raw_mask[:, 0]
        for t in range(1, T):
            current_m = raw_mask[:, t] * current_m
            mask_list.append(current_m)
        visit_mask = torch.stack(mask_list, dim=1).unsqueeze(-1)   # [B, T, 1]

        if fixed_visits is not None:
            visit_mask[:, fixed_visits:] = 0

        # 6. Temporal continuous (Sigmoid -> [0,1])
        temporal_cont = self.temporal_cont_head(h_seq) * visit_mask

        # 7. Temporal categorical
        temporal_cat    = {}
        irrev_var_names = [n for n, irrev in self.temporal_cat_irrev.items() if irrev]

        for name, head in self.temporal_cat_heads.items():
            if self.temporal_cat_irrev[name]:
                hazard = torch.sigmoid(head(h_seq).squeeze(-1))    # [B, T]
                if teacher_forcing and real_irr is not None:
                    irr_idx    = irrev_var_names.index(name)
                    irr_states = real_irr[:, :, irr_idx].float()
                else:
                    irr_states = self._cummax_irreversible(hazard)
                one_hot = torch.stack([1.0 - irr_states, irr_states], dim=-1)
                temporal_cat[name] = one_hot * visit_mask
            else:
                y = F.gumbel_softmax(head(h_seq), tau=temperature, hard=True, dim=-1)
                temporal_cat[name] = y * visit_mask

        return {
            "static_cont":      static_cont,
            "static_cat":       static_cat       if static_cat       else None,
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont":    temporal_cont,
            "temporal_cat":     temporal_cat,
            "visit_mask":       visit_mask,
            "visit_times":      t_norm,
        }

class HierarchicalGenerator1(nn.Module):
    """
    Generatore DoppelGANger-style con:
      - static latent z_s
      - temporal latent z_t
      - GRU autoregressiva
      - generazione Δt -> visit_times cumulativi
      - generazione visit_mask
      - categoriche irreversibili con hazard
    """

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim: int,
        z_temporal_dim: int,
        hidden_dim: int,
        gru_layers: int,
        dropout: float,
        cond_dim: int = 0,
    ):
        super().__init__()

        self.data_config  = data_config
        self.preprocessor = preprocessor
        self.max_len      = data_config.max_len
        self.hidden_dim   = hidden_dim

        # -------------------------------
        # static pathway
        # -------------------------------
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # -------------------------------
        # temporal core
        # -------------------------------
        self.gru = nn.GRU(
            input_size=z_temporal_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # -------------------------------
        # heads
        # -------------------------------
        self.time_head = nn.Linear(hidden_dim, 1)      # Δt > 0
        self.visit_head = nn.Linear(hidden_dim, 1)     # Bernoulli logits

        #self.temporal_cont_head = nn.Linear(hidden_dim, data_config.n_temp_cont)
        self.temporal_cont_head = nn.Sequential(
            nn.Linear(hidden_dim, data_config.n_temp_cont),
        )


        # categoriche temporali
        self.temporal_cat_heads = nn.ModuleDict()
        self.temporal_cat_irrev = {}

        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue

            if var.irreversible:
                # binaria irreversibile: stato 0/1
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, 1)
                self.temporal_cat_irrev[var.name] = True
            else:
                self.temporal_cat_heads[var.name] = nn.Linear(
                    hidden_dim, len(var.mapping)
                )
                self.temporal_cat_irrev[var.name] = False
        
        # ======= static heads =========================
        self.static_cont_head = None
        if data_config.n_static_cont > 0:
            self.static_cont_head = nn.Linear(hidden_dim, data_config.n_static_cont)

        self.static_cat_heads = nn.ModuleDict()
        for v in data_config.static_cat:
            if v.name in preprocessor.embedding_configs:
                # embedding → vettore continuo
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, preprocessor.embedding_configs[v.name]
                )
            else:
                # one-hot
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, v.n_categories
                )


    # ------------------------------------------------------------------
    def forward(
        self,
        z_static: torch.Tensor,
        z_temporal: torch.Tensor,
        temperature: float = 1.0,
        teacher_forcing: bool = False,
        real_irr: Optional[torch.Tensor] = None,
        fixed_visits=None
    ) -> Dict:

        B, T, _ = z_temporal.shape
        device  = z_temporal.device

        # -------- static --------
        static_h = self.fc_static(z_static)                  # [B, H]
        static_h_rep = static_h.unsqueeze(1).expand(-1, T, -1)

        # -------- GRU --------
        gru_in = torch.cat([z_temporal, static_h_rep], dim=-1)
        h_seq, _ = self.gru(gru_in)                           # [B, T, H]

        # ==============================================================
        # static variables
        # ==============================================================

        static_cont = None
        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(static_h)   # [B, S_cont]

        static_cat = {}
        static_cat_embed = {}

        embed_cfg = self.preprocessor.embedding_configs

        for name, head in self.static_cat_heads.items():
            out = head(static_h)

            if name in embed_cfg:
                # 🔒 SEMPRE Tensor [B, D]
                static_cat_embed[name] = out.contiguous()
            else:
                static_cat[name] = F.gumbel_softmax(
                    out, tau=temperature, hard=True, dim=-1
                )


        # ==============================================================
        # visit times (Δt → cumulativo)
        # ==============================================================
        delta_t = F.softplus(self.time_head(h_seq)).squeeze(-1)   # [B,T], >0
        visit_times = torch.cumsum(delta_t, dim=1)

        # normalizzazione per-sequenza → [0,1]
        visit_times = visit_times / (
            visit_times.max(dim=1, keepdim=True)[0] + 1e-8
        )

        # ==============================================================
        # visit mask
        # ==============================================================
        visit_logits = self.visit_head(h_seq).squeeze(-1)
        visit_prob   = torch.sigmoid(visit_logits)

        visit_mask = torch.bernoulli(visit_prob)
        visit_mask[:, 0] = 1.0  # prima visita sempre presente

        #Invece di un loop che modifica m, creiamo una nuova lista
        mask_list = []
        current_m = visit_mask[:, 0] # [B]
        mask_list.append(current_m)

        for t in range(1, T):
            # Ogni step è il prodotto del corrente per il precedente
            current_m = visit_mask[:, t] * current_m
            mask_list.append(current_m)

        # Stack sulla dimensione del tempo -> [B, T]
        visit_mask = torch.stack(mask_list, dim=1) 

        # AGGIUNGI IL DIMENSIONES 1 PER IL BROADCASTING -> [B, T, 1]
        visit_mask = visit_mask.unsqueeze(-1)

        if fixed_visits is not None:
            visit_mask[:, fixed_visits:] = 0


        # ==============================================================
        # temporal continuous
        # ==============================================================
        temporal_cont = self.temporal_cont_head(h_seq)
        temporal_cont = temporal_cont * visit_mask
        #raw = self.temporal_cont_head(h_seq)
        #temporal_cont = torch.exp(raw) - 1e-3
        #temporal_cont = temporal_cont * visit_mask

        # ==============================================================
        # temporal categorical
        # ==============================================================
        temporal_cat = {}
        prev_irr = {}

        for name, head in self.temporal_cat_heads.items():
            irrev = self.temporal_cat_irrev[name]

            if irrev:
                logits = head(h_seq).squeeze(-1)              # [B,T]
                hazard = torch.sigmoid(logits)

                irr_states = []
                prev = torch.zeros(B, device=device)

                for t in range(T):
                    if teacher_forcing and real_irr is not None:
                        prev = real_irr[:, t, list(self.temporal_cat_irrev).index(name)]
                    else:
                        flip = torch.bernoulli(hazard[:, t] * (1 - prev))
                        prev = prev + flip

                    irr_states.append(prev)

                irr_states = torch.stack(irr_states, dim=1)   # [B,T]
                one_hot = torch.stack([1 - irr_states, irr_states], dim=-1)
                temporal_cat[name] = one_hot * visit_mask

            else:
                logits = head(h_seq)
                y = F.gumbel_softmax(
                    logits, tau=temperature, hard=True, dim=-1
                )
                temporal_cat[name] = y * visit_mask

        return {
            "static_cont": static_cont,
            "static_cat": static_cat if static_cat else None,
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont": temporal_cont,
            "temporal_cat":  temporal_cat,
            "visit_mask":    visit_mask,
            "visit_times":   visit_times,
        }

