import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class HierarchicalGenerator(nn.Module):
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

        self.temporal_cont_head = nn.Linear(
            hidden_dim, data_config.n_temp_cont
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

        # ==============================================================
        # temporal continuous
        # ==============================================================
        temporal_cont = self.temporal_cont_head(h_seq)
        temporal_cont = temporal_cont * visit_mask

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

