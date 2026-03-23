"""
model/generator.py  [v2-fully-parametrized]
================================================================================
Rispetto alla versione precedente:

  [NUOVO] Rumore AR configurabile (noise_ar_rho da model_config):
    Se noise_ar_rho > 0, il rumore temporale è autocorrelato:
      z_t[:, 0, :] ~ N(0, I)
      z_t[:, t, :] = rho * z_t[:, t-1, :] + sqrt(1 - rho^2) * eps_t
    Raccomandato per dati clinici PBC (biomarker autocorrelati): 0.3–0.5.
    Default 0.0 = Gaussiano puro N(0, I) (comportamento precedente).

    Il rumore AR viene applicato in _sample_noise() e usato in dgan.py
    per generare z_temporal in modo coerente.

  [NUOVO] Tutti i parametri letti da model_config (nessun hardcoded):
    - noise_ar_rho
    - z_static_dim, z_temporal_dim dal config
    - hidden_dim, n_layers, dropout dal subconfig generator

  [INVARIATO]
    - Architettura GRU stile Gretel DGAN
    - valid_flag [B,T] bool
    - followup_head, n_visits_head
    - Gumbel-Softmax per categoriche
    - cummax per variabili irreversibili
================================================================================
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ==================================================================
# GENERATOR
# ==================================================================

class DGANGenerator(nn.Module):
    """
    Generatore stile Gretel DGAN, completamente parametrizzato da config.

    Parametri:
      data_config    : DataConfig (da config_loader)
      preprocessor   : Preprocessor (per embedding_configs e vars)
      z_static_dim   : dim del rumore statico per paziente
      z_temporal_dim : dim del rumore per step temporale
      hidden_dim     : dim hidden GRU e MLP
      n_layers       : strati GRU
      dropout        : dropout nel GRU (solo se n_layers > 1)
      noise_ar_rho   : correlazione AR temporale del rumore (0.0 = puro N(0,I))
    """

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim:   int   = 64,
        z_temporal_dim: int   = 32,
        hidden_dim:     int   = 128,
        n_layers:       int   = 2,
        dropout:        float = 0.1,
        noise_ar_rho:   float = 0.0,
        min_visits:     float = 1
    ):
        super().__init__()
        self.data_config    = data_config
        self.preprocessor   = preprocessor
        self.max_len        = data_config.max_len
        self.min_visits     = min_visits  # da config
        self.hidden_dim     = hidden_dim
        self.z_static_dim   = z_static_dim
        self.z_temporal_dim = z_temporal_dim
        self.n_layers       = n_layers
        self.noise_ar_rho   = float(noise_ar_rho)

        # Validazione
        if z_static_dim < 1:
            raise ValueError(f"z_static_dim deve essere >= 1, ricevuto: {z_static_dim}")
        if z_temporal_dim < 1:
            raise ValueError(f"z_temporal_dim deve essere >= 1, ricevuto: {z_temporal_dim}")
        if not 0.0 <= noise_ar_rho < 1.0:
            raise ValueError(
                f"noise_ar_rho deve essere in [0, 1), ricevuto: {noise_ar_rho}"
            )

        # ── Static branch ──────────────────────────────────────────────
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.Tanh(),
        )
        self.to_h0 = nn.Linear(hidden_dim, n_layers * hidden_dim)

        self.followup_head = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.followup_head[-2].bias, -0.62)

        self.n_visits_head = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.n_visits_head[-1].bias, 6.0)

        # ── Temporal GRU ───────────────────────────────────────────────
        self.gru = nn.GRU(
            input_size  = z_temporal_dim + 1,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        # ── Output statici ─────────────────────────────────────────────
        if data_config.n_static_cont > 0:
            self.static_cont_head = nn.Linear(hidden_dim, data_config.n_static_cont)
            nn.init.zeros_(self.static_cont_head.weight)
            nn.init.zeros_(self.static_cont_head.bias)
        else:
            self.static_cont_head = None

        self.static_cat_heads = nn.ModuleDict()
        for v in data_config.static_cat:
            if v.name in preprocessor.embedding_configs:
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, preprocessor.embedding_configs[v.name])
            else:
                if v.n_categories is None or v.n_categories < 2:
                    raise ValueError(
                        f"Variabile categorica statica '{v.name}' ha mapping invalido "
                        f"(n_categories={v.n_categories}). Controlla il config JSON."
                    )
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

        # ── Output temporali ───────────────────────────────────────────
        if data_config.n_temp_cont < 1:
            warnings.warn(
                "n_temp_cont = 0: nessuna feature continua temporale. "
                "Il temporal_cont_head genererà un tensore vuoto. "
                "Assicurati di avere almeno una variabile continua temporale.",
                UserWarning,
                stacklevel=2,
            )
        out_dim = max(data_config.n_temp_cont, 1)  # evita Linear(H, 0)
        self.temporal_cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        nn.init.zeros_(self.temporal_cont_head[-1].weight)
        nn.init.zeros_(self.temporal_cont_head[-1].bias)

        self.temporal_cat_heads = nn.ModuleDict()
        self.temporal_cat_irrev: Dict[str, bool] = {}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.name not in [v.name for v in data_config.temporal_cat]:
                continue
            if var.mapping is None or len(var.mapping) < 2:
                raise ValueError(
                    f"Variabile categorica temporale '{var.name}' ha mapping invalido "
                    f"(len={len(var.mapping) if var.mapping else 0}). "
                    f"Controlla il config JSON."
                )
            if var.irreversible:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, 1)
                self.temporal_cat_irrev[var.name] = True
            else:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, len(var.mapping))
                self.temporal_cat_irrev[var.name] = False

    # ------------------------------------------------------------------

    def _make_t_norm(
        self,
        n_visits:       torch.Tensor,   # [B] float
        followup_scale: torch.Tensor,   # [B] ∈ [0,1]
    ) -> torch.Tensor:
        """
        Crea t_norm [B, T, 1]: step uniformi riscalati su followup_scale.
        t_norm[b, t] = (t / (n_v[b]-1)) * followup_scale[b]  ∈ [0, followup_scale]
        """
        B, T   = followup_scale.shape[0], self.max_len
        device = followup_scale.device
        t_pos  = torch.arange(T, dtype=torch.float32, device=device)
        n_v    = n_visits.clamp(2.0, float(T)).unsqueeze(1)
        denom  = (n_v - 1.0).clamp(min=1.0)
        t_frac = (t_pos.unsqueeze(0) / denom).clamp(max=1.0)
        t_norm = t_frac * followup_scale.unsqueeze(1)
        return t_norm.unsqueeze(-1)   # [B,T,1]

    @staticmethod
    def _cummax_irr(hazard: torch.Tensor) -> torch.Tensor:
        """Rende monotonicamente crescente (per variabili irreversibili)."""
        states, _ = torch.cummax(hazard, dim=1)
        return states

    def sample_noise(self, batch_size: int, device: torch.device) -> tuple:
        """
        Campiona il rumore latente.
        Ritorna (z_static [B, z_s], z_temporal [B, T, z_t]).

        Se noise_ar_rho > 0, z_temporal è AR(1) correlato nel tempo:
          z_t[:, t, :] = rho * z_t[:, t-1, :] + sqrt(1 - rho^2) * eps_t
        Questo genera traiettorie smooth, adatte a biomarker clinici autocorrelati.
        """
        z_s = torch.randn(batch_size, self.z_static_dim, device=device)

        if self.noise_ar_rho <= 0.0:
            z_t = torch.randn(batch_size, self.max_len, self.z_temporal_dim, device=device)
        else:
            rho      = self.noise_ar_rho
            std_innov = (1.0 - rho ** 2) ** 0.5
            z_t       = torch.zeros(batch_size, self.max_len, self.z_temporal_dim, device=device)
            z_t[:, 0, :] = torch.randn(batch_size, self.z_temporal_dim, device=device)
            for t in range(1, self.max_len):
                eps           = torch.randn(batch_size, self.z_temporal_dim, device=device)
                z_t[:, t, :] = rho * z_t[:, t - 1, :] + std_innov * eps

        return z_s, z_t

    # ------------------------------------------------------------------

    def forward(
        self,
        z_static:    torch.Tensor,              # [B, z_s]
        z_temporal:  torch.Tensor,              # [B, T, z_t]
        temperature: float                      = 1.0,
        real_irr:    Optional[torch.Tensor]    = None,
    ) -> Dict:
        B, T, _ = z_temporal.shape
        device   = z_temporal.device

        # 1. Static encoding → h0
        s_h = self.fc_static(z_static)                          # [B, H]
        h0  = self.to_h0(s_h)                                   # [B, n_layers*H]
        h0  = h0.view(B, self.n_layers, self.hidden_dim) \
                 .permute(1, 0, 2).contiguous()                 # [n_layers, B, H]

        # 2. Scalar predictions
        followup_scale = self.followup_head(z_static).squeeze(-1)            # [B]
        n_v_raw        = F.softplus(self.n_visits_head(z_static).squeeze(-1)) + 1.0
        n_visits       = n_v_raw.clamp(1.0, float(T))                        # [B]

        # 3. Time features
        t_norm = self._make_t_norm(n_visits, followup_scale)    # [B,T,1]

        # 4. GRU
        gru_in   = torch.cat([z_temporal, t_norm], dim=-1)      # [B,T, z_t+1]
        h_seq, _ = self.gru(gru_in, h0)                         # [B,T,H]

        # 5. Statici (dal primo step)
        h0_step     = h_seq[:, 0, :]                             # [B,H]
        static_cont = None
        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(h0_step)

        static_cat       = {}
        static_cat_soft  = {}
        static_cat_embed = {}
        embed_cfg        = self.preprocessor.embedding_configs

        for name, head in self.static_cat_heads.items():
            out = head(h0_step)
            if name in embed_cfg:
                static_cat_embed[name] = out
            else:
                static_cat[name]      = F.gumbel_softmax(out, tau=temperature, hard=True,  dim=-1)
                static_cat_soft[name] = F.gumbel_softmax(out, tau=temperature, hard=False, dim=-1)

        # 6. Temporali
        temporal_cont_raw = self.temporal_cont_head(h_seq)       # [B,T, n_cont (o 1)]
        # Ritaglia a n_temp_cont effettivo
        n_cont = self.data_config.n_temp_cont
        temporal_cont = temporal_cont_raw[:, :, :n_cont] if n_cont > 0 else \
                        temporal_cont_raw[:, :, :0]

        temporal_cat = {}
        irrev_names  = [n for n, irr in self.temporal_cat_irrev.items() if irr]

        for name, head in self.temporal_cat_heads.items():
            if self.temporal_cat_irrev[name]:
                hazard = torch.sigmoid(head(h_seq).squeeze(-1))  # [B,T]
                if real_irr is not None and len(irrev_names) > 0:
                    try:
                        irr_idx    = irrev_names.index(name)
                        irr_states = real_irr[:, :, irr_idx].float()
                    except (ValueError, IndexError):
                        irr_states = self._cummax_irr(hazard)
                else:
                    irr_states = self._cummax_irr(hazard)
                temporal_cat[name] = torch.stack([1.0 - irr_states, irr_states], dim=-1)
            else:
                temporal_cat[name] = F.gumbel_softmax(
                    head(h_seq), tau=temperature, hard=True, dim=-1)

        # 7. valid_flag: [B,T] bool — True per i primi n_visits step
        # n_visits è clampato a [min_visits, T] per garantire il minimo configurato
        t_pos      = torch.arange(T, dtype=torch.float32, device=device)
        n_v_round  = n_visits.round().long().clamp(self.min_visits, T)
        valid_flag = t_pos.unsqueeze(0) < n_v_round.float().unsqueeze(1)  # [B,T] bool

        return {
            "static_cont":      static_cont,
            "static_cat":       static_cat      if static_cat      else None,
            "static_cat_soft":  static_cat_soft if static_cat_soft else {},
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont":    temporal_cont,
            "temporal_cat":     temporal_cat,
            "valid_flag":       valid_flag,       # [B,T] bool
            "visit_times":      t_norm.squeeze(-1),
            "followup_norm":    followup_scale,
            "n_visits":         n_visits,
            "n_visits_pred":    n_v_raw,
        }