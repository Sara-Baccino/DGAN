"""
model/generator.py  [gretel-style v3 — LSTM + channel min/max]
================================================================================
Cambiamenti rispetto alla versione GRU:

  [NUOVO] LSTM al posto di GRU
    L'LSTM ha due stati nascosti (h, c): la cell-state c porta memoria a lungo
    raggio (utile per traiettorie cliniche lunghe), mentre h è l'output di ogni
    passo. Questo riduce il "forgetting" precoce dei biomarker iniziali.
    to_h0 ora produce sia h0 che c0: [n_layers*H*2] → split in due metà.

  [NUOVO] Channel min/max clamping per le feature continue temporali
    Dopo la head continua, ogni feature viene clampata ai valori
    [channel_min[j], channel_max[j]] appresi dai dati reali durante il fit.
    Questo evita la generazione di valori fuori scala senza normalizzare
    globalmente il tempo (il tempo segue la propria logica via followup_norm).
    I bound sono registrati come buffer (non parametri) → salvati nel checkpoint.

    Flusso:
      temporal_cont_head(h_seq) → raw_out → clamp(channel_min, channel_max)

  [INVARIATO]
    - Ramo statico s_h → h0/c0, followup_head, n_visits_head, static outputs
    - Proiezione s_h_proj concatenata all'input LSTM ogni step
    - Self-Attention post-LSTM su h_seq
    - Gumbel-Softmax per categoriche (statiche e temporali)
    - cummax per variabili irreversibili
    - Noise AR configurabile via noise_ar_rho
    - valid_flag da n_visits_head
    - Vincolo t_FUP via ratio-scaling dei deltas

================================================================================
API pubblica (invariata rispetto alla v2 GRU):
  DGANGenerator(data_config, preprocessor, ...)
  .set_channel_bounds(channel_min, channel_max)   ← [NUOVO] chiamato da DGAN.fit()
  .sample_noise(batch_size, device) → (z_s, z_t)
  .forward(z_static, z_temporal, temperature, real_irr) → dict
================================================================================
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DGANGenerator(nn.Module):
    """Generatore DGAN con LSTM e channel min/max clamping."""

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim:    int   = 64,
        z_temporal_dim:  int   = 48,
        hidden_dim:      int   = 64,
        n_layers:        int   = 2,
        dropout:         float = 0.1,
        noise_ar_rho:    float = 0.0,
        min_visits:      int   = 2,
        static_proj_dim: int   = 16,
        attn_heads:      int   = 4,
        device                 = None,
    ):
        super().__init__()
        self.data_config     = data_config
        self.preprocessor    = preprocessor
        self.max_len         = data_config.max_len
        self.hidden_dim      = hidden_dim
        self.z_static_dim    = z_static_dim
        self.z_temporal_dim  = z_temporal_dim
        self.n_layers        = n_layers
        self.noise_ar_rho    = float(noise_ar_rho)
        self.min_visits      = max(1, int(min_visits))
        self.static_cond_dim = static_proj_dim
        self.attn_heads      = attn_heads
        self.device          = device if device is not None else torch.device("cpu")

        n_cont = data_config.n_temp_cont

        # ── [1] Ramo statico ──────────────────────────────────────────
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        # [LSTM] h0 + c0: ogni layer ha un hidden e un cell state
        self.to_h0 = nn.Linear(hidden_dim, n_layers * hidden_dim * 2)

        # ── t_FUP dal ramo statico ─────────────────────────────────────
        self.followup_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.followup_head[-2].bias, -0.62)

        # ── n_visits dal ramo statico ──────────────────────────────────
        self.n_visits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.n_visits_head[-1].bias, 4.0)

        # ── Static outputs ─────────────────────────────────────────────
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
                        f"Categorica statica '{v.name}' ha mapping invalido "
                        f"(n_categories={v.n_categories})."
                    )
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

        # ── Proiezione statica per conditioning a ogni step LSTM ───────
        self.static_cond_proj = nn.Linear(hidden_dim, static_proj_dim)

        # ── Categorie temporali per autoregression ─────────────────────
        self._temp_cat_sizes: Dict[str, int] = {}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.name not in [v.name for v in data_config.temporal_cat]:
                continue
            if var.mapping and len(var.mapping) >= 2:
                self._temp_cat_sizes[var.name] = (
                    2 if var.irreversible else len(var.mapping)
                )

        total_cat_prev_dim = sum(self._temp_cat_sizes.values())
        self._n_cont_input = max(n_cont, 0)

        # ── [LSTM] Temporal LSTM ───────────────────────────────────────
        # Input: [z_t, x_prev_cont, cat_prev_ohe, delta_prev, s_h_proj]
        lstm_input_size = (
            z_temporal_dim
            + self._n_cont_input
            + total_cat_prev_dim
            + 1                    # delta_prev
            + static_proj_dim      # proiezione statica
        )
        self.lstm = nn.LSTM(
            input_size  = lstm_input_size,
            hidden_size = hidden_dim,
            num_layers  = n_layers,
            batch_first = True,
            dropout     = dropout if n_layers > 1 else 0.0,
        )

        # ── [5] Self-Attention post-LSTM ──────────────────────────────
        if attn_heads > 0:
            actual_heads = attn_heads
            while hidden_dim % actual_heads != 0 and actual_heads > 1:
                actual_heads -= 1
            self.self_attn = nn.MultiheadAttention(
                embed_dim   = hidden_dim,
                num_heads   = actual_heads,
                dropout     = dropout,
                batch_first = True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)
        else:
            self.self_attn = None
            self.attn_norm = None

        # ── Output temporali continui ──────────────────────────────────
        out_cont_dim = max(n_cont, 1)
        self.temporal_cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_cont_dim),
        )
        nn.init.zeros_(self.temporal_cont_head[-1].weight)
        nn.init.zeros_(self.temporal_cont_head[-1].bias)

        # ── [NUOVO] Channel min/max buffers ───────────────────────────
        # Inizializzati a (-inf, +inf) → nessun clamping finché non vengono
        # impostati da set_channel_bounds() durante il fit.
        # Registrati come buffer: seguono .to(device) e vengono salvati nel checkpoint.
        self.register_buffer(
            "channel_min",
            torch.full((out_cont_dim,), float("-inf"))
        )
        self.register_buffer(
            "channel_max",
            torch.full((out_cont_dim,), float("+inf"))
        )

        # ── Intervallo inter-visita Δt ─────────────────────────────────
        self.interval_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        nn.init.constant_(self.interval_head[-1].bias, 0.5)

        # ── Categoriche temporali ──────────────────────────────────────
        self.temporal_cat_heads = nn.ModuleDict()
        self.temporal_cat_irrev: Dict[str, bool] = {}
        for var in preprocessor.vars:
            if var.static or var.kind != "categorical":
                continue
            if var.name not in [v.name for v in data_config.temporal_cat]:
                continue
            if var.mapping is None or len(var.mapping) < 2:
                raise ValueError(
                    f"Categorica temporale '{var.name}' ha mapping invalido."
                )
            if var.irreversible:
                self.temporal_cat_heads[var.name] = nn.Linear(hidden_dim, 1)
                self.temporal_cat_irrev[var.name] = True
            else:
                self.temporal_cat_heads[var.name] = nn.Linear(
                    hidden_dim, len(var.mapping))
                self.temporal_cat_irrev[var.name] = False

    # ------------------------------------------------------------------
    # API PUBBLICA AGGIUNTIVA
    # ------------------------------------------------------------------

    def set_channel_bounds(
        self,
        channel_min: torch.Tensor,   # [n_cont]
        channel_max: torch.Tensor,   # [n_cont]
    ) -> None:
        """
        Imposta i limiti [min, max] per ogni feature continua temporale.

        Tipicamente chiamato da DGAN.fit() dopo aver calcolato i bound
        dal dataset reale (es. min/max per-feature su tutti gli step validi).
        I bound vengono copiati nei buffer e NON richiedono gradient.

        Nota: il tempo (visit_times) NON è una feature continua in temporal_cont
        e NON viene gestito qui. La normalizzazione temporale è già garantita
        dalla struttura ratio-scaling del generatore (vedi Step 5 del forward).
        """
        n = self.channel_min.shape[0]
        if channel_min.shape[0] != n or channel_max.shape[0] != n:
            raise ValueError(
                f"channel_min/max devono avere shape [{n}], "
                f"ricevuto {channel_min.shape} / {channel_max.shape}."
            )
        self.channel_min.copy_(channel_min.to(self.channel_min.device))
        self.channel_max.copy_(channel_max.to(self.channel_max.device))

    # ------------------------------------------------------------------

    @staticmethod
    def _cummax_irr(hazard: torch.Tensor) -> torch.Tensor:
        """Garantisce monotonia crescente per gli stati irreversibili."""
        states, _ = torch.cummax(hazard, dim=1)
        return states

    def sample_noise(self, batch_size: int, device: torch.device) -> tuple:
        """
        Campiona rumore latente (z_s, z_t).

        Se noise_ar_rho > 0, z_t segue un processo AR(1):
            z_t[t] = rho * z_t[t-1] + sqrt(1-rho^2) * eps
        Questo introduce correlazione temporale nel rumore di input,
        aiutando il generatore a produrre traiettorie più smooth.
        """
        z_s = torch.randn(batch_size, self.z_static_dim, device=device)
        if self.noise_ar_rho <= 0.0:
            z_t = torch.randn(
                batch_size, self.max_len, self.z_temporal_dim, device=device)
        else:
            rho       = self.noise_ar_rho
            std_innov = (1.0 - rho ** 2) ** 0.5
            z_t       = torch.zeros(
                batch_size, self.max_len, self.z_temporal_dim, device=device)
            z_t[:, 0, :] = torch.randn(
                batch_size, self.z_temporal_dim, device=device)
            for t in range(1, self.max_len):
                eps          = torch.randn(
                    batch_size, self.z_temporal_dim, device=device)
                z_t[:, t, :] = rho * z_t[:, t - 1, :] + std_innov * eps
        return z_s, z_t

    # ------------------------------------------------------------------

    def forward(
        self,
        z_static:    torch.Tensor,             # [B, z_s]
        z_temporal:  torch.Tensor,             # [B, T, z_t]
        temperature: float                   = 1.0,
        real_irr:    Optional[torch.Tensor]  = None,
    ) -> Dict:
        B, T, _ = z_temporal.shape
        device   = self.device
        n_cont   = self.data_config.n_temp_cont

        # ── Step 1: Ramo statico ───────────────────────────────────────
        s_h = self.fc_static(z_static)                          # [B, H]

        # [LSTM] Inizializza h0 e c0 separatamente
        h0_flat = self.to_h0(s_h)                               # [B, n_layers*H*2]
        h0_all  = h0_flat.view(B, self.n_layers, self.hidden_dim * 2)
        h0_all  = h0_all.permute(1, 0, 2).contiguous()         # [n_layers, B, H*2]
        h0      = h0_all[:, :, :self.hidden_dim].contiguous()  # [n_layers, B, H]
        c0      = h0_all[:, :, self.hidden_dim:].contiguous()  # [n_layers, B, H]

        # Static outputs da s_h
        static_cont = None
        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(s_h)

        static_cat       = {}
        static_cat_soft  = {}
        static_cat_embed = {}
        embed_cfg        = self.preprocessor.embedding_configs

        for name, head in self.static_cat_heads.items():
            out = head(s_h)
            if name in embed_cfg:
                static_cat_embed[name] = out
            else:
                y_soft = F.gumbel_softmax(out, tau=temperature, hard=False, dim=-1)
                index  = y_soft.argmax(dim=-1, keepdim=True)
                cols   = torch.arange(y_soft.shape[-1], device=device).reshape(1, -1)
                y_hard = (index == cols).float()
                static_cat[name]      = y_hard - y_soft.detach() + y_soft
                static_cat_soft[name] = F.gumbel_softmax(out, tau=temperature,
                                                          hard=False, dim=-1)

        followup_norm = self.followup_head(s_h).squeeze(-1)     # [B] ∈ [0,1]
        n_v_raw       = F.softplus(self.n_visits_head(s_h).squeeze(-1)) + 1.0
        n_visits      = n_v_raw.clamp(float(self.min_visits), float(T))

        t_pos      = torch.arange(T, dtype=torch.float32, device=device)
        n_v_round  = n_visits.round().long().clamp(self.min_visits, T)
        valid_flag = t_pos.unsqueeze(0) < n_v_round.float().unsqueeze(1)  # [B,T]

        # Proiezione statica per conditioning a ogni step
        s_cond = self.static_cond_proj(s_h)                    # [B, static_cond_dim]

        # ── Step 2: Loop AR con LSTM ───────────────────────────────────
        h_seq      = torch.zeros(B, T, self.hidden_dim, device=device)
        deltas_buf = torch.zeros(B, T, device=device)
        cont_buf   = torch.zeros(B, T, max(n_cont, 1), device=device)

        cat_logit_bufs: Dict[str, torch.Tensor] = {}
        for name, n_cat in self._temp_cat_sizes.items():
            cat_logit_bufs[name] = torch.zeros(B, T, n_cat, device=device)

        x_prev_cont  = torch.zeros(B, self._n_cont_input, device=device)
        cat_prev_ohe = torch.zeros(B, sum(self._temp_cat_sizes.values()), device=device)
        delta_prev   = torch.zeros(B, 1, device=device)

        # [LSTM] stato nascosto inizializzato da ramo statico
        hidden = (h0, c0)

        for t in range(T):
            components = [z_temporal[:, t, :], x_prev_cont,
                          cat_prev_ohe, delta_prev, s_cond]
            lstm_in = torch.cat(
                [c.to(device).float() for c in components], dim=-1
            ).unsqueeze(1)                                      # [B, 1, input_size]

            h_t_3d, hidden = self.lstm(lstm_in, hidden)        # [B, 1, H]
            h_t = h_t_3d.squeeze(1)                            # [B, H]
            h_seq[:, t, :] = h_t

            # Continui
            x_cont_t = (self.temporal_cont_head(h_t)[:, :n_cont]
                        if n_cont > 0 else h_t.new_zeros(B, 0))
            if n_cont > 0:
                cont_buf[:, t, :n_cont] = x_cont_t
                x_prev_cont = x_cont_t.detach()

            # Delta Δt
            # [BASELINE FIX] Il primo step è sempre al baseline (t=0, delta=0).
            # I pazienti reali hanno sempre la prima visita a mese 0.
            # Forziamo delta[0]=0 per costruzione: il generatore non deve impararlo.
            if t == 0:
                delta_t = torch.zeros(B, 1, device=device)
            else:
                delta_t = F.softplus(self.interval_head(h_t))  # [B, 1]
            deltas_buf[:, t] = delta_t.squeeze(-1)
            delta_prev = delta_t.detach()

            # Categoriche temporali
            cat_parts = []
            for name, n_cat in self._temp_cat_sizes.items():
                logit_t_new = self.temporal_cat_heads[name](h_t)
                if self.temporal_cat_irrev.get(name, False):
                    p   = torch.sigmoid(logit_t_new).squeeze(-1)
                    ohe = torch.stack([1 - p, p], dim=-1)
                    cat_logit_bufs[name][:, t, :] = ohe.detach()
                    cat_parts.append(ohe.detach())
                else:
                    cat_logit_bufs[name][:, t, :] = logit_t_new.detach()
                    soft_ohe = F.gumbel_softmax(
                        logit_t_new.detach(), tau=temperature, hard=False, dim=-1)
                    cat_parts.append(soft_ohe)

            if cat_parts:
                cat_prev_ohe = torch.cat(cat_parts, dim=-1)

        # ── Step 3: Self-Attention post-LSTM ──────────────────────────
        if self.self_attn is not None:
            attn_mask_key = ~valid_flag                         # [B, T] bool
            attn_out, _   = self.self_attn(
                h_seq, h_seq, h_seq,
                key_padding_mask = attn_mask_key,
            )
            h_seq = self.attn_norm(h_seq + attn_out)

        # ── Step 4: Output temporali finali con channel clamping ───────
        raw_cont = self.temporal_cont_head(h_seq)               # [B, T, out_cont_dim]

        # [NUOVO] Channel min/max clamping (per-feature, ignorando il tempo)
        # channel_min/max sono buffer [out_cont_dim] → broadcast su [B, T, out_cont_dim]
        # Se non impostati (ancora -inf/+inf), torch.clamp è un no-op.
        temporal_cont = torch.clamp(
            raw_cont,
            min = self.channel_min,
            max = self.channel_max,
        )
        if n_cont == 0:
            temporal_cont = temporal_cont[:, :, :0]
        else:
            temporal_cont = temporal_cont[:, :, :n_cont]

        # Categoriche temporali finali dall'h_seq post-attention
        temporal_cat = {}
        irrev_names  = [n for n, irr in self.temporal_cat_irrev.items() if irr]

        for name, head in self.temporal_cat_heads.items():
            if self.temporal_cat_irrev[name]:
                hazard = torch.sigmoid(head(h_seq).squeeze(-1))
                if real_irr is not None and len(irrev_names) > 0:
                    try:
                        irr_idx    = irrev_names.index(name)
                        irr_states = real_irr[:, :, irr_idx].float()
                    except (ValueError, IndexError):
                        irr_states = self._cummax_irr(hazard)
                else:
                    irr_states = self._cummax_irr(hazard)
                temporal_cat[name] = torch.stack(
                    [1.0 - irr_states, irr_states], dim=-1)
            else:
                temporal_cat[name] = F.gumbel_softmax(
                    head(h_seq), tau=temperature, hard=True, dim=-1)

        # ── Step 5: visit_times con vincolo t_FUP ─────────────────────
        # cumsum dei delta → T_obs = tempo ultima visita valida
        # Se T_obs > t_FUP: scala tutti i delta per t_FUP/T_obs  (ratio ≤ 1)
        # Garantisce visit_last <= t_FUP per costruzione, senza clipping brusco.
        # Il TEMPO non viene clampato via channel_min/max: ha la sua logica.
        cumsum_d = torch.cumsum(deltas_buf, dim=1)              # [B, T]
        last_idx = (n_v_round - 1).clamp(0, T - 1).unsqueeze(1)
        T_obs    = cumsum_d.gather(1, last_idx).squeeze(1).clamp(min=1e-6)
        ratio    = torch.minimum(followup_norm / T_obs, torch.ones_like(followup_norm))
        delta_final = deltas_buf * ratio.unsqueeze(1)           # [B, T]
        visit_times = (torch.cumsum(delta_final, dim=1) /
                       followup_norm.clamp(min=1e-6).unsqueeze(1)).clamp(0.0, 1.0)

        return {
            "static_cont":      static_cont,
            "static_cat":       static_cat      if static_cat      else None,
            "static_cat_soft":  static_cat_soft if static_cat_soft else {},
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont":    temporal_cont,
            "temporal_cat":     temporal_cat,
            "valid_flag":       valid_flag,          # [B, T] bool
            "visit_times":      visit_times,         # [B, T] ∈ [0,1]
            "deltas":           delta_final,         # [B, T] intervalli scalati
            "followup_norm":    followup_norm,       # [B] ∈ [0,1] — t_FUP
            "n_visits":         n_visits,
            "n_visits_pred":    n_v_raw,
        }