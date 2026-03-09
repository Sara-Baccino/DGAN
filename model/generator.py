"""
model/generator.py
================================================================================
[v9] Rumore temporale strutturato + curriculum learning sulla visit_mask

CAMBIAMENTI RISPETTO A v8:

  1. StructuredTemporalNoise (NUOVO):
     Sostituisce z_temporal iid con una decomposizione a tre componenti:

       z_t[b,t] = sigma_g * z_global[b]      # identità paziente (costante nel tempo)
                + sigma_r * z_ar[b,t]         # trend correlato AR(1), rho appreso
                + sigma_e * epsilon[b,t]      # rumore di visita (iid)

     - z_global: cattura le caratteristiche temporali invarianti del paziente
                 (es. velocità di progressione, livello basale dei biomarker).
     - z_ar:     cattura i trend a medio termine (es. risposta al trattamento).
                 rho è un parametro appreso ∈ (0,1) via sigmoid.
     - epsilon:  cattura la variabilità visita-per-visita (es. fluttuazioni lab).

     Questo è il modello più generale possibile per dati longitudinali:
     - Progressione lenta (PBC, ADNI):       rho → 0.85-0.95, sigma_g > sigma_e
     - Progressione episodica (scompenso):   rho → 0.5-0.7,  sigma_e > sigma_g
     - Progressione mista (diabete, BPCO):   rho → 0.7-0.8,  sigma_r ≈ sigma_g

     I pesi sigma_* sono parametri appresi tramite softplus (>0) e normalizzati
     per mantenere Var(z_t) ≈ 1 (fondamentale per stabilità del GRU/LSTM).

  2. Curriculum learning sulla visit_mask (NUOVO):
     Problema con fixed_visits=real_n_visits in training:
     - Il discriminatore vede sempre sequenze della lunghezza corretta.
     - n_visits_head non riceve gradiente end-to-end dalla visit_mask.
     - A inference time, n_visits_head ha imparato male → crollo qualità.

     Soluzione: schedule lineare da "100% reale" a "100% predetto":
       mask_source = "real"  se epoch < warmup_mask_epochs
       mask_source = mix(real, pred, p=curriculum_p(epoch))  nelle epoche intermedie
       mask_source = "pred"  se epoch >= total_epochs - finetune_mask_epochs

     curriculum_p(epoch): da 0.0 a 1.0 linearmente nella finestra intermedia.
     Controllato da HierarchicalGenerator.set_curriculum_p(p).

  3. TimeEncoder v8 invariato (rescaling su slot attivi).

  4. Retrocompatibilità completa:
     - StructuredTemporalNoise è usato da DGAN._generate_fake().
     - HierarchicalGenerator.forward() invariato nell'interfaccia.
     - Il curriculum è gestito da DGAN.fit() via generator.set_curriculum_p().
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ==================================================================
# STRUCTURED TEMPORAL NOISE
# ==================================================================

class StructuredTemporalNoise(nn.Module):
    """
    Decomposizione del rumore temporale in tre componenti ortogonali.

    MOTIVAZIONE:
      z_temporal iid per ogni slot temporale costringe il backbone (GRU/LSTM)
      a costruire correlazione temporale partendo da input scorrelati.
      Questo porta a traiettorie piatte (slope≈0) perché il GRU "media" il
      rumore producendo output smooth ma senza varianza individuale.

    MODELLO:
      z_t[b,t] = sigma_g * z_global[b]   +   sigma_r * z_ar[b,t]   +   sigma_e * eps[b,t]
                 ─────────────────────       ───────────────────────   ─────────────────────
                 identità paziente           trend correlato AR(1)     rumore visita (iid)

    PARAMETRI APPRESI (tutti ∈ ℝ, poi trasformati):
      log_rho:      rho = sigmoid(log_rho) ∈ (0,1)  — persistenza AR(1)
      log_sigma_*:  sigma_* = softplus(log_sigma_*) — pesi componenti

    NORMALIZZAZIONE:
      I sigma vengono normalizzati per garantire Var(z_t) ≈ 1:
        Var(z_t) = sigma_g² + sigma_r² * Var(z_ar) + sigma_e²
      dove Var(z_ar) ≈ 1/(1-rho²) per AR(1) stazionario.
      Senza normalizzazione, il GRU riceve input con varianza variabile
      durante il training → instabilità.

    GENERALITY:
      Funziona per qualsiasi dataset longitudinale. rho appreso si adatta
      automaticamente alla velocità di progressione della patologia.
      - PBC, ADNI (progressione lenta):      rho converge verso 0.85-0.95
      - Scompenso, sepsi (episodico):         rho converge verso 0.5-0.7
      - Diabete, BPCO (misto):               rho converge verso 0.7-0.8
    """

    def __init__(self, z_dim: int):
        super().__init__()
        self.z_dim = z_dim

        # rho: persistenza AR(1), init sigmoid(1.5) ≈ 0.82
        self.log_rho = nn.Parameter(torch.tensor(1.5))

        # Pesi delle tre componenti (tutti init a softplus(1.0) ≈ 1.31)
        self.log_sigma_g = nn.Parameter(torch.tensor(1.0))  # global
        self.log_sigma_r = nn.Parameter(torch.tensor(1.0))  # ar trend
        self.log_sigma_e = nn.Parameter(torch.tensor(0.0))  # episodic (init più basso)

    def forward(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """
        Campiona z_temporal strutturato [B, T, z_dim].

        Args:
            B: batch size
            T: sequenza massima (max_len)
            device: device target

        Returns:
            z_t: [B, T, z_dim] con struttura temporale AR(1) + global + iid
        """
        rho     = torch.sigmoid(self.log_rho)                    # ∈ (0,1)
        sigma_g = F.softplus(self.log_sigma_g)
        sigma_r = F.softplus(self.log_sigma_r)
        sigma_e = F.softplus(self.log_sigma_e)

        # ── Componente globale: costante nel tempo per paziente ───────
        z_global = torch.randn(B, self.z_dim, device=device)    # [B, D]
        z_global_seq = z_global.unsqueeze(1).expand(-1, T, -1)  # [B, T, D]

        # ── Componente AR(1): trend correlato ─────────────────────────
        # z_ar[t] = rho * z_ar[t-1] + sqrt(1-rho²) * eps[t]
        # Var(z_ar) = 1 per stazionarietà (verifica: Var = sigma_eps² / (1-rho²) = 1)
        noise_scale = torch.sqrt(1.0 - rho ** 2 + 1e-8)
        eps_ar      = torch.randn(B, T, self.z_dim, device=device)
        z_ar        = torch.zeros(B, T, self.z_dim, device=device)
        # Stato iniziale: campionato dalla distribuzione stazionaria N(0,1)
        z_ar_t      = torch.randn(B, self.z_dim, device=device)
        for t in range(T):
            z_ar_t    = rho * z_ar_t + noise_scale * eps_ar[:, t, :]
            z_ar[:, t, :] = z_ar_t

        # ── Componente episodica: rumore iid per visita ───────────────
        z_eps = torch.randn(B, T, self.z_dim, device=device)

        # ── Varianza totale per normalizzazione ───────────────────────
        # Var(z_t) = sigma_g² * Var(z_global) + sigma_r² * Var(z_ar) + sigma_e² * Var(eps)
        # Var(z_global) = 1 (N(0,1)), Var(z_ar) ≈ 1 per costruzione, Var(eps) = 1
        var_total = sigma_g ** 2 + sigma_r ** 2 + sigma_e ** 2
        norm      = torch.sqrt(var_total + 1e-8)

        z_t = (sigma_g * z_global_seq + sigma_r * z_ar + sigma_e * z_eps) / norm
        return z_t

    def get_stats(self) -> Dict[str, float]:
        """Restituisce statistiche interpretabili per il logging."""
        with torch.no_grad():
            rho     = float(torch.sigmoid(self.log_rho))
            sigma_g = float(F.softplus(self.log_sigma_g))
            sigma_r = float(F.softplus(self.log_sigma_r))
            sigma_e = float(F.softplus(self.log_sigma_e))
            total   = (sigma_g**2 + sigma_r**2 + sigma_e**2) ** 0.5
        return {
            "rho":     rho,
            "w_global": sigma_g / total,
            "w_ar":     sigma_r / total,
            "w_episod": sigma_e / total,
        }


# ==================================================================
# POSITIONAL ENCODING TEMPORALE CONTINUO
# ==================================================================

class ContinuousTemporalPE(nn.Module):
    def __init__(self, hidden_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.n_frequencies = n_frequencies
        freqs = torch.pow(
            2.0, torch.linspace(0, math.log2(max(n_frequencies, 2)), n_frequencies)
        )
        self.register_buffer("freqs", freqs)
        self.pe_proj      = nn.Linear(2 * n_frequencies, hidden_dim)
        self.static_to_pe = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, t_norm: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        angles  = 2.0 * math.pi * t_norm.unsqueeze(-1) * self.freqs
        pe_in   = torch.cat([angles.sin(), angles.cos()], dim=-1)
        pe      = self.pe_proj(pe_in)
        return pe + self.static_to_pe(static_h).unsqueeze(1)


# ==================================================================
# TIME ENCODER  [v8 — rescaling su slot attivi, invariato]
# ==================================================================

class TimeEncoder(nn.Module):
    """
    [v8] Rescaling sui soli slot attivi.

    sum(delta_months[slot attivi]) == D3_fup_months per costruzione.
    => t[n_visits-1] == D3_fup => Cov ≈ 0 automaticamente.

    Grad-safe: nessuna assegnazione in-place su tensori con grad.
    """
    def __init__(self, z_temporal_dim: int, hidden_dim: int,
                 global_time_max: float = 400.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_temporal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, 2.3)
        gtm = float(global_time_max) if global_time_max and global_time_max > 0 else 400.0
        self.register_buffer("global_time_max", torch.tensor(gtm))

    def forward(
        self,
        z_temporal:   torch.Tensor,
        d3_fup_scale: Optional[torch.Tensor],
        n_visits:     Optional[torch.Tensor],
    ):
        B, T, _ = z_temporal.shape
        device  = z_temporal.device

        delta_raw = F.softplus(self.mlp(z_temporal).squeeze(-1))   # [B,T]

        t_pos = torch.arange(T, dtype=torch.float32, device=device)

        if n_visits is not None:
            n_v         = n_visits.float().clamp(1.0, float(T))
            active_mask = (t_pos.unsqueeze(0) < n_v.unsqueeze(1)).float()
        else:
            active_mask = torch.ones(B, T, device=device)

        slot0_mask   = (t_pos > 0).float().unsqueeze(0)
        delta_active = delta_raw * active_mask * slot0_mask         # [B,T]

        if d3_fup_scale is not None:
            d3_fup_months = d3_fup_scale.float().clamp(0.01, 1.0) * self.global_time_max
        else:
            d3_fup_months = torch.full(
                (B,), float(self.global_time_max) * 0.35, device=device)

        sum_active   = delta_active[:, 1:].sum(dim=1).clamp(min=1e-8)
        scale_factor = (d3_fup_months / sum_active).unsqueeze(1)
        delta_months = delta_active * scale_factor                  # [B,T] mesi

        t_months     = torch.cumsum(delta_months, dim=1)
        t_norm       = (t_months / self.global_time_max).clamp(max=1.5)
        delta_t_norm = torch.cat([
            torch.zeros(B, 1, device=device),
            t_norm[:, 1:] - t_norm[:, :-1],
        ], dim=1)
        t_feat = torch.stack([t_norm, delta_t_norm], dim=-1)

        return t_norm, delta_t_norm, t_feat, t_months, delta_months


# ==================================================================
# TEMPORAL BACKBONES
# ==================================================================

class GRUBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.static_to_h0 = nn.Linear(hidden_dim, n_layers * hidden_dim)

    def forward(self, x: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        T  = x.shape[1]
        s  = static_h.unsqueeze(1).expand(-1, T, -1)
        x  = self.input_proj(torch.cat([x, s], dim=-1))
        h0 = self.static_to_h0(static_h)
        h0 = h0.view(-1, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        h, _ = self.gru(x, h0)
        return h


class LSTMBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.static_to_h0 = nn.Linear(hidden_dim, 2 * n_layers * hidden_dim)

    def forward(self, x: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        x   = self.input_proj(x)
        hc  = self.static_to_h0(static_h).view(-1, 2 * self.n_layers, self.hidden_dim)
        h0  = hc[:, :self.n_layers].permute(1, 0, 2).contiguous()
        c0  = hc[:, self.n_layers:].permute(1, 0, 2).contiguous()
        h, _ = self.lstm(x, (h0, c0))
        return h


class TransformerBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int,
                 n_heads: int, dropout: float, pe_frequencies: int):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.temporal_pe = ContinuousTemporalPE(hidden_dim, pe_frequencies)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        t_norm = x[:, :, -2]
        h      = self.input_proj(x)
        h      = h + self.temporal_pe(t_norm, static_h) + static_h.unsqueeze(1)
        return self.transformer(h)


# ==================================================================
# HIERARCHICAL GENERATOR  [v9]
# ==================================================================

class HierarchicalGenerator(nn.Module):
    """
    [v9] Aggiunge:
      - StructuredTemporalNoise: modulo che vive nel generator e produce
        z_temporal strutturato. I suoi parametri (rho, sigma_*) sono
        ottimizzati end-to-end col resto del generatore.
      - set_curriculum_p(p): controlla la probabilità di usare n_visits_pred
        invece di n_visits_real per la visit_mask. Chiamato da DGAN.fit().
        p=0.0 → sempre reale (warmup), p=1.0 → sempre predetto (fine training).
    """

    def __init__(
        self,
        data_config,
        preprocessor,
        z_static_dim:         int   = 64,
        z_temporal_dim:       int   = 32,
        hidden_dim:           int   = 128,
        backbone:             str   = "gru",
        rnn_layers:           int   = 2,
        dropout:              float = 0.1,
        n_visits_sharpness:   float = 10.0,
        n_transformer_layers: int   = 2,
        n_heads:              int   = 4,
        pe_frequencies:       int   = 16,
        gru_layers:           int   = None,
    ):
        super().__init__()
        self.data_config        = data_config
        self.preprocessor       = preprocessor
        self.max_len            = data_config.max_len
        self.hidden_dim         = hidden_dim
        self.z_temporal_dim     = z_temporal_dim
        self.n_visits_sharpness = n_visits_sharpness
        self.backbone_name      = backbone

        # Curriculum: prob di usare n_visits_pred per la mask [0,1]
        # Aggiornato da DGAN.fit() via set_curriculum_p()
        self._curriculum_p = 0.0

        if gru_layers is not None:
            rnn_layers = gru_layers

        # ── Structured temporal noise [v9] ────────────────────────────
        self.noise_model = StructuredTemporalNoise(z_dim=z_temporal_dim)

        # ── Time encoder [v8] ─────────────────────────────────────────
        gtm = float(getattr(preprocessor, "global_time_max", 400.0) or 400.0)
        self.time_encoder = TimeEncoder(z_temporal_dim, hidden_dim, global_time_max=gtm)

        # ── Static branch ─────────────────────────────────────────────
        self.fc_static = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
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

        # ── Temporal backbone ─────────────────────────────────────────
        input_dim = z_temporal_dim + 2
        if backbone == "gru":
            self.backbone = GRUBackbone(input_dim, hidden_dim, rnn_layers, dropout)
        elif backbone == "lstm":
            self.backbone = LSTMBackbone(input_dim, hidden_dim, rnn_layers, dropout)
        elif backbone == "transformer":
            self.backbone = TransformerBackbone(
                input_dim, hidden_dim, n_transformer_layers, n_heads, dropout, pe_frequencies)
        else:
            raise ValueError(f"backbone non valido: {backbone!r}")

        # ── Output statici ────────────────────────────────────────────
        self.static_cont_head = None
        if data_config.n_static_cont > 0:
            self.static_cont_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, data_config.n_static_cont),
            )
            nn.init.zeros_(self.static_cont_head[-1].weight)
            nn.init.zeros_(self.static_cont_head[-1].bias)

        self.static_cat_heads = nn.ModuleDict()
        for v in data_config.static_cat:
            if v.name in preprocessor.embedding_configs:
                self.static_cat_heads[v.name] = nn.Linear(
                    hidden_dim, preprocessor.embedding_configs[v.name])
            else:
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

        # ── Output temporali ──────────────────────────────────────────
        self.temporal_cont_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, data_config.n_temp_cont),
        )
        nn.init.zeros_(self.temporal_cont_head[-1].weight)
        nn.init.zeros_(self.temporal_cont_head[-1].bias)

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

    # ─────────────────────────────────────────────────────────────────

    def set_curriculum_p(self, p: float):
        """
        Imposta la probabilità curriculum per la visit_mask.

        p=0.0: usa sempre n_visits reale (warmup, discriminatore non vede errori di lunghezza)
        p=1.0: usa sempre n_visits_pred (fine training, end-to-end)

        Chiamato da DGAN.fit() ad ogni epoca.
        """
        self._curriculum_p = float(max(0.0, min(1.0, p)))

    def sample_noise(self, B: int, device: torch.device):
        """
        Campiona z_static e z_temporal strutturato.
        Usato da DGAN._generate_fake() invece di campionare z_t direttamente.

        Returns:
            z_s: [B, z_static_dim]
            z_t: [B, T, z_temporal_dim]  — strutturato con StructuredTemporalNoise
        """
        z_s = torch.randn(B, self.data_config.z_static_dim
                          if hasattr(self.data_config, "z_static_dim")
                          else self.fc_static[0].in_features,
                          device=device)
        z_t = self.noise_model(B, self.max_len, device)
        return z_s, z_t

    @staticmethod
    def _cummax_irreversible(hazard: torch.Tensor) -> torch.Tensor:
        states, _ = torch.cummax(hazard, dim=1)
        return states

    def _soft_visit_mask(self, n_v: torch.Tensor) -> torch.Tensor:
        T      = self.max_len
        device = n_v.device
        t_pos  = torch.arange(T, dtype=torch.float32, device=device)
        logits = self.n_visits_sharpness * (n_v.unsqueeze(-1) - 0.5 - t_pos.unsqueeze(0))
        return torch.sigmoid(logits).unsqueeze(-1)

    def _build_visit_mask(
        self,
        n_v_pred:     torch.Tensor,
        fixed_visits: Optional[torch.Tensor],
        hard_mask:    bool,
        device:       torch.device,
    ):
        """
        Costruisce visit_mask e n_v applicando il curriculum learning.

        Logica:
          1. Se fixed_visits è fornito (da DGAN con real_n_visits):
             - Con prob (1 - curriculum_p): usa fixed_visits (reale)
             - Con prob curriculum_p:       usa n_v_pred (predetto)
          2. Se fixed_visits è None:
             - hard_mask=True: n_v_pred arrotondato
             - hard_mask=False: soft mask differenziabile

        In entrambi i casi la mask è differenziabile rispetto a n_v_pred
        quando si usa il ramo predetto, permettendo gradiente end-to-end.
        """
        T = self.max_len
        t_pos = torch.arange(T, dtype=torch.float32, device=device)

        if fixed_visits is not None:
            n_v_real = fixed_visits.to(device).float().clamp(2.0, float(T))

            # Curriculum: decide per ogni campione del batch se usare reale o predetto
            if self._curriculum_p > 0.0 and self.training:
                # Maschera booleana: True → usa predetto, False → usa reale
                use_pred = (torch.rand(n_v_real.shape[0], device=device)
                            < self._curriculum_p)
                # n_v misto: reale o predetto per ogni campione
                n_v_mix  = torch.where(use_pred, n_v_pred.clamp(2.0, float(T)), n_v_real)
            else:
                n_v_mix = n_v_real

            # Soft mask differenziabile per i campioni con n_v predetto
            visit_mask = self._soft_visit_mask(n_v_mix)
            n_v        = n_v_mix

        elif hard_mask:
            n_v_hard   = n_v_pred.round().clamp(2.0, float(T))
            visit_mask = (n_v_hard.unsqueeze(-1) > t_pos.unsqueeze(0)).float().unsqueeze(-1)
            n_v        = n_v_hard
        else:
            visit_mask = self._soft_visit_mask(n_v_pred)
            n_v        = n_v_pred

        return visit_mask, n_v

    # ─────────────────────────────────────────────────────────────────

    def forward(
        self,
        z_static:           torch.Tensor,
        z_temporal:         torch.Tensor,
        temperature:        float                   = 1.0,
        teacher_forcing:    bool                    = False,
        real_irr:           Optional[torch.Tensor] = None,
        hard_mask:          bool                    = False,
        fixed_visits:       Optional[torch.Tensor] = None,
        real_followup_norm: Optional[torch.Tensor] = None,
    ) -> Dict:
        B, T, _ = z_temporal.shape
        device   = z_temporal.device

        # 1. Follow-up prediction
        followup_scale = self.followup_head(z_static).squeeze(-1)   # [B] ∈ [0,1]

        # 2. n_visits predetto
        n_v_raw  = F.softplus(self.n_visits_head(z_static).squeeze(-1)) + 1.0
        n_v_pred = n_v_raw.clamp(1.0, float(T))

        # 3. Visit mask con curriculum [v9]
        visit_mask, n_v = self._build_visit_mask(
            n_v_pred=n_v_pred, fixed_visits=fixed_visits,
            hard_mask=hard_mask, device=device)

        # 4. Time encoding [v8]: rescaling su slot attivi
        t_norm, delta_t, t_feat, t_months, delta_months = self.time_encoder(
            z_temporal, d3_fup_scale=followup_scale, n_visits=n_v)

        # 5. Static encoding
        static_h = self.fc_static(z_static)   # [B, H]

        # 6. Backbone temporale
        backbone_in = torch.cat([z_temporal, t_feat], dim=-1)
        h_seq       = self.backbone(backbone_in, static_h)

        # 7. Output statici
        static_cont     = None
        static_cat      = {}
        static_cat_soft = {}
        static_cat_embed= {}
        embed_cfg       = self.preprocessor.embedding_configs

        if self.static_cont_head is not None:
            static_cont = self.static_cont_head(static_h)

        for name, head in self.static_cat_heads.items():
            out = head(static_h)
            if name in embed_cfg:
                static_cat_embed[name] = out.contiguous()
            else:
                static_cat[name]      = F.gumbel_softmax(out, tau=temperature, hard=True,  dim=-1)
                static_cat_soft[name] = F.gumbel_softmax(out, tau=temperature, hard=False, dim=-1)

        # 8. Output temporali
        temporal_cont = self.temporal_cont_head(h_seq) * visit_mask

        temporal_cat    = {}
        irrev_var_names = [n for n, irr in self.temporal_cat_irrev.items() if irr]

        for name, head in self.temporal_cat_heads.items():
            if self.temporal_cat_irrev[name]:
                hazard = torch.sigmoid(head(h_seq).squeeze(-1))
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
            "static_cont":        static_cont,
            "static_cat":         static_cat       if static_cat       else None,
            "static_cat_soft":    static_cat_soft  if static_cat_soft  else {},
            "static_cat_embed":   static_cat_embed if static_cat_embed else None,
            "temporal_cont":      temporal_cont,
            "temporal_cat":       temporal_cat,
            "visit_mask":         visit_mask,
            "visit_times":        t_norm,
            "visit_times_months": t_months,
            "delta_months":       delta_months,
            "followup_norm":      followup_scale,
            "n_visits":           n_v,
            "n_visits_pred":      n_v_pred,
        }