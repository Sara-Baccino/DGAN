"""
model/generator.py
================================================================================
[v6] Backbone configurabile: GRU | LSTM | Transformer

  SELEZIONE VIA CONFIG JSON:
    "generator": {
      "backbone":           "gru",   # "gru" | "lstm" | "transformer"
      "hidden_dim":          128,
      "rnn_layers":          2,      # GRU / LSTM
      "dropout":             0.1,
      "n_transformer_layers": 2,     # Transformer only
      "n_heads":             4,      # Transformer only
      "pe_frequencies":      16,     # Transformer only
      "z_static_dim":        64,
      "z_temporal_dim":      32
    }

  RACCOMANDAZIONE PER ~800 PAZIENTI, T_MEDIO=7, TRAINING LOCALE:
    backbone="gru", hidden_dim=128, rnn_layers=2
    → ~120k parametri generatore, ~5-10 sec/epoca su CPU
    Transformer con T_medio=7 è sovradimensionato: O(T²) attention
    non porta vantaggi su sequenze così corte ed è 5-10x più lento.

  PERCHÉ GRU (non LSTM):
    - 33% meno parametri a parità di hidden_dim
    - Nessun vantaggio documentato di LSTM per T<20
    - Converge più velocemente su sequenze corte

  ARCHITETTURA GRU:
    z_static → fc_static → static_h [B,H]
                          → followup_head → followup_norm [B]
                          → n_visits_head → n_v_pred [B]

    z_temporal [B,T,Zt] || t_feat [B,T,2]
      → input_proj [B,T,H]
      → GRU con h_0 = fc_static_to_h0(static_h)   [B,T,H]

    h_seq → temporal_cont_head (output lineare, z-score compatibile)
    static_h → static_cont_head

  NOTA IMPORTANTE: n_visits_pred ≠ n_visits
    - n_visits_pred: output di n_visits_head, usato dalla supervision loss
    - n_visits: quello che determina la visit_mask (= fixed_visits in training)
    I due vengono separati per permettere il gradiente su n_visits_head
    anche quando fixed_visits è fornito (altrimenti NvL = 0 sempre).
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# ==================================================================
# POSITIONAL ENCODING TEMPORALE CONTINUO  (usato solo da Transformer)
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
        angles      = 2.0 * math.pi * t_norm.unsqueeze(-1) * self.freqs
        pe_in       = torch.cat([angles.sin(), angles.cos()], dim=-1)
        pe          = self.pe_proj(pe_in)
        static_bias = self.static_to_pe(static_h).unsqueeze(1)
        return pe + static_bias


# ==================================================================
# TIME ENCODER
# ==================================================================

class TimeEncoder(nn.Module):
    """
    Produce t_norm ∈ [0,1] e delta_t da z_temporal.

    [v5.1] Prima visita sempre a t=0: delta_raw[:,0] = 0.

    [v6.1] Ancoraggio a D3_fup via d3_fup_scale (opzionale).
      - d3_fup_scale ∈ [0,1]: follow-up normalizzato del paziente
        (D3_fup_reale / D3_fup_max, come prodotto dal preprocessor)
      - Se fornito: t_norm[-1] ≈ d3_fup_scale → nessuna visita
        sintetica supera il follow-up reale del paziente
      - In training: passato come real_followup_norm dal batch reale
      - In inference: campionato da followup_head(z_static)

    PERCHÉ SERVE:
      Senza ancoraggio, il TimeEncoder genera t_norm ∈ [0,1] per tutti
      i pazienti indipendentemente da D3_fup. Un paziente con 6 mesi
      di follow-up reale ottiene visite sintetiche fino a mese 36
      (se D3_fup_max=36). Questo causa:
        1. KS(D3_fup) = 0.68 — distribuzione del follow-up completamente
           sbagliata (il sintetico concentra tutto a 150-200 mesi)
        2. Traiettorie temporali che esplodono dopo il mese 15 — perché
           il discriminatore vede solo i primi mesi reali ma il generatore
           continua a produrre valori oltre quell'orizzonte
    """
    def __init__(self, z_temporal_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(z_temporal_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        z_temporal:   torch.Tensor,
        d3_fup_scale: Optional[torch.Tensor] = None,   # [B] ∈ [0,1]
    ):
        delta_raw        = F.softplus(self.mlp(z_temporal).squeeze(-1))
        delta_raw        = delta_raw.clone()
        delta_raw[:, 0]  = 0.0
        t_cumul          = torch.cumsum(delta_raw, dim=1)

        # Normalizza lo spacing relativo tra le visite (invariante per paziente)
        t_self_max       = t_cumul[:, -1:].clamp(min=1e-8)
        t_norm_relative  = t_cumul / t_self_max        # ∈ [0,1], shape [B,T]

        if d3_fup_scale is not None:
            # Scala in modo che t_norm[-1] = d3_fup_scale del paziente
            # d3_fup_scale = D3_fup / D3_fup_max, già ∈ [0,1] dal preprocessor
            scale  = d3_fup_scale.unsqueeze(1).clamp(min=1e-3, max=1.0)  # [B,1]
            t_norm = t_norm_relative * scale                               # [B,T]
        else:
            t_norm = t_norm_relative

        # delta_t = incrementi consecutivi (primo sempre 0 per costruzione)
        delta_t          = torch.zeros_like(t_norm)
        delta_t[:, 1:]   = t_norm[:, 1:] - t_norm[:, :-1]

        t_feat = torch.stack([t_norm, delta_t], dim=-1)
        return t_norm, delta_t, t_feat


# ==================================================================
# TEMPORAL BACKBONES
# ==================================================================

class GRUBackbone(nn.Module):
    """
    GRU con h_0 inizializzato da static_h.
    Il conditioning statico è iniettato nell'hidden state iniziale:
    più efficiente del broadcast ad ogni step, stesso gradiente.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.static_to_h0 = nn.Linear(hidden_dim, n_layers * hidden_dim)

    def forward(self, x: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        x   = self.input_proj(x)
        h0  = self.static_to_h0(static_h)
        h0  = h0.view(-1, self.n_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        h, _ = self.gru(x, h0)
        return h


class LSTMBackbone(nn.Module):
    """
    LSTM con (h_0, c_0) inizializzati da static_h.
    Più parametri di GRU (1.33x); considera LSTM solo per T_medio > 15.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_layers   = n_layers
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
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
    """
    Transformer con PE continuo. Preferibile per T_medio > 15 e con GPU.
    Con T_medio=7: nessun vantaggio rispetto a GRU, costo 5-10x maggiore.
    """
    def __init__(
        self,
        input_dim: int, hidden_dim: int, n_layers: int,
        n_heads: int, dropout: float, pe_frequencies: int,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.temporal_pe = ContinuousTemporalPE(hidden_dim, pe_frequencies)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False,
        )

    def forward(self, x: torch.Tensor, static_h: torch.Tensor) -> torch.Tensor:
        # Le ultime 2 colonne di x sono (t_norm, delta_t) → usate per PE
        t_norm = x[:, :, -2]
        h      = self.input_proj(x)
        h      = h + self.temporal_pe(t_norm, static_h) + static_h.unsqueeze(1)
        return self.transformer(h)


# ==================================================================
# HIERARCHICAL GENERATOR
# ==================================================================

class HierarchicalGenerator(nn.Module):
    """
    Generatore con backbone selezionabile via config.

    PARAMETRI CHIAVE:
      backbone:    "gru" | "lstm" | "transformer"
      hidden_dim:  dimensione uniforme di tutte le proiezioni
      rnn_layers:  profondità GRU/LSTM (ignorato da Transformer)

    OUTPUT forward() include SEMPRE:
      n_visits_pred: output di n_visits_head (usato da supervision loss)
      n_visits:      n_visits effettivo per la mask (= fixed_visits in training)
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
        # retrocompatibilità
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

        if gru_layers is not None:
            rnn_layers = gru_layers

        # ── Time encoder ─────────────────────────────────────────────
        self.time_encoder = TimeEncoder(z_temporal_dim, hidden_dim)

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
        # [v6.2] Warm start followup_head: bias → sigmoid⁻¹(0.35) ≈ -0.62
        # Il follow-up mediano reale è ~35% del massimo (distribuzione skew-right).
        # Senza warm start, sigmoid(0)=0.5 → il generatore sovrastima il follow-up
        # nei primi epoch e il gradiente diverge prima di stabilizzarsi.
        nn.init.constant_(self.followup_head[-2].bias, -0.62)

        self.n_visits_head = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.constant_(self.n_visits_head[-1].bias, 6.0)  # warm start ≈7 visite

        # ── Temporal backbone ─────────────────────────────────────────
        input_dim = z_temporal_dim + 2   # z_t || (t_norm, delta_t)
        if backbone == "gru":
            self.backbone = GRUBackbone(input_dim, hidden_dim, rnn_layers, dropout)
        elif backbone == "lstm":
            self.backbone = LSTMBackbone(input_dim, hidden_dim, rnn_layers, dropout)
        elif backbone == "transformer":
            self.backbone = TransformerBackbone(
                input_dim, hidden_dim,
                n_transformer_layers, n_heads, dropout, pe_frequencies,
            )
        else:
            raise ValueError(
                f"backbone deve essere 'gru', 'lstm' o 'transformer'. Ricevuto: {backbone!r}"
            )

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
                    hidden_dim, preprocessor.embedding_configs[v.name]
                )
            else:
                self.static_cat_heads[v.name] = nn.Linear(hidden_dim, v.n_categories)

        # ── Output temporali ──────────────────────────────────────────
        # Output lineare (z-score), LayerNorm + zero-init per stabilità.
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

    @staticmethod
    def _cummax_irreversible(hazard: torch.Tensor) -> torch.Tensor:
        states, _ = torch.cummax(hazard, dim=1)
        return states

    def _soft_visit_mask(self, n_v: torch.Tensor) -> torch.Tensor:
        T      = self.max_len
        device = n_v.device
        t_pos  = torch.arange(T, dtype=torch.float32, device=device)
        logits = self.n_visits_sharpness * (n_v.unsqueeze(-1) - 0.5 - t_pos.unsqueeze(0))
        return torch.sigmoid(logits).unsqueeze(-1)  # [B,T,1]

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

        # 1. Follow-up prediction — followup_head è SEMPRE calcolato e usato
        #    [v6.2] In training: followup_head riceve gradiente sia da lambda_fup
        #    (supervisione verso il valore reale) sia dal TimeEncoder → WGAN.
        #    NON usiamo più il valore reale direttamente nel TimeEncoder perché
        #    questo interrompeva il gradiente su followup_head in training,
        #    rendendo followup_head inutile in inference.
        #    real_followup_norm viene usato solo come target della fup_loss.
        followup_scale = self.followup_head(z_static).squeeze(-1)   # [B] ∈ [0,1]

        # 2. n_visits predetto — MAI sovrascritto da fixed_visits
        #    È l'output grezzo di n_visits_head che la supervision loss deve allenare.
        n_v_raw  = F.softplus(self.n_visits_head(z_static).squeeze(-1)) + 1.0
        n_v_pred = n_v_raw.clamp(1.0, float(T))

        # 3. Visit mask
        if fixed_visits is not None:
            # Training: mask esatta basata sul n_visits reale campionato
            n_v_used   = fixed_visits.to(device).float().clamp(1.0, float(T))
            t_pos      = torch.arange(T, dtype=torch.float32, device=device)
            visit_mask = (n_v_used.unsqueeze(-1) > t_pos.unsqueeze(0)).float().unsqueeze(-1)
            n_v        = n_v_used
        elif hard_mask:
            # Inference con mask binaria
            n_v_hard   = n_v_pred.round()
            t_pos      = torch.arange(T, dtype=torch.float32, device=device)
            visit_mask = (n_v_hard.unsqueeze(-1) > t_pos.unsqueeze(0)).float().unsqueeze(-1)
            n_v        = n_v_hard
        else:
            # Training senza conditioning: soft mask differenziabile
            visit_mask = self._soft_visit_mask(n_v_pred)
            n_v        = n_v_pred

        # 4. Time encoding — [v6.1] ancoraggio a followup_scale (D3_fup normalizzato)
        # followup_scale è già calcolato sopra (reale in training, predetto in inference)
        t_norm, delta_t, t_feat = self.time_encoder(z_temporal, d3_fup_scale=followup_scale)

        # 5. Static encoding
        static_h = self.fc_static(z_static)   # [B, H]

        # 6. Backbone temporale
        backbone_in = torch.cat([z_temporal, t_feat], dim=-1)   # [B, T, Zt+2]
        h_seq       = self.backbone(backbone_in, static_h)       # [B, T, H]

        # 7. Output statici
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

        # 8. Output temporali (moltiplicati per visit_mask: padding → 0)
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
            "static_cont":      static_cont,
            "static_cat":       static_cat       if static_cat       else None,
            "static_cat_embed": static_cat_embed if static_cat_embed else None,
            "temporal_cont":    temporal_cont,
            "temporal_cat":     temporal_cat,
            "visit_mask":       visit_mask,
            "visit_times":      t_norm,
            "followup_norm":    followup_scale,
            "n_visits":         n_v,          # per la mask e il log Nv=
            "n_visits_pred":    n_v_pred,     # per n_visits_supervision_loss
        }