"""
model/discriminator.py  [gretel-style]
================================================================================
Discriminatori semplificati: input densi con valid_flag per il pooling.

Cambiamenti rispetto alla versione precedente:
  - prepare_discriminator_inputs: niente masking inline, usa valid_flag
  - StaticDiscriminator: invariato (MLP residuale + spectral norm)
  - TemporalDiscriminator: CNN dilatata con attention pooling via valid_flag
  - GRU opzionale per retrocompatibilità
================================================================================
"""

import warnings
import torch
import torch.nn as nn
from typing import Optional, Dict


# ==================================================================
# HELPER: prepara input concatenati
# ==================================================================

def prepare_discriminator_inputs(batch: Dict, preprocessor) -> Dict:
    """
    Concatena le feature in tensori flat [B,D] (static) e [B,T,D] (temporal).
    Non applica maschere inline: il valid_flag viene passato separatamente
    e usato solo nel pooling del discriminatore temporale.

    [gretel] followup_norm broadcastato su T come feature temporale globale.
    """
    static_parts = []

    if "static_cont" in batch and batch["static_cont"] is not None:
        static_parts.append(batch["static_cont"])

    if "static_cat" in batch and batch["static_cat"] is not None:
        sc = batch["static_cat"]
        if torch.is_tensor(sc):
            static_parts.append(sc)
        elif isinstance(sc, dict):
            for t in sc.values():
                static_parts.append(t)
        else:
            raise TypeError(f"static_cat must be Tensor or Dict, got {type(sc)}")

    if "static_cat_embed" in batch and batch["static_cat_embed"]:
        for var_name, payload in batch["static_cat_embed"].items():
            vec = (preprocessor.embeddings[var_name](payload)
                   if payload.dim() == 1 else payload)
            static_parts.append(vec)

    if not static_parts:
        raise ValueError(
            "Nessuna feature statica trovata nel batch. "
            "Il batch deve contenere almeno uno tra: static_cont, static_cat, static_cat_embed. "
            "Verifica che la configurazione abbia variabili statiche definite."
        )
    static = torch.cat(static_parts, dim=-1)

    # ── Temporal ─────────────────────────────────────────────────
    temporal_cont = batch["temporal_cont"]
    B, T, _       = temporal_cont.shape

    temporal_parts = [temporal_cont]

    temp_cat_order = [
        v.name for v in preprocessor.vars
        if not v.static and v.kind == "categorical"
    ]
    for name in temp_cat_order:
        if "temporal_cat" not in batch or name not in batch["temporal_cat"]:
            raise KeyError(f"Missing temporal_cat '{name}' in batch")
        temporal_parts.append(batch["temporal_cat"][name])

    # followup_norm broadcastato
    if "followup_norm" in batch and batch["followup_norm"] is not None:
        fn = batch["followup_norm"]
        fn_expanded = fn.view(B, 1, 1).expand(B, T, 1)
    else:
        fn_expanded = torch.full((B, T, 1), 0.5, device=temporal_cont.device)
    temporal_parts.append(fn_expanded)

    temporal = torch.cat(temporal_parts, dim=-1)

    # valid_flag: bool [B,T] — True = step reale, False = padding
    if "valid_flag" in batch:
        vf = batch["valid_flag"]
        if vf.dtype != torch.bool:
            vf = vf.bool()
        if vf.dim() == 3:
            vf = vf.squeeze(-1)
    else:
        # fallback: tutte valide (no padding)
        vf = torch.ones(B, T, dtype=torch.bool, device=temporal_cont.device)

    return {
        "static":     static,
        "temporal":   temporal,
        "valid_flag": vf,
    }


# ==================================================================
# STATIC DISCRIMINATOR
# ==================================================================

class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.utils.spectral_norm(nn.Linear(dim, dim)),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.utils.spectral_norm(nn.Linear(dim, dim)),
        )

    def forward(self, x):
        return x + self.block(x)


class StaticDiscriminator(nn.Module):
    """Residual MLP con Spectral Norm e Feature Matching."""

    def __init__(
        self,
        input_dim:            int,
        hidden:               int,
        static_layers:        int,
        dropout:              float = 0.05,
        embed_var_categories: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden)),
            nn.LeakyReLU(0.2),
        )
        self.res_blocks = nn.ModuleList(
            [_ResBlock(hidden, dropout) for _ in range(static_layers)])
        self.head = nn.utils.spectral_norm(nn.Linear(hidden, 1))

        self.aux_heads = nn.ModuleDict()
        if embed_var_categories:
            for var_name, n_cats in embed_var_categories.items():
                self.aux_heads[var_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, n_cats),
                )

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.get_features(x))

    def auxiliary_loss(
        self, x_real: torch.Tensor, targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if not self.aux_heads:
            return torch.tensor(0.0, device=x_real.device)
        losses = []
        for var_name, head in self.aux_heads.items():
            if var_name not in targets:
                continue
            logits = head(x_real)
            losses.append(nn.functional.cross_entropy(logits, targets[var_name].long()))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=x_real.device)


# ==================================================================
# CNN TEMPORAL DISCRIMINATOR  [gretel-style]
# ==================================================================

class _TransposedConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _DilatedCNNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.LayerNorm(channels),
            _TransposedConv(nn.utils.spectral_norm(
                nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad))),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            _TransposedConv(nn.utils.spectral_norm(nn.Conv1d(channels, channels, 1))),
        )

    def forward(self, x):
        return x + self.block(x)


class CNNTemporalDiscriminator(nn.Module):
    """
    CNN dilatata con attention pooling guidato da valid_flag.
    Input: [B,T,temporal_dim] + valid_flag [B,T] bool.
    """

    def __init__(
        self,
        static_dim:   int,
        temporal_dim: int,
        hidden_dim:   int   = 64,
        kernel_size:  int   = 3,
        n_layers:     int   = 3,
        dilation_base: int  = 2,
        mlp_layers:   int   = 2,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(temporal_dim, hidden_dim)),
            nn.LeakyReLU(0.2),
        )
        self.cnn_blocks = nn.ModuleList([
            _DilatedCNNBlock(hidden_dim, kernel_size, dilation_base ** i, dropout)
            for i in range(n_layers)
        ])
        self.attn = nn.Linear(hidden_dim, 1)

        mlp, d_in = [], hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp += [nn.utils.spectral_norm(nn.Linear(d_in, hidden_dim)), nn.LeakyReLU(0.2)]
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            d_in = hidden_dim
        mlp.append(nn.utils.spectral_norm(nn.Linear(d_in, 1)))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        static:     torch.Tensor,   # [B, static_dim]
        temporal:   torch.Tensor,   # [B, T, temporal_dim]
        valid_flag: torch.Tensor,   # [B, T] bool
    ) -> torch.Tensor:
        x = self.input_proj(temporal)
        for block in self.cnn_blocks:
            x = block(x)

        # Attention pooling: ignora step di padding via valid_flag
        scores = self.attn(x).squeeze(-1)                        # [B,T]
        scores = scores.masked_fill(~valid_flag, -1e9)
        weights = torch.softmax(scores, dim=1)                   # [B,T]
        h = (weights.unsqueeze(-1) * x).sum(dim=1)              # [B, H]

        return self.mlp(torch.cat([static, h], dim=-1))


class GRUTemporalDiscriminator(nn.Module):
    """GRU-based discriminatore (retrocompatibilità)."""

    def __init__(
        self,
        static_dim:     int,
        temporal_dim:   int,
        mlp_hidden_dim: int   = 128,
        gru_hidden_dim: int   = 64,
        gru_layers:     int   = 2,
        mlp_layers:     int   = 2,
        dropout:        float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = temporal_dim,
            hidden_size = gru_hidden_dim,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(gru_hidden_dim, 1)
        mlp, d = [], gru_hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp += [nn.utils.spectral_norm(nn.Linear(d, mlp_hidden_dim)), nn.LeakyReLU(0.2)]
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            d = mlp_hidden_dim
        mlp.append(nn.utils.spectral_norm(nn.Linear(d, 1)))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, static, temporal, valid_flag):
        h_seq, _ = self.gru(temporal)
        scores   = self.attn(h_seq).squeeze(-1)
        scores   = scores.masked_fill(~valid_flag, float("-inf"))
        # Gestione batch con tutti padding (non dovrebbe accadere post-imputation)
        all_pad  = (~valid_flag).all(dim=1, keepdim=True)
        scores   = torch.where(all_pad.expand_as(scores),
                               torch.zeros_like(scores), scores)
        weights  = torch.softmax(scores, dim=1)
        h        = (weights.unsqueeze(-1) * h_seq).sum(dim=1)
        return self.mlp(torch.cat([static, h], dim=-1))


# ==================================================================
# FACTORY FUNCTION
# ==================================================================

def TemporalDiscriminator(
    static_dim:   int,
    temporal_dim: int,
    model_config  = None,
    **kwargs,
):
    td   = getattr(model_config, "temporal_discriminator", None) if model_config else None
    arch = getattr(td, "arch", kwargs.pop("arch", "cnn")).lower()

    def _g(key, default):
        return getattr(td, key, kwargs.pop(key, default)) if td else kwargs.pop(key, default)

    if arch == "cnn":
        return CNNTemporalDiscriminator(
            static_dim    = static_dim,
            temporal_dim  = temporal_dim,
            hidden_dim    = _g("hidden_dim",    64),
            kernel_size   = _g("kernel_size",    3),
            n_layers      = _g("n_layers",       3),
            dilation_base = _g("dilation_base",  2),
            mlp_layers    = _g("mlp_layers",     2),
            dropout       = _g("dropout",       0.1),
        )
    elif arch == "gru":
        return GRUTemporalDiscriminator(
            static_dim     = static_dim,
            temporal_dim   = temporal_dim,
            mlp_hidden_dim = _g("mlp_hidden_dim", 128),
            gru_hidden_dim = _g("gru_hidden_dim",  64),
            gru_layers     = _g("gru_layers",       2),
            mlp_layers     = _g("mlp_layers",       2),
            dropout        = _g("dropout",         0.1),
        )
    else:
        raise ValueError(f"arch deve essere 'cnn' o 'gru'. Ricevuto: '{arch}'")