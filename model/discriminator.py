"""
model/discriminator.py  [v6 — discriminatore temporale configurabile]
================================================================================
Novità rispetto a v5:

  DISCRIMINATORE TEMPORALE — arch configurabile via JSON:
    "temporal_discriminator": {
      "arch": "cnn",   # "cnn" (default, raccomandato) | "gru" (retrocompat)
      ...
    }

  CNN (raccomandato per T_avg ≤ 10):
    3 blocchi Conv1D dilatati con dilation 1→2→4:
      Layer 1: receptive field = 3 step
      Layer 2: receptive field = 5 step
      Layer 3: receptive field = 7 step  → copre T_avg=7 esatto
    2-3x più veloce del GRU, completamente parallelizzabile.
    Cattura pattern locali (es. "ALP decresce in 2 step") meglio del GRU.

  GRU (retrocompatibilità v5):
    Identico alla versione precedente, attivato con arch="gru".

  TemporalDiscriminator è ora una factory function che istanzia
  CNNTemporalDiscriminator o GRUTemporalDiscriminator in base ad arch.
================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


# ==================================================================
# HELPER: prepara input concatenati per i discriminatori
# ==================================================================

def prepare_discriminator_inputs(batch: Dict, preprocessor) -> Dict:
    """
    Concatena tutte le feature in tensori flat per i discriminatori.

    [v4] followup_norm broadcastato su tutti gli step temporali.
    [v6] invariato — compatibile con generatori GRU e Transformer.
    """
    static_parts      = []
    static_mask_parts = []

    if "static_cont" in batch and batch["static_cont"] is not None:
        sc = batch["static_cont"]
        if "static_cont_mask" in batch and batch["static_cont_mask"] is not None:
            sc = sc * batch["static_cont_mask"]
            static_mask_parts.append(batch["static_cont_mask"])
        static_parts.append(sc)

    if "static_cat" in batch and batch["static_cat"] is not None:
        if torch.is_tensor(batch["static_cat"]):
            static_parts.append(batch["static_cat"])
            if "static_cat_mask" in batch and batch["static_cat_mask"] is not None:
                static_mask_parts.append(batch["static_cat_mask"])
        elif isinstance(batch["static_cat"], dict):
            for name, scat in batch["static_cat"].items():
                if ("static_cat_mask" in batch
                        and batch["static_cat_mask"] is not None
                        and name in batch["static_cat_mask"]):
                    mask = batch["static_cat_mask"][name]
                    scat = scat * mask
                    static_mask_parts.append(mask)
                static_parts.append(scat)
        else:
            raise TypeError(f"static_cat must be Tensor or Dict, got {type(batch['static_cat'])}")

    if "static_cat_embed" in batch and batch["static_cat_embed"]:
        for var_name, payload in batch["static_cat_embed"].items():
            if isinstance(payload, dict):
                raise TypeError(f"static_cat_embed['{var_name}'] is a dict, expected Tensor")
            vec = (
                preprocessor.embeddings[var_name](payload)
                if payload.dim() == 1 else payload
            )
            if var_name in (batch.get("static_cat_embed_mask") or {}):
                mask = batch["static_cat_embed_mask"][var_name]
                vec  = vec * mask.unsqueeze(-1)
                static_mask_parts.append(mask.unsqueeze(-1).expand_as(vec))
            else:
                ones_mask = torch.ones(vec.shape[0], device=vec.device)
                static_mask_parts.append(ones_mask.unsqueeze(-1).expand_as(vec))
            static_parts.append(vec)

    static      = torch.cat(static_parts, dim=-1)
    static_mask = torch.cat(static_mask_parts, dim=-1) if static_mask_parts else None

    # ── Temporal ─────────────────────────────────────────────────
    temporal_cont = batch["temporal_cont"]
    B, T, _       = temporal_cont.shape

    if "temporal_cont_mask" in batch and batch["temporal_cont_mask"] is not None:
        temporal_cont = temporal_cont * batch["temporal_cont_mask"]

    temporal_parts      = [temporal_cont]
    temporal_mask_parts = []

    if "temporal_cont_mask" in batch and batch["temporal_cont_mask"] is not None:
        temporal_mask_parts.append(batch["temporal_cont_mask"])
    else:
        temporal_mask_parts.append(torch.ones_like(temporal_cont))

    temp_cat_order = [
        v.name for v in preprocessor.vars
        if not v.static and v.kind == "categorical"
    ]
    for name in temp_cat_order:
        if "temporal_cat" not in batch or name not in batch["temporal_cat"]:
            raise KeyError(f"Missing temporal_cat '{name}' in batch")
        cat_ohe = batch["temporal_cat"][name]
        n_cat   = cat_ohe.shape[-1]
        temporal_parts.append(cat_ohe)
        if (
            "temporal_cat_mask" in batch
            and batch["temporal_cat_mask"] is not None
            and name in batch["temporal_cat_mask"]
        ):
            mask = batch["temporal_cat_mask"][name]
            temporal_mask_parts.append(mask.unsqueeze(-1).expand(-1, -1, n_cat))
        else:
            temporal_mask_parts.append(
                torch.ones(B, T, n_cat, device=cat_ohe.device, dtype=cat_ohe.dtype)
            )

    if "followup_norm" in batch and batch["followup_norm"] is not None:
        fn = batch["followup_norm"]
        fn_expanded = fn.unsqueeze(-1).unsqueeze(-1).expand(B, T, 1) if fn.dim() == 1 \
                      else fn.expand(B, T, 1)
        temporal_parts.append(fn_expanded)
        temporal_mask_parts.append(torch.ones(B, T, 1, device=fn.device))
    else:
        fn_expanded = torch.full((B, T, 1), 0.5, device=temporal_cont.device)
        temporal_parts.append(fn_expanded)
        temporal_mask_parts.append(torch.ones(B, T, 1, device=temporal_cont.device))

    temporal      = torch.cat(temporal_parts,      dim=-1)
    temporal_mask = torch.cat(temporal_mask_parts, dim=-1)

    vm = batch["visit_mask"]
    if vm.dim() == 3:
        vm = vm.squeeze(-1)

    return {
        "static":        static,
        "temporal":      temporal,
        "visit_mask":    vm,
        "temporal_mask": temporal_mask,
        "static_mask":   static_mask,
    }


# ==================================================================
# STATIC DISCRIMINATOR — Residual MLP con Feature Matching
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
    """
    Residual MLP con Spectral Norm e Feature Matching.

    get_features(x) → [B, hidden]: usato per feature matching loss.
    auxiliary_loss: cross-entropy per variabili con embedding (CENTRE).
    """

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
        self.res_blocks = nn.ModuleList([_ResBlock(hidden, dropout) for _ in range(static_layers)])
        self.head       = nn.utils.spectral_norm(nn.Linear(hidden, 1))

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

    def auxiliary_loss(self, x_real: torch.Tensor, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
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
# CNN TEMPORAL DISCRIMINATOR
# ==================================================================

class _TransposedConv(nn.Module):
    """Wrapper [B,T,C] → Conv1d([B,C,T]) → [B,T,C]"""
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _DilatedCNNBlock(nn.Module):
    """
    Residual block Conv1D dilatato con Spectral Norm.

    Con kernel=3 e dilation=2^i:
      i=0 → RF=3, i=1 → RF=5, i=2 → RF=7 (copre T_avg=7 con 3 layer)
    """
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.LayerNorm(channels),
            _TransposedConv(
                nn.utils.spectral_norm(
                    nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
                )
            ),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            _TransposedConv(
                nn.utils.spectral_norm(nn.Conv1d(channels, channels, 1))
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class CNNTemporalDiscriminator(nn.Module):
    """
    Discriminatore temporale CNN con dilatazioni esponenziali.

    Architettura:
      [B,T,D] → input_proj → [B,T,C]
      → DilatedBlock(d=1) → DilatedBlock(d=2) → DilatedBlock(d=4)
      → masked mean pooling → [B,C]
      → concat [static] → MLP → score [B,1]

    Vantaggi su T=7:
      - Completamente parallelizzabile: 2-3x più veloce del GRU
      - Nessuno stato nascosto: training più stabile in WGAN
      - Receptive field = T_avg = 7 con soli 3 layer
    """

    def __init__(
        self,
        static_dim:    int,
        temporal_dim:  int,
        hidden_dim:    int   = 64,
        kernel_size:   int   = 3,
        n_layers:      int   = 3,
        dilation_base: int   = 2,
        mlp_layers:    int   = 2,
        dropout:       float = 0.1,
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
        mlp  = []
        d_in = hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp.extend([
                nn.utils.spectral_norm(nn.Linear(d_in, hidden_dim)),
                nn.LeakyReLU(0.2),
            ])
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            d_in = hidden_dim
        mlp.append(nn.utils.spectral_norm(nn.Linear(d_in, 1)))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        static:        torch.Tensor,
        temporal:      torch.Tensor,
        visit_mask:    torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        vm  = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask
        x   = self.input_proj(temporal)
        for block in self.cnn_blocks:
            x = block(x)
        n_valid = vm.sum(dim=1, keepdim=True).clamp(min=1.0)
        h       = (x * vm.unsqueeze(-1)).sum(dim=1) / n_valid
        return self.mlp(torch.cat([static, h], dim=-1))


# ==================================================================
# GRU TEMPORAL DISCRIMINATOR  (retrocompatibilità v5)
# ==================================================================

class GRUTemporalDiscriminator(nn.Module):
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
        self.gru  = nn.GRU(
            input_size  = temporal_dim,
            hidden_size = gru_hidden_dim,
            num_layers  = gru_layers,
            batch_first = True,
            dropout     = dropout if gru_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(gru_hidden_dim, 1)
        mlp       = []
        d         = gru_hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp.extend([
                nn.utils.spectral_norm(nn.Linear(d, mlp_hidden_dim)),
                nn.LeakyReLU(0.2),
            ])
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            d = mlp_hidden_dim
        mlp.append(nn.utils.spectral_norm(nn.Linear(d, 1)))
        self.mlp = nn.Sequential(*mlp)

    def forward(self, static, temporal, visit_mask, temporal_mask=None):
        h_seq, _ = self.gru(temporal)
        vm       = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask
        logits   = self.attn(h_seq).squeeze(-1)
        masked   = logits.masked_fill(vm == 0, float("-inf"))
        all_pad  = (vm.sum(dim=1) == 0).unsqueeze(-1)
        safe     = torch.where(all_pad.expand_as(masked), torch.zeros_like(masked), masked)
        weights  = torch.softmax(safe, dim=1)
        h        = (weights.unsqueeze(-1) * h_seq).sum(dim=1)
        return self.mlp(torch.cat([static, h], dim=-1))


# ==================================================================
# FACTORY FUNCTION
# ==================================================================

def TemporalDiscriminator(
    static_dim:   int,
    temporal_dim: int,
    model_config=None,
    **kwargs,
):
    """
    Factory: restituisce CNNTemporalDiscriminator o GRUTemporalDiscriminator
    in base a model_config.temporal_discriminator.arch.

    Se model_config è None, usa kwargs (retrocompat diretta).
    """
    td = getattr(model_config, "temporal_discriminator", None) if model_config else None
    arch = getattr(td, "arch", kwargs.pop("arch", "cnn")).lower()

    def _get(key, default):
        return getattr(td, key, kwargs.pop(key, default)) if td else kwargs.pop(key, default)

    if arch == "cnn":
        return CNNTemporalDiscriminator(
            static_dim    = static_dim,
            temporal_dim  = temporal_dim,
            hidden_dim    = _get("hidden_dim",    64),
            kernel_size   = _get("kernel_size",    3),
            n_layers      = _get("n_layers",       3),
            dilation_base = _get("dilation_base",  2),
            mlp_layers    = _get("mlp_layers",     2),
            dropout       = _get("dropout",       0.1),
        )
    elif arch == "gru":
        return GRUTemporalDiscriminator(
            static_dim     = static_dim,
            temporal_dim   = temporal_dim,
            mlp_hidden_dim = _get("mlp_hidden_dim", 128),
            gru_hidden_dim = _get("gru_hidden_dim",  64),
            gru_layers     = _get("gru_layers",       2),
            mlp_layers     = _get("mlp_layers",       2),
            dropout        = _get("dropout",         0.1),
        )
    else:
        raise ValueError(
            f"temporal_discriminator.arch deve essere 'cnn' o 'gru'. Ricevuto: '{arch}'"
        )