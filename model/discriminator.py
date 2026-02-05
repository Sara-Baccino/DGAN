import torch
import torch.nn as nn
from typing import Optional, Dict


# ==================================================================
# HELPER: prepara input concatenati per i discriminatori
# ==================================================================
def prepare_discriminator_inputs(batch: Dict, preprocessor) -> Dict:
    """
    Costruisce tensori concatenati dalle componenti del batch.

    Funziona sia sul batch REALE (ha maschere dal preprocessor)
    sia sull'output del GENERATORE (non ha maschere → le crea da 1).

    Returns:
        static          : [B, static_dim_total]
        temporal        : [B, T, temporal_dim_total]
        visit_mask      : [B, T]
        temporal_mask   : [B, T, temporal_dim_total]
        static_mask     : [B, static_dim_total]  o None
    """
    # ---- STATIC ----
    static_parts      = []
    static_mask_parts = []

    # continuous
    if "static_cont" in batch and batch["static_cont"] is not None:
        sc = batch["static_cont"]                                          # [B, n_cont]
        if "static_cont_mask" in batch and batch["static_cont_mask"] is not None:
            sc = sc * batch["static_cont_mask"]
            static_mask_parts.append(batch["static_cont_mask"])
        static_parts.append(sc)

    # categoriche statiche OHE (dict[name → Tensor[B,K]])
    # categoriche statiche
    if "static_cat" in batch and batch["static_cat"] is not None:

        # ===== CASO 1: batch reale → Tensor [B, sum_ohe]
        if torch.is_tensor(batch["static_cat"]):
            static_parts.append(batch["static_cat"])

            if "static_cat_mask" in batch and batch["static_cat_mask"] is not None:
                static_mask_parts.append(batch["static_cat_mask"])

        # ===== CASO 2: batch fake → dict[name → Tensor[B,K]]
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
            raise TypeError(
                f"static_cat must be Tensor or Dict, got {type(batch['static_cat'])}"
            )


    # embedding statici  →  due casi:
    #   batch reale:  "static_cat_embed" = {name: [B]}    indici   → applica layer
    #   batch fake:   "static_cat_embed" = {name: [B, D]} vettori  → usa direttamente
    
    if "static_cat_embed" in batch and batch["static_cat_embed"]:
        for var_name, payload in batch["static_cat_embed"].items():
            if isinstance(payload, dict):
                raise TypeError(f"static_cat_embed['{var_name}'] is a dict, expected Tensor")
            if payload.dim() == 1:
                vec = preprocessor.embeddings[var_name](payload)           # [B, D]
            else:
                vec = payload                                              # [B, D]

            # maschera embedding (presente solo nel batch reale)
            if ("static_cat_embed_mask" in batch
                    and batch["static_cat_embed_mask"] is not None
                    and var_name in batch["static_cat_embed_mask"]):
                mask = batch["static_cat_embed_mask"][var_name]            # [B]
                vec  = vec * mask.unsqueeze(-1)
                static_mask_parts.append(mask.unsqueeze(-1).expand_as(vec))

            static_parts.append(vec)

    for i, p in enumerate(static_parts):
        assert torch.is_tensor(p), f"static_parts[{i}] is {type(p)}"

    static      = torch.cat(static_parts, dim=-1)                         # [B, S]
    static_mask = torch.cat(static_mask_parts, dim=-1) if static_mask_parts else None

    # ---- TEMPORAL ----
    temporal_cont = batch["temporal_cont"]                                 # [B, T, n_cont]
    B, T, _       = temporal_cont.shape

    temporal_parts      = [temporal_cont]
    temporal_mask_parts = []

    # maschera continuous temporale: usa se presente, altrimenti tutto 1
    if "temporal_cont_mask" in batch and batch["temporal_cont_mask"] is not None:
        temporal_mask_parts.append(batch["temporal_cont_mask"])            # [B, T, n_cont]
    else:
        temporal_mask_parts.append(torch.ones_like(temporal_cont))

    # categoriche temporali nell'ordine dei preprocessor.vars
    temp_cat_order = [v.name for v in preprocessor.vars
                      if not v.static and v.kind == "categorical"]

    for name in temp_cat_order:
        if "temporal_cat" not in batch or name not in batch["temporal_cat"]:
            raise KeyError(f"Missing temporal_cat '{name}' in batch")
        cat_ohe = batch["temporal_cat"][name]   # [B, T, K]
        if cat_ohe.shape[-1] == 3:
            cat_ohe = cat_ohe[..., 1:]  # droppa "missing"

        n_cat   = cat_ohe.shape[-1]
        temporal_parts.append(cat_ohe)

        # maschera categorica temporale: usa se presente, altrimenti tutto 1
        if ("temporal_cat_mask" in batch
                and batch["temporal_cat_mask"] is not None
                and name in batch["temporal_cat_mask"]):
            mask = batch["temporal_cat_mask"][name]                        # [B, T]
            temporal_mask_parts.append(mask.unsqueeze(-1).expand(-1, -1, n_cat))
        else:
            temporal_mask_parts.append(torch.ones(B, T, n_cat,
                                                  device=cat_ohe.device,
                                                  dtype=cat_ohe.dtype))

    # sanity check REAL vs FAKE temporal dims
    expected_temp_dim = sum(p.shape[-1] for p in temporal_parts)

    temporal      = torch.cat(temporal_parts,      dim=-1)                # [B, T, D_temp]
    temporal_mask = torch.cat(temporal_mask_parts, dim=-1)                # [B, T, D_temp]
    #print("TEMP PARTS:", [p.shape for p in temporal_parts])

    # visit_mask: garantisci [B, T]
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
# STATIC DISCRIMINATOR
# ==================================================================
class StaticDiscriminator(nn.Module):
    """
    MLP su un singolo tensore concatenato [B, static_dim].
    Le maschere vengono applicate PRIMA della chiamata,
    dentro prepare_discriminator_inputs.
    """
    def __init__(self, input_dim: int, hidden: int, static_layers: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(static_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, static_dim] → [B, 1]"""
        return self.net(x)


# ==================================================================
# TEMPORAL DISCRIMINATOR
# ==================================================================
class TemporalDiscriminator(nn.Module):
    """
    GRU sulla sequenza temporale → summary pesata da visit_mask
    → concat con static → MLP → score.
    """
    def __init__(
        self,
        static_dim:     int,
        temporal_dim:   int,
        mlp_hidden_dim: int,
        gru_hidden_dim: int,
        gru_layers:     int,
        mlp_layers:     int,
        dropout:        float = 0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=temporal_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        mlp = []
        d   = gru_hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp.append(nn.Linear(d, mlp_hidden_dim))
            mlp.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            d = mlp_hidden_dim
        mlp.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        static:        torch.Tensor,                     # [B, S]
        temporal:      torch.Tensor,                     # [B, T, D]
        visit_mask:    torch.Tensor,                     # [B, T]
        temporal_mask: Optional[torch.Tensor] = None,    # [B, T, D]
    ) -> torch.Tensor:

        # applica maschera valori (zero-out dei missing)
        if temporal_mask is not None:
            temporal = temporal * temporal_mask

        h_seq, _ = self.gru(temporal)                    # [B, T, H]

        # applica visit_mask → solo le visite presenti contribuiscono
        if visit_mask.dim() == 2:
            visit_mask = visit_mask.unsqueeze(-1)         # [B, T, 1]
        h_seq = h_seq * visit_mask

        # media pesata sulle visite presenti
        lengths = visit_mask.sum(dim=1).clamp(min=1.0)   # [B, 1]
        h       = h_seq.sum(dim=1) / lengths             # [B, H]

        x = torch.cat([static, h], dim=-1)               # [B, S+H]
        return self.mlp(x)                               # [B, 1]