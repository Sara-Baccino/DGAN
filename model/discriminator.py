import torch
import torch.nn as nn
from typing import Optional, Dict, List


# ==================================================================
# HELPER: prepara input concatenati per i discriminatori
# ==================================================================
def prepare_discriminator_inputs(batch: Dict, preprocessor) -> Dict:
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
            vec = preprocessor.embeddings[var_name](payload) if payload.dim() == 1 else payload
            if ("static_cat_embed_mask" in batch
                    and batch["static_cat_embed_mask"] is not None
                    and var_name in batch["static_cat_embed_mask"]):
                mask = batch["static_cat_embed_mask"][var_name]
                vec  = vec * mask.unsqueeze(-1)
                static_mask_parts.append(mask.unsqueeze(-1).expand_as(vec))
            static_parts.append(vec)

    static      = torch.cat(static_parts, dim=-1)
    static_mask = torch.cat(static_mask_parts, dim=-1) if static_mask_parts else None

    temporal_cont = batch["temporal_cont"]
    B, T, _       = temporal_cont.shape
    temporal_parts      = [temporal_cont]
    temporal_mask_parts = []

    if "temporal_cont_mask" in batch and batch["temporal_cont_mask"] is not None:
        temporal_mask_parts.append(batch["temporal_cont_mask"])
    else:
        temporal_mask_parts.append(torch.ones_like(temporal_cont))

    temp_cat_order = [v.name for v in preprocessor.vars
                      if not v.static and v.kind == "categorical"]

    for name in temp_cat_order:
        if "temporal_cat" not in batch or name not in batch["temporal_cat"]:
            raise KeyError(f"Missing temporal_cat '{name}' in batch")
        cat_ohe = batch["temporal_cat"][name]
        if cat_ohe.shape[-1] == 3:
            cat_ohe = cat_ohe[..., 1:]
        n_cat = cat_ohe.shape[-1]
        temporal_parts.append(cat_ohe)
        if ("temporal_cat_mask" in batch
                and batch["temporal_cat_mask"] is not None
                and name in batch["temporal_cat_mask"]):
            mask = batch["temporal_cat_mask"][name]
            temporal_mask_parts.append(mask.unsqueeze(-1).expand(-1, -1, n_cat))
        else:
            temporal_mask_parts.append(
                torch.ones(B, T, n_cat, device=cat_ohe.device, dtype=cat_ohe.dtype)
            )

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
# STATIC DISCRIMINATOR con Spectral Norm + Auxiliary Embedding Loss
# ==================================================================
class StaticDiscriminator(nn.Module):
    """
    MLP con Spectral Normalization.

    Aggiunta: auxiliary_heads per le variabili embedded (es. CENTRE).
    Ogni head ausiliaria predice la categoria originale dato il vettore
    static concatenato. Questo forza gli embedding del generatore a
    rimanere interpretabili: se il generatore produce un embedding
    che il discriminatore non riesce a classificare come categoria reale,
    viene penalizzato anche tramite questa loss aggiuntiva.

    La auxiliary loss e' calcolata separatamente in DGAN._train_discriminators
    e sommata alla loss WGAN con peso lambda_aux.
    """

    def __init__(
        self,
        input_dim:    int,
        hidden:       int,
        static_layers: int,
        dropout:      float = 0.1,
        # Dizionario {nome_variabile: n_categorie} per le auxiliary heads
        embed_var_categories: Optional[Dict[str, int]] = None,
    ):
        super().__init__()

        # MLP principale con Spectral Norm
        layers = []
        d = input_dim
        for _ in range(static_layers):
            layers.append(nn.utils.spectral_norm(nn.Linear(d, hidden)))
            layers.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.utils.spectral_norm(nn.Linear(d, 1)))
        self.net = nn.Sequential(*layers)

        # Auxiliary heads per variabili embedded:
        # dato il vettore static completo, predice la categoria originale.
        # Permette al discriminatore di "vedere" se gli embedding generati
        # corrispondono a categorie reali, ottimizzando il mapping insieme.
        self.aux_heads = nn.ModuleDict()
        if embed_var_categories:
            for var_name, n_cats in embed_var_categories.items():
                self.aux_heads[var_name] = nn.Sequential(
                    nn.Linear(input_dim, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, n_cats),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, static_dim] -> [B, 1]"""
        return self.net(x)

    def auxiliary_loss(
        self,
        x_real:  torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Cross-entropy tra la predizione ausiliaria e le categorie reali.

        Args:
            x_real:  [B, static_dim] input reale concatenato
            targets: {var_name: [B] indici categoria interi}
        Returns:
            loss scalare (media su tutte le variabili embedded)
        """
        if not self.aux_heads:
            return torch.tensor(0.0, device=x_real.device)

        losses = []
        for var_name, head in self.aux_heads.items():
            if var_name not in targets:
                continue
            logits = head(x_real)                  # [B, n_cats]
            target = targets[var_name].long()       # [B]
            losses.append(
                nn.functional.cross_entropy(logits, target)
            )
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=x_real.device)


# ==================================================================
# TEMPORAL DISCRIMINATOR con Attention Pooling
# ==================================================================
class TemporalDiscriminator(nn.Module):
    """
    GRU + attention pooling mascherato + MLP.

    L'attention pooling sostituisce la media semplice:
    impara a pesare step temporali diversamente (es. pesa di piu
    il momento di una transizione irreversibile rispetto agli step
    dove tutto rimane stabile). La media semplice diluisce il
    segnale degli eventi rari su sequenze lunghe (~350 mesi).
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

        # Attention: scalare per ogni step
        self.attn = nn.Linear(gru_hidden_dim, 1)

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
        static:        torch.Tensor,
        temporal:      torch.Tensor,
        visit_mask:    torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if temporal_mask is not None:
            temporal = temporal * temporal_mask

        h_seq, _ = self.gru(temporal)               # [B, T, H]

        # Masked softmax attention
        attn_logits = self.attn(h_seq).squeeze(-1)  # [B, T]
        vm = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask

        masked_logits = attn_logits.masked_fill(vm == 0, float("-inf"))
        # Edge case: tutti gli step mascherati -> pesi uniformi
        all_masked    = (vm.sum(dim=1) == 0).unsqueeze(-1)
        safe_logits   = torch.where(
            all_masked.expand_as(masked_logits),
            torch.zeros_like(masked_logits),
            masked_logits,
        )
        attn_weights  = torch.softmax(safe_logits, dim=1)    # [B, T]

        h = (attn_weights.unsqueeze(-1) * h_seq).sum(dim=1)  # [B, H]
        x = torch.cat([static, h], dim=-1)
        return self.mlp(x)

