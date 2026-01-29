import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class HierarchicalGenerator(nn.Module):
    """
    Generatore gerarchico parametrizzato con GRU, per dati statici e temporali,
    con supporto per variabili irreversibili, teacher forcing e conditioning opzionale.
    """

    def __init__(
        self,
        data_cfg,
        *,
        z_static_dim: int,
        z_temporal_dim: int,
        hidden_dim: int,
        gru_layers: int,
        dropout: float,
        cond_dim: int = 0
    ):
        super().__init__()

        self.data_cfg = data_cfg
        self.max_len = data_cfg.max_len
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        # -------------------------
        # STATIC BASELINE
        # -------------------------
        self.static_net = nn.Sequential(
            nn.Linear(z_static_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.static_cont_head = nn.Linear(hidden_dim, data_cfg.n_static_cont)
        self.static_cat_heads = nn.ModuleList([
            nn.Linear(hidden_dim, k) for k in data_cfg.n_static_cat
        ])

        # -------------------------
        # INITIAL IRREVERSIBLE STATE
        # -------------------------
        self.init_irr_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in data_cfg.irreversible_idx
        ])

        # -------------------------
        # GRU
        # -------------------------
        irr_state_dim = len(data_cfg.irreversible_idx)
        gru_input_dim = z_temporal_dim + hidden_dim + 1 + irr_state_dim
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        # -------------------------
        # TEMPORAL HEADS
        # -------------------------
        self.temp_cont_head = nn.Linear(hidden_dim, data_cfg.n_temp_cont)
        self.temp_cat_heads = nn.ModuleList()
        self.hazard_heads = nn.ModuleList()

        for i, k in enumerate(data_cfg.n_temp_cat):
            if i in data_cfg.irreversible_idx:
                # irreversibile → hazard head
                self.hazard_heads.append(nn.Linear(hidden_dim + 1, 1))
                self.temp_cat_heads.append(None)
            else:
                self.temp_cat_heads.append(nn.Linear(hidden_dim, k))
                self.hazard_heads.append(None)

    # =====================================================
    # FORWARD
    # =====================================================
    def forward(
        self,
        z_static: torch.Tensor,
        z_temporal: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        teacher_forcing: bool = False,
        real_irr: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ):
        B, T, _ = z_temporal.shape
        device = z_static.device

        # -------------------------
        # STATIC BASELINE
        # -------------------------
        if condition is not None:
            z_static = torch.cat([z_static, condition], dim=-1)

        baseline = self.static_net(z_static)
        static_cont = torch.sigmoid(self.static_cont_head(baseline))

        static_cat = [
            F.gumbel_softmax(head(baseline), tau=temperature, hard=False)
            for head in self.static_cat_heads
        ]
        static_cat = torch.cat(static_cat, dim=-1)

        # -------------------------
        # INITIAL IRREVERSIBLE STATE
        # -------------------------
        irr_states = [
            F.gumbel_softmax(head(baseline), tau=temperature, hard=False)[:, 1]
            for head in self.init_irr_heads
        ]
        irr_states = torch.stack(irr_states, dim=-1)  # [B, n_irr]

        # -------------------------
        # VISIT TIMES
        # -------------------------
        t = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, -1)

        if lengths is None:
            lengths = torch.randint(1, T + 1, (B,), device=device)

        visit_mask = torch.zeros(B, T, 1, device=device)
        for i, L in enumerate(lengths):
            visit_mask[i, :L, 0] = 1

        # -------------------------
        # GRU LOOP
        # -------------------------
        h = torch.zeros(self.gru.num_layers, B, self.hidden_dim, device=device)
        temp_cont_list = []
        temp_cat_list = []

        irr = irr_states.clone()

        for step in range(T):
            # GRU input: z_temporal + baseline + time + irreversible
            gru_in = torch.cat([
                z_temporal[:, step],
                baseline,
                t[:, step:step+1],
                irr
            ], dim=-1).unsqueeze(1)

            out, h = self.gru(gru_in, h)
            out = out.squeeze(1)

            # --- CONTINUOUS ---
            temp_cont_list.append(torch.sigmoid(self.temp_cont_head(out)))

            # --- CATEGORICAL ---
            cat_step = []
            irr_idx = 0

            for i, k in enumerate(self.data_cfg.n_temp_cat):
                if i in self.data_cfg.irreversible_idx:
                    hazard = torch.sigmoid(
                        self.hazard_heads[i](torch.cat([out, t[:, step:step+1]], dim=-1))
                    ).squeeze(-1)

                    if teacher_forcing and real_irr is not None:
                        irr[:, irr_idx] = real_irr[:, step, irr_idx]
                    else:
                        flip = torch.bernoulli(hazard * (1 - irr[:, irr_idx]))
                        irr[:, irr_idx] = torch.clamp(irr[:, irr_idx] + flip, max=1.0)

                    cat_step.append(irr[:, irr_idx:irr_idx+1])
                    irr_idx += 1
                else:
                    logits = self.temp_cat_heads[i](out)
                    cat_step.append(F.gumbel_softmax(logits, tau=temperature, hard=False))

            temp_cat_list.append(torch.cat(cat_step, dim=-1))

        # Stack time dimension
        temp_cont = torch.stack(temp_cont_list, dim=1) * visit_mask
        temp_cat = torch.stack(temp_cat_list, dim=1) * visit_mask

        return {
            "static_cont": static_cont,
            "static_cat": static_cat,
            "temporal_cont": temp_cont,
            "temporal_cat": temp_cat,
            "visit_mask": visit_mask,
            "visit_times": t
        }
