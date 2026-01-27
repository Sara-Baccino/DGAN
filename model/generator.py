"""
================================================================================
MODULO 2: GENERATOR.PY
Generatore gerarchico con GRU e hazard rate
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import DataConfig
from typing import Dict, Optional 


class HierarchicalGenerator(nn.Module):
    """
    Generator gerarchico con:
    - Hazard rate CON stato iniziale
    - Time-to-event opzionale
    - Output continuous in [0,1] (compatibile con Wasserstein)
    """
    
    def __init__(
        self,
        data_config: DataConfig,
        z_static_dim: int,
        z_temporal_dim: int,
        hidden_dim: int,
        gru_layers: int
    ):
        super().__init__()
        
        self.data_config = data_config
        self.z_static_dim = z_static_dim
        self.z_temporal_dim = z_temporal_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_len = data_config.max_sequence_len
        
        # Dimensioni
        self.n_static_cont = len(data_config.static_continuous)
        self.n_static_cat = len(data_config.static_categorical)
        self.n_temporal_cont = len(data_config.temporal_continuous)
        self.n_temporal_cat = len(data_config.temporal_categorical)
        
        # === Rete Baseline (Static) ===
        self.static_net = nn.Sequential(
            nn.Linear(z_static_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Output statici
        if self.n_static_cont > 0:
            self.static_continuous_head = nn.Linear(hidden_dim, self.n_static_cont)
        
        if self.n_static_cat > 0:
            self.static_categorical_heads = nn.ModuleList([
                nn.Linear(hidden_dim, len(var.categories))
                for var in data_config.static_categorical
            ])
        
        # Output stato iniziale per variabili irreversibili
        self.n_irreversible = len(data_config.irreversible_vars)
        if self.n_irreversible > 0:
            self.initial_state_heads = nn.ModuleList([
                nn.Linear(hidden_dim, 2)  # Logits per (0, 1)
                for _ in data_config.irreversible_vars
            ])
        
        # === GRU per Dinamica Temporale ===
        # Input: z_temporal + baseline + time_normalized + initial_states
        gru_input_dim = z_temporal_dim + hidden_dim + 1 + self.n_irreversible * 2
        
        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0
        )
        
        # Output temporali
        if self.n_temporal_cont > 0:
            self.temporal_continuous_head = nn.Linear(hidden_dim, self.n_temporal_cont)
        
        if self.n_temporal_cat > 0:
            self.temporal_categorical_heads = nn.ModuleList()
            self.is_irreversible = []
            
            for var in data_config.temporal_categorical:
                n_cats = len(var.categories)
                if var.is_irreversible:
                    # Hazard: probabilità transizione 0->1
                    self.temporal_categorical_heads.append(nn.Linear(hidden_dim, 1))
                    self.is_irreversible.append(True)
                else:
                    # Logits standard
                    self.temporal_categorical_heads.append(nn.Linear(hidden_dim, n_cats))
                    self.is_irreversible.append(False)
        
        # NUOVO: Time-to-event
        if data_config.generate_time_to_event:
            self.time_to_event_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        
        # NUOVO: Visit times generator
        if data_config.visit_times_variable:
            self.visit_times_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        z_static: torch.Tensor,
        z_temporal: torch.Tensor,
        temperature: float = 1.0,
        visit_times: Optional[torch.Tensor] = None,  # [B, T] se forniti
        initial_states: Optional[torch.Tensor] = None 
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_static: [B, z_static_dim]
            z_temporal: [B, T, z_temporal_dim]
            temperature: temperatura Gumbel-Softmax
            visit_times: tempi visite normalizzati [0,1], opzionale
        """
        B, T, _ = z_temporal.shape
        outputs = {}

        baseline = self.static_net(z_static)

        # Static continuous
        if self.n_static_cont > 0:
            x = torch.tanh(self.static_continuous_head(baseline)) * 0.5 + 0.5
            outputs['static_continuous'] = x

        # Static categorical
        if self.n_static_cat > 0:
            cats = []
            for head in self.static_categorical_heads:
                logits = head(baseline)
                cats.append(F.gumbel_softmax(logits, temperature))
            outputs['static_categorical'] = torch.cat(cats, dim=-1)

        baseline_exp = baseline[:, None].expand(-1, T, -1)

        # Time index
        if visit_times is None:
            visit_times = torch.linspace(0, 1, T, device=z_static.device)[None].expand(B, -1)
        t_idx = visit_times[:, :, None]

        if initial_states is not None:
            init = initial_states[:, None].expand(-1, T, -1, -1).reshape(B, T, -1)
            gru_input = torch.cat([z_temporal, baseline_exp, t_idx, init], dim=-1)
        else:
            gru_input = torch.cat([z_temporal, baseline_exp, t_idx], dim=-1)

        h, _ = self.gru(gru_input)

        # Temporal continuous
        if self.n_temporal_cont > 0:
            x = torch.tanh(self.temporal_continuous_head(h)) * 0.5 + 0.5
            outputs['temporal_continuous'] = x

        # Temporal categorical (hazard)
        cats = []
        irr_idx = 0
        for head, is_irr in zip(self.temporal_categorical_heads, self.is_irreversible):
            if is_irr:
                hazard = torch.sigmoid(head(h))
                state = torch.zeros(B, T, 2, device=h.device)
                state[:, 0] = initial_states[:, irr_idx]
                for t in range(1, T):
                    stay = state[:, t-1, 1] > 0.5
                    state[stay, t, 1] = 1
                    state[~stay, t, 1] = hazard[~stay, t, 0]
                    state[~stay, t, 0] = 1 - hazard[~stay, t, 0]
                cats.append(state)
                irr_idx += 1
            else:
                cats.append(F.gumbel_softmax(head(h), temperature))
        outputs['temporal_categorical'] = torch.cat(cats, dim=-1)

        outputs['temporal_mask'] = torch.ones(B, T, 1, device=h.device)
        return outputs