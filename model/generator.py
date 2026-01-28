"""
================================================================================
MODULO 2: GENERATOR.PY
Generatore gerarchico con GRU e hazard rate
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from config.config import DataConfig


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
        visit_times: Optional[torch.Tensor] = None  # [B, T] se forniti
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            z_static: [B, z_static_dim]
            z_temporal: [B, T, z_temporal_dim]
            temperature: temperatura Gumbel-Softmax
            visit_times: tempi visite normalizzati [0,1], opzionale
        """
        B = z_static.size(0)
        T = z_temporal.size(1)
        device = z_static.device
        
        outputs = {}
        
        # === STATIC ===
        baseline = self.static_net(z_static)
        
        # Continuous: tanh -> [0,1]
        if self.n_static_cont > 0:
            static_cont = self.static_continuous_head(baseline)
            static_cont = torch.tanh(static_cont) * 0.5 + 0.5
            outputs['static_continuous'] = static_cont
        
        # Categorical: Gumbel-Softmax
        if self.n_static_cat > 0:
            static_cat_list = []
            for head in self.static_categorical_heads:
                logits = head(baseline)
                probs = F.gumbel_softmax(logits, tau=temperature, hard=False)
                static_cat_list.append(probs)
            outputs['static_categorical'] = torch.cat(static_cat_list, dim=-1)
        
        # NUOVO: Stati iniziali per irreversibili
        initial_states_list = []
        if self.n_irreversible > 0:
            for head in self.initial_state_heads:
                logits = head(baseline)
                initial_state = F.gumbel_softmax(logits, tau=temperature, hard=False)
                initial_states_list.append(initial_state)
            initial_states = torch.stack(initial_states_list, dim=1)  # [B, n_irr, 2]
            outputs['initial_states'] = initial_states
        else:
            initial_states = None
        
        # NUOVO: Time-to-event
        if self.data_config.generate_time_to_event:
            tte = self.time_to_event_head(baseline)
            tte = torch.sigmoid(tte)  # [0,1]
            outputs['time_to_event'] = tte
        
        # === TEMPORAL ===
        baseline_expanded = baseline.unsqueeze(1).expand(-1, T, -1)
        
        # NUOVO: Visit times
        if visit_times is None:
            if self.data_config.visit_times_variable:
                # Genera tempi visite
                visit_times_logits = self.visit_times_head(baseline_expanded)
                visit_times = torch.sigmoid(visit_times_logits).squeeze(-1)  # [B, T]
                # Assicura monotonia
                visit_times = torch.cumsum(visit_times, dim=1)
                visit_times = visit_times / (visit_times[:, -1:] + 1e-8)  # Normalizza [0,1]
            else:
                # Tempi equidistanti
                visit_times = torch.linspace(0, 1, T, device=device)
                visit_times = visit_times.unsqueeze(0).expand(B, -1)
        
        outputs['visit_times'] = visit_times
        time_idx = visit_times.unsqueeze(-1)  # [B, T, 1]
        
        # Espandi initial states per GRU input
        if initial_states is not None:
            initial_states_expanded = initial_states.unsqueeze(1).expand(-1, T, -1, -1)
            initial_states_flat = initial_states_expanded.reshape(B, T, -1)
            gru_input = torch.cat([z_temporal, baseline_expanded, time_idx, initial_states_flat], dim=-1)
        else:
            gru_input = torch.cat([z_temporal, baseline_expanded, time_idx], dim=-1)
        
        gru_output, _ = self.gru(gru_input)  # [B, T, hidden]
        
        # Continuous: tanh -> [0,1]
        if self.n_temporal_cont > 0:
            temporal_cont = self.temporal_continuous_head(gru_output)
            temporal_cont = torch.tanh(temporal_cont) * 0.5 + 0.5
            outputs['temporal_continuous'] = temporal_cont
        
        # Categorical con hazard + stato iniziale
        if self.n_temporal_cat > 0:
            temporal_cat_list = []
            irreversible_idx = 0
            
            for head_idx, (head, is_irr) in enumerate(zip(self.temporal_categorical_heads, self.is_irreversible)):
                if is_irr:
                    # === HAZARD con STATO INIZIALE ===
                    hazard = torch.sigmoid(head(gru_output))  # [B, T, 1]
                    
                    # Stato iniziale da baseline
                    init_state = initial_states[:, irreversible_idx, :]  # [B, 2]
                    irreversible_idx += 1
                    
                    state = torch.zeros(B, T, 2, device=device)
                    state[:, 0, :] = init_state  # USA stato iniziale
                    
                    for t in range(1, T):
                        # Chi è già in stato 1 resta in 1
                        already_1 = state[:, t-1, 1] > 0.5
                        state[already_1, t, 1] = 1
                        state[already_1, t, 0] = 0
                        
                        # Chi è in stato 0 può transitare
                        still_0 = state[:, t-1, 0] > 0.5
                        if still_0.any():
                            h_t = hazard[still_0, t, 0]
                            logits = torch.stack([
                                torch.log(1 - h_t + 1e-8),
                                torch.log(h_t + 1e-8)
                            ], dim=-1)
                            transition = F.gumbel_softmax(logits, tau=temperature, hard=False)
                            state[still_0, t] = transition
                    
                    temporal_cat_list.append(state)
                    
                else:
                    # Standard categorical
                    logits = head(gru_output)
                    probs = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
                    temporal_cat_list.append(probs)
            
            outputs['temporal_categorical'] = torch.cat(temporal_cat_list, dim=-1)
        
        # Mask (sempre 1 in generazione)
        outputs['temporal_mask'] = torch.ones(B, T, 1, device=device)
        
        return outputs

