"""
================================================================================
MODULO 3: DISCRIMINATOR.PY
Discriminatori per static e temporal
================================================================================
"""

import torch
import torch.nn as nn


class StaticDiscriminator(nn.Module):
    """Discriminatore per features statiche."""
    
    def __init__(self, input_dim: int, num_layers: int = 5, num_units: int = 200):
        super().__init__()
        
        layers = []
        last_dim = input_dim
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(last_dim, num_units),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            last_dim = num_units
        
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, static_features: torch.Tensor) -> torch.Tensor:
        return self.net(static_features)


class TemporalDiscriminator(nn.Module):
    """
    Discriminatore temporale con GRU per preservare CAUSALITÀ.
    Condizionato su static features.
    """
    
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        num_layers: int = 3,
        num_units: int = 200
    ):
        super().__init__()
        
        # Proiezione static
        self.static_projection = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GRU per processare sequenza temporale (preserva causalità)
        # Input: temporal_features + static_projection + mask
        self.gru = nn.GRU(
            input_size=temporal_dim + hidden_dim + 1,  # +1 per mask
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.1 if gru_layers > 1 else 0
        )
        
        # Rete finale: prende ultimo hidden state GRU + static_projection
        layers = []
        last_dim = hidden_dim * 2  # GRU output + static
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(last_dim, num_units),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            last_dim = num_units
        
        layers.append(nn.Linear(last_dim, 1))
        self.final_net = nn.Sequential(*layers)
    
    def forward(
        self,
        static_features: torch.Tensor,
        temporal_sequence: torch.Tensor,
        temporal_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            static_features: [B, static_dim]
            temporal_sequence: [B, T, temporal_dim]
            temporal_mask: [B, T, 1]
        """
        B = temporal_sequence.size(0)
        T = temporal_sequence.size(1)
        
        # Proietta static
        static_proj = self.static_projection(static_features)  # [B, hidden]
        static_expanded = static_proj.unsqueeze(1).expand(-1, T, -1)
        
        # Combina temporal + static + mask
        gru_input = torch.cat([temporal_sequence, static_expanded, temporal_mask], dim=-1)
        
        # GRU (preserva ordine temporale)
        gru_output, gru_hidden = self.gru(gru_input)
        
        # Usa ultimo hidden state (summary dell'intera sequenza)
        last_hidden = gru_hidden[-1]  # [B, hidden]
        
        # Combina con static projection
        combined = torch.cat([last_hidden, static_proj], dim=-1)
        
        return self.final_net(combined)
