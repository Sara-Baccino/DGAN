import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float = 0.0):
        super().__init__()
        net = []
        d = input_dim
        for _ in range(layers):
            net.append(nn.Linear(d, hidden))
            net.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            d = hidden
        net.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalDiscriminator(nn.Module):
    def __init__(
        self,
        static_dim: int,
        temporal_dim: int,
        hidden_dim: int,
        gru_layers: int,
        mlp_layers: int,
        mlp_units: int,
        dropout: float = 0.0
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=temporal_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )

        mlp = []
        in_dim = hidden_dim + static_dim
        for _ in range(mlp_layers):
            mlp.append(nn.Linear(in_dim, mlp_units))
            mlp.append(nn.LeakyReLU(0.2))
            if dropout > 0:
                mlp.append(nn.Dropout(dropout))
            in_dim = mlp_units
        mlp.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def forward(
        self,
        static: torch.Tensor,        # [B, static_dim]
        temporal: torch.Tensor,      # [B, T, temporal_dim]
        mask: torch.Tensor           # [B, T]
    ) -> torch.Tensor:

        h_seq, _ = self.gru(temporal)   # [B, T, H]

        # Applica mask
        mask = mask.unsqueeze(-1)       # [B, T, 1]
        h_seq = h_seq * mask

        # Media pesata
        lengths = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        h = h_seq.sum(dim=1) / lengths            # [B, H]

        # Concat static
        x = torch.cat([static, h], dim=-1)

        return self.mlp(x)
