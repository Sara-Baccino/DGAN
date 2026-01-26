import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, z_patient=32, z_time=16, hidden=64, T=6, n_biomarkers=5, n_events=2):
        super().__init__()
        self.T = T
        self.n_biomarkers = n_biomarkers
        self.n_events = n_events

        # Embedding del baseline
        self.base_net = nn.Sequential(
            nn.Linear(z_patient, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Dinamica per ciascun timestep
        self.time_net = nn.Sequential(
            nn.Linear(z_time + hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

        # Output: biomarker + hazard per timestep
        self.biomarker_head = nn.Linear(hidden, n_biomarkers)
        self.hazard_head = nn.Linear(hidden, n_events)  # logit per evento binario

    def forward(self, z_patient, z_time):
        """
        z_patient: [B, z_patient]
        z_time: [B, T, z_time]
        """
        B = z_patient.size(0)
        baseline = self.base_net(z_patient)
        biomarker_seq = []
        hazard_seq = []

        for t in range(self.T):
            visit_idx = torch.full((B,1), t/self.T, device=z_patient.device)
            x_t = torch.cat([z_time[:, t], baseline, visit_idx], dim=1)
            h_t = self.time_net(x_t)

            biomarker_seq.append(self.biomarker_head(h_t))
            hazard_seq.append(torch.sigmoid(self.hazard_head(h_t)))  # valori in [0,1]

        # Output: [B, T, n_features]
        return torch.stack(biomarker_seq, dim=1), torch.stack(hazard_seq, dim=1)
