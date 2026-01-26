import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, z_p, z_t, hidden, T, out_dim):
        super().__init__()
        self.T = T

        self.base = nn.Sequential(
            nn.Linear(z_p, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.time = nn.Sequential(
            nn.Linear(z_t + hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )

        self.out = nn.Linear(hidden, out_dim)

    def forward(self, z_p, z_t):
        B = z_p.size(0)
        base = self.base(z_p)
        seq = []

        for t in range(self.T):
            vt = torch.full((B,1), t/self.T, device=z_p.device)
            h = self.time(torch.cat([z_t[:,t], base, vt], 1))
            seq.append(self.out(h))

        return torch.stack(seq, 1)

class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

class DGAN:
    def __init__(self, cfg, data_dim):
        self.T = cfg["time"]["max_visits"]
        self.G = Generator(
            cfg["model"]["z_patient"],
            cfg["model"]["z_time"],
            cfg["model"]["hidden"],
            self.T,
            data_dim
        )
        self.D = Discriminator(self.T * data_dim)

        self.optG = optim.Adam(self.G.parameters(), lr=cfg["model"]["lr"])
        self.optD = optim.Adam(self.D.parameters(), lr=cfg["model"]["lr"])

    def train(self, real_data, epochs):
        for ep in range(epochs):
            z_p = torch.randn(real_data.size(0), 32)
            z_t = torch.randn(real_data.size(0), self.T, 16)

            fake = self.G(z_p, z_t).reshape(real_data.size(0), -1)
            real = real_data.reshape(real_data.size(0), -1)

            lossD = self.D(fake.detach()).mean() - self.D(real).mean()
            self.optD.zero_grad()
            lossD.backward()
            self.optD.step()

            lossG = -self.D(fake).mean()
            self.optG.zero_grad()
            lossG.backward()
            self.optG.step()

    def generate(self, n):
        z_p = torch.randn(n, 32)
        z_t = torch.randn(n, self.T, 16)
        return self.G(z_p, z_t)
