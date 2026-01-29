"""
================================================================================
MODULO 5: DGAN.PY
Classe principale con training e generation
================================================================================
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
import numpy as np
from opacus import PrivacyEngine
from utils.losses import gradient_penalty
from model.generator import HierarchicalGenerator
from model.discriminator import StaticDiscriminator, TemporalDiscriminator

class LongitudinalSequenceDataset(Dataset):
    """Dataset per sequenze longitudinali."""
    def __init__(self, data_tuple):
        self.data = data_tuple
        self.N = next(x.shape[0] for x in data_tuple if x is not None)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return tuple(x[idx] if x is not None else None for x in self.data)


class DGAN:
    """DoppelGANger parametrizzato con DataConfig e ModelConfig."""
    def __init__(self, data_config, model_config, device=None):
        self.data_config = data_config
        self.model_config = model_config
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # --- dimensioni statiche e temporali ---
        self.static_dim = data_config.n_static_cont + sum(data_config.n_static_cat)
        self.temporal_dim = data_config.n_temp_cont + sum(data_config.n_temp_cat)
        self.max_visits = data_config.max_len

        self._build_model()

        self.current_temperature = model_config.gumbel_temperature_start
        self.loss_history = {"generator": [], "disc_static": [], "disc_temporal": [], "epsilon": []}

        self.privacy_engine = None

    def _build_model(self):
        """Costruisce generator e discriminatori parametrizzati dai config."""

        gen_cfg = self.model_config.generator
        disc_cfg = self.model_config.discriminator

        self.generator = HierarchicalGenerator(
            cfg=self.data_config,
            z_static_dim=gen_cfg.z_static_dim,
            z_temporal_dim=gen_cfg.z_temporal_dim,
            hidden_dim=gen_cfg.hidden_dim,
            gru_layers=gen_cfg.gru_layers,
            cond_dim=0
        ).to(self.device)

        self.disc_static = StaticDiscriminator(
            input_dim=self.static_dim,
            hidden=disc_cfg.hidden_dim,
            layers=disc_cfg.static_layers,
            dropout=disc_cfg.dropout
        ).to(self.device)

        self.disc_temporal = TemporalDiscriminator(
            static_dim=self.static_dim,
            temporal_dim=self.temporal_dim,
            hidden_dim=disc_cfg.hidden_dim,
            gru_layers=disc_cfg.temporal_layers,  # puoi separare gru_layers e mlp_layers se vuoi
            mlp_layers=disc_cfg.temporal_layers,
            mlp_units=disc_cfg.hidden_dim,
            dropout=disc_cfg.dropout
        ).to(self.device)

        # --- ottimizzatori ---
        self.opt_gen = torch.optim.Adam(self.generator.parameters(), lr=self.model_config.lr)
        self.opt_disc_static = torch.optim.Adam(self.disc_static.parameters(), lr=self.model_config.lr)
        self.opt_disc_temporal = torch.optim.Adam(self.disc_temporal.parameters(), lr=self.model_config.lr)

    def set_train(self):
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()

    def _attach_privacy_engine(self, dataloader):
        """Abilita DP su discriminatori."""
        self.privacy_engine = PrivacyEngine()

        self.disc_static, self.opt_disc_static, dataloader = self.privacy_engine.make_private(
            module=self.disc_static,
            optimizer=self.opt_disc_static,
            data_loader=dataloader,
            noise_multiplier=self.model_config.dp_noise_multiplier,
            max_grad_norm=self.model_config.dp_max_grad_norm
        )

        self.disc_temporal, self.opt_disc_temporal, _ = self.privacy_engine.make_private(
            module=self.disc_temporal,
            optimizer=self.opt_disc_temporal,
            data_loader=dataloader,
            noise_multiplier=self.model_config.dp_noise_multiplier,
            max_grad_norm=self.model_config.dp_max_grad_norm
        )

        return dataloader

    def _combine_features(self, outputs: Dict[str, torch.Tensor]):
        """Concatena continuous e categorical per generator/discriminator."""
        # --- static ---
        static_parts = []
        if "static_cont" in outputs: static_parts.append(outputs["static_cont"])
        if "static_cat" in outputs: static_parts.append(outputs["static_cat"])
        static = torch.cat(static_parts, dim=-1) if static_parts else None

        # --- temporal ---
        temporal_parts = []
        if "temporal_cont" in outputs: temporal_parts.append(outputs["temporal_cont"])
        if "temporal_cat" in outputs: temporal_parts.append(outputs["temporal_cat"])
        temporal = torch.cat(temporal_parts, dim=-1) if temporal_parts else None

        mask = outputs.get("visit_mask")
        return static, temporal, mask

    def train_step(self, batch):
        static_cont, static_cat, temporal_cont, temporal_cat, temporal_mask, visit_times = batch
        batch_size = static_cont.size(0)

        # ---- move to device ----
        static_cont = static_cont.to(self.device)
        static_cat = static_cat.to(self.device)
        temporal_cont = temporal_cont.to(self.device)
        temporal_cat = temporal_cat.to(self.device)
        temporal_mask = temporal_mask.to(self.device)

        real_outputs = {
            "static_cont": static_cont,
            "static_cat": static_cat,
            "temporal_cont": temporal_cont,
            "temporal_cat": temporal_cat,
            "visit_mask": temporal_mask
        }
        real_static, real_temporal, real_mask = self._combine_features(real_outputs)

        # ===== Discriminatori =====
        self.opt_disc_static.zero_grad()
        self.opt_disc_temporal.zero_grad()

        z_static = torch.randn(batch_size, self.model_config.z_static_dim, device=self.device)
        z_temporal = torch.randn(batch_size, self.max_visits, self.model_config.z_temporal_dim, device=self.device)

        fake_outputs = self.generator(z_static, z_temporal, temperature=self.current_temperature)
        fake_static, fake_temporal, fake_mask = self._combine_features(fake_outputs)

        # --- Static Discriminator ---
        d_real_s = self.disc_static(real_static)
        d_fake_s = self.disc_static(fake_static.detach())
        gp_s = gradient_penalty(self.disc_static, real_static, fake_static.detach(), self.device)
        loss_d_static = d_fake_s.mean() - d_real_s.mean() + self.model_config.grad_clip * gp_s
        loss_d_static.backward()
        self.opt_disc_static.step()

        # --- Temporal Discriminator ---
        d_real_t = self.disc_temporal(real_static, real_temporal, real_mask)
        d_fake_t = self.disc_temporal(fake_static.detach(), fake_temporal.detach(), fake_mask)
        gp_t = gradient_penalty(self.disc_temporal, real_temporal, fake_temporal.detach(), self.device,
                                additional_inputs=(real_static, real_mask))
        loss_d_temporal = d_fake_t.mean() - d_real_t.mean() + self.model_config.grad_clip * gp_t
        loss_d_temporal.backward()
        self.opt_disc_temporal.step()

        # ===== Generator =====
        self.opt_gen.zero_grad()
        z_static = torch.randn(batch_size, self.model_config.z_static_dim, device=self.device)
        z_temporal = torch.randn(batch_size, self.max_visits, self.model_config.z_temporal_dim, device=self.device)

        fake_outputs = self.generator(z_static, z_temporal, temperature=self.current_temperature)
        fake_static, fake_temporal, fake_mask = self._combine_features(fake_outputs)

        loss_g = - (self.disc_static(fake_static).mean() + self.disc_temporal(fake_static, fake_temporal, fake_mask).mean())
        loss_g.backward()
        self.opt_gen.step()

        return {
            "generator": loss_g.item(),
            "disc_static": loss_d_static.item(),
            "disc_temporal": loss_d_temporal.item()
        }

    def fit(self, train_data, epochs: int = None):
        epochs = epochs or self.model_config.epochs
        dataset = LongitudinalSequenceDataset(train_data)
        loader = DataLoader(dataset, batch_size=self.model_config.batch_size, shuffle=True, drop_last=True)

        if getattr(self.model_config, "use_dp", False):
            loader = self._attach_privacy_engine(loader)

        for epoch in range(epochs):
            self.set_train()
            losses_epoch = []
            for batch in loader:
                losses = self.train_step(batch)
                losses_epoch.append(losses)

            # Aggiorna temperatura Gumbel
            self.current_temperature = max(
                getattr(self.model_config, "gumbel_temperature_end", 0.1),
                self.current_temperature * getattr(self.model_config, "gumbel_temperature_decay", 0.95)
            )

            mean_losses = {k: np.mean([l[k] for l in losses_epoch]) for k in losses_epoch[0]}
            print(f"[Epoch {epoch}] " +
                  f"G={mean_losses['generator']:.3f} | " +
                  f"D_s={mean_losses['disc_static']:.3f} | " +
                  f"D_t={mean_losses['disc_temporal']:.3f}")

            if self.privacy_engine:
                eps = self.privacy_engine.get_epsilon(getattr(self.model_config, "dp_delta", 1e-5))
                print(f"[DP] ε = {eps:.2f}")

    @torch.no_grad()
    def generate(self, n_samples: int):
        self.generator.eval()
        outputs_list = []

        for _ in range(0, n_samples, self.model_config.batch_size):
            B = min(self.model_config.batch_size, n_samples)
            z_static = torch.randn(B, self.model_config.generator.z_static_dim, device=self.device)
            z_temporal = torch.randn(B, self.max_visits, self.model_config.generator.z_temporal_dim, device=self.device)

            out = self.generator(z_static, z_temporal, temperature=self.current_temperature)
            outputs_list.append(out)

        final = {k: torch.cat([o[k] for o in outputs_list], dim=0)[:n_samples] for k in outputs_list[0]}
        return final

