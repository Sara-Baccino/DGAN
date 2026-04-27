"""
model/dgan.py  [gretel-style v3 — LSTM + channel min/max]
================================================================================
Cambiamenti rispetto alla versione precedente:

  [NUOVO] Calcolo channel min/max in fit()
    Prima dell'inizio del training, calcola per ogni feature continua temporale
    i valori min e max sui dati reali (solo step valid_flag=True).
    I bound vengono passati al generatore via generator.set_channel_bounds().
    Il tempo (visit_times) NON è incluso: ha la propria normalizzazione.

  [NUOVO] Loss module separato
    Tutte le loss sono importate da model.losses (file dedicato).
    Il training loop usa solo le loss necessarie, senza alias privati.

  [SEMPLIFICATO] Training loop
    Rimossi: GP curriculum, lambda_scat warmup coseno, instance noise.
    Rimasti: WGAN, irr, fup, nv, scat, fm, var (intra-paziente), delta, autocorr.
    Ogni loss è attivata solo se il suo lambda > 0 e i dati sono disponibili.

  [SEMPLIFICATO] Architettura
    Generator: LSTM (via generator.py v3)
    TemporalDiscriminator: LSTM (arch="lstm", via discriminator.py v3)

  [INVARIATO]
    - EMA generatore
    - Early stopping su disc_loss
    - Save/Load con channel_bounds nel checkpoint
    - generate() con temperatura configurabile
    - Warm-start followup e n_visits
================================================================================
"""

import copy
import warnings
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List
import time

from model.generator     import DGANGenerator
from model.discriminator import (StaticDiscriminator, TemporalDiscriminator,
                                  prepare_discriminator_inputs)
from utils.losses import (
    wgan_d_loss,
    wgan_g_loss,
    gradient_penalty,
    delta_distribution_loss,
    intra_patient_variance_loss,
    autocorrelation_loss,
    followup_norm_loss,
    n_visits_loss,
    static_cat_marginal_loss,
    irreversibility_loss,
    feature_matching_loss,
    temporal_irr_prevalence_loss,
    check_finite,
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================================================================
# DGAN
# ==================================================================

class DGAN:

    def __init__(self, data_config, model_config, preprocessor, device=None):
        self.data_config  = data_config
        self.model_config = model_config
        self.preprocessor = preprocessor
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.max_len          = data_config.max_len
        self.irreversible_idx = data_config.irreversible_idx

        self.static_dim   = self._calc_static_dim()
        self.temporal_dim = self._calc_temporal_dim()

        self.embed_var_categories: Dict[str, int] = {}
        for var_name in preprocessor.embedding_configs:
            var = next((v for v in data_config.static_cat if v.name == var_name), None)
            if var:
                self.embed_var_categories[var_name] = len(var.mapping)

        # ── Lambda ────────────────────────────────────────────────────
        mc = model_config
        self.lambda_gp_s      = float(mc.lambda_gp_s)
        self.lambda_gp_t      = float(mc.lambda_gp_t)
        self.lambda_irr       = float(mc.alpha_irr)
        self.lambda_fup       = float(mc.lambda_fup)
        self.lambda_nv        = float(mc.lambda_nv)
        self.lambda_scat      = float(mc.lambda_static_cat)
        self.lambda_fm        = float(mc.lambda_fm)
        self.lambda_var       = float(mc.lambda_var)
        self.lambda_delta     = float(mc.lambda_delta)
        self.lambda_autocorr    = float(getattr(mc, "lambda_autocorr", 0.5))
        self.autocorr_max_lag   = int(getattr(mc, "autocorr_max_lag", 2))
        self.lambda_aux         = float(mc.lambda_aux)
        self.lambda_irr_prev    = float(getattr(mc, "lambda_irr_prev", 5.0))

        self.current_temperature = float(mc.gumbel_temperature_start)
        self.temperature_min     = float(mc.temperature_min)

        self.ema_decay     = float(mc.ema_decay)
        self.ema_generator = None

        self.target_probs_static:       Dict[str, torch.Tensor] = {}
        self.target_prevalence_temporal: Dict[str, float]         = {}

        self._build_model()

        self.loss_history: Dict[str, List] = {
            "generator":      [],
            "disc_static":    [],
            "disc_temporal":  [],
            "irr_loss":       [],
            "fup_loss":       [],
            "nv_loss":        [],
            "scat_loss":      [],
            "fm_loss":        [],
            "var_loss":       [],
            "delta_loss":     [],
            "autocorr_loss":  [],
            "irr_prev_loss":  [],
            "mean_n_visits":  [],
            "fake_cont_mean": [],
            "fake_cont_std":  [],
        }

    # ─────────────────────────────────────────────────────────────────

    def _calc_static_dim(self) -> int:
        dim = self.data_config.n_static_cont
        for i, k in enumerate(self.data_config.n_static_cat):
            name = self.data_config.static_cat[i].name
            if name not in self.preprocessor.embedding_configs:
                dim += k
            else:
                dim += self.preprocessor.embedding_configs[name]
        return dim

    def _calc_temporal_dim(self) -> int:
        dim = self.data_config.n_temp_cont
        for v in self.preprocessor.vars:
            if v.static or v.kind != "categorical":
                continue
            dim += 2 if v.irreversible else len(v.mapping)
        dim += 1   # followup_norm
        return dim

    def _build_model(self):
        mc = self.model_config

        self.generator = DGANGenerator(
            data_config    = self.data_config,
            preprocessor   = self.preprocessor,
            z_static_dim   = mc.z_static_dim,
            z_temporal_dim = mc.z_temporal_dim,
            hidden_dim     = mc.generator.hidden_dim,
            n_layers       = mc.generator.n_layers,
            dropout        = mc.generator.dropout,
            noise_ar_rho   = mc.noise_ar_rho,
            min_visits     = self.data_config.min_visits,
            device         = self.device,
        ).to(self.device)

        self.disc_static = StaticDiscriminator(
            input_dim            = self.static_dim,
            hidden               = mc.static_discriminator.mlp_hidden_dim,
            static_layers        = mc.static_discriminator.static_layers,
            dropout              = mc.static_discriminator.dropout,
            embed_var_categories = self.embed_var_categories,
        ).to(self.device)

        self.disc_temporal = TemporalDiscriminator(
            static_dim   = self.static_dim,
            temporal_dim = self.temporal_dim,
            model_config = mc,
        ).to(self.device)

        if hasattr(self.preprocessor, "embeddings"):
            for name in self.preprocessor.embeddings:
                self.preprocessor.embeddings[name].to(self.device)

        b1, b2 = mc.optimizer_beta1, mc.optimizer_beta2

        self.opt_gen = torch.optim.Adam(
            list(self.generator.parameters())
            + (list(self.preprocessor.embeddings.parameters())
               if self.preprocessor.embeddings else []),
            lr=mc.lr_g, betas=(b1, b2),
        )
        self.opt_disc_static = torch.optim.Adam(
            self.disc_static.parameters(), lr=mc.lr_d_s, betas=(b1, b2))
        self.opt_disc_temporal = torch.optim.Adam(
            self.disc_temporal.parameters(), lr=mc.lr_d_t, betas=(b1, b2))

        if self.ema_decay > 0.0:
            self.ema_generator = copy.deepcopy(self.generator)
            for p in self.ema_generator.parameters():
                p.requires_grad_(False)
        else:
            self.ema_generator = None

    # ─────────────────────────────────────────────────────────────────

    def _update_ema(self):
        if self.ema_generator is None:
            return
        decay = self.ema_decay
        for ema_p, gen_p in zip(self.ema_generator.parameters(),
                                 self.generator.parameters()):
            ema_p.data.mul_(decay).add_(gen_p.data, alpha=1.0 - decay)

    def set_train(self):
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()

    def _move(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {n: t.to(self.device) for n, t in v.items()}
            else:
                out[k] = v
        return out

    def _extract_real_irr(self, batch: Dict) -> Optional[torch.Tensor]:
        if not self.irreversible_idx:
            return None
        return torch.stack([
            batch["temporal_cat"][self.data_config.temporal_cat[idx].name][:, :, 1]
            for idx in self.irreversible_idx
        ], dim=-1)

    def _extract_fake_irr(self, fake_cat: Dict) -> torch.Tensor:
        return torch.stack([
            fake_cat[self.data_config.temporal_cat[idx].name][:, :, 1]
            for idx in self.irreversible_idx
        ], dim=-1)

    def _build_embed_targets(self, batch: Dict) -> Dict:
        targets = {}
        for var_name in self.embed_var_categories:
            embed = batch.get("static_cat_embed", {})
            if var_name in embed:
                p = embed[var_name]
                if p.dim() == 1:
                    targets[var_name] = p
        return targets

    # ─────────────────────────────────────────────────────────────────
    # CHANNEL BOUNDS: calcola min/max reali e li imposta nel generatore
    # ─────────────────────────────────────────────────────────────────

    def _compute_channel_bounds(self, tensors_dict: Dict) -> None:
        """
        Calcola per ogni feature continua temporale i valori min e max
        sui dati reali (solo step valid_flag=True) e li imposta nel generatore.

        Il TEMPO non è incluso: è gestito dalla struttura ratio-scaling
        del generatore, non da un clamping esplicito.

        I bound vengono salvati come buffer nel generatore e persistono
        nel checkpoint via save/load.
        """
        tc = tensors_dict.get("temporal_cont")
        vf = tensors_dict.get("valid_flag")

        if tc is None or self.data_config.n_temp_cont == 0:
            return   # nessuna feature continua temporale

        if vf is not None:
            vf_bool = vf.bool()
            if vf_bool.dim() == 3:
                vf_bool = vf_bool.squeeze(-1)
        else:
            vf_bool = torch.ones(tc.shape[0], tc.shape[1], dtype=torch.bool)

        n_cont = tc.shape[-1]
        ch_min = torch.zeros(n_cont)
        ch_max = torch.zeros(n_cont)

        for j in range(n_cont):
            vals = tc[:, :, j][vf_bool]
            if vals.numel() > 0:
                ch_min[j] = float(vals.min())
                ch_max[j] = float(vals.max())
            else:
                ch_min[j] = float("-inf")
                ch_max[j] = float("+inf")

        self.generator.set_channel_bounds(ch_min, ch_max)
        logger.info(
            f"Channel bounds impostati per {n_cont} feature continue temporali.")
        print(f"  Channel bounds: min={ch_min.tolist()}  max={ch_max.tolist()}")

    # ─────────────────────────────────────────────────────────────────
    # GENERATE FAKE
    # ─────────────────────────────────────────────────────────────────

    def _generate_fake(
        self,
        batch_size: int,
        real_irr:   Optional[torch.Tensor] = None,
    ):
        z_s, z_t = self.generator.sample_noise(batch_size, torch.device(self.device))
        fake_out  = self.generator(
            z_s, z_t,
            temperature = self.current_temperature,
            real_irr    = real_irr,
        )
        fake_disc = prepare_discriminator_inputs(fake_out, self.preprocessor)
        return fake_out, fake_disc

    # ─────────────────────────────────────────────────────────────────
    # TRAIN DISCRIMINATORS
    # ─────────────────────────────────────────────────────────────────

    def _train_discriminators(
        self,
        real_disc:       Dict,
        embed_targets:   Dict,
        batch_size:      int,
        update_static:   bool = True,
        update_temporal: bool = True,
    ):
        if batch_size < 2:
            warnings.warn(
                f"Batch size = {batch_size} è troppo piccolo per il gradient penalty. "
                f"Raccomandato >= 16.",
                UserWarning,
            )

        with torch.no_grad():
            _, fake_disc = self._generate_fake(batch_size)

        real_s  = real_disc["static"].detach()
        real_t  = real_disc["temporal"].detach()
        fake_s  = fake_disc["static"].detach()
        fake_t  = fake_disc["temporal"].detach()
        real_vf = real_disc["valid_flag"]
        fake_vf = fake_disc["valid_flag"]

        # ── Static discriminator ──────────────────────────────────────
        d_real_s = self.disc_static(real_s)
        d_fake_s = self.disc_static(fake_s)
        aux_loss = self.disc_static.auxiliary_loss(real_s, embed_targets)

        if self.lambda_gp_s > 0:
            gp_s = gradient_penalty(
                lambda x: self.disc_static(x), real_s, fake_s,
                self.device, lambda_gp=self.lambda_gp_s)
        else:
            gp_s = torch.tensor(0.0, device=self.device)

        loss_d_s = (wgan_d_loss(d_real_s, d_fake_s)
                    + gp_s
                    + self.lambda_aux * aux_loss)
        loss_d_s = check_finite(loss_d_s, "disc_static")

        if update_static:
            self.opt_disc_static.zero_grad()
            loss_d_s.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.disc_static.parameters(), self.model_config.grad_clip)
            self.opt_disc_static.step()

        # ── Temporal discriminator ────────────────────────────────────
        d_real_t = self.disc_temporal(real_s, real_t, real_vf)
        d_fake_t = self.disc_temporal(fake_s, fake_t, fake_vf)

        if self.lambda_gp_t > 0:
            gp_t = gradient_penalty(
                lambda x: self.disc_temporal(real_s, x, real_vf), real_t, fake_t,
                self.device, lambda_gp=self.lambda_gp_t)
        else:
            gp_t = torch.tensor(0.0, device=self.device)

        loss_d_t = wgan_d_loss(d_real_t, d_fake_t) + gp_t
        loss_d_t = check_finite(loss_d_t, "disc_temporal")

        if update_temporal:
            self.opt_disc_temporal.zero_grad()
            loss_d_t.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.disc_temporal.parameters(), self.model_config.grad_clip)
            self.opt_disc_temporal.step()

        return loss_d_s.item(), loss_d_t.item(), aux_loss.item()

    # ─────────────────────────────────────────────────────────────────
    # TRAIN GENERATOR
    # ─────────────────────────────────────────────────────────────────

    def _train_generator(
        self,
        real_disc:  Dict,
        batch_size: int,
        real_irr:   Optional[torch.Tensor],
        real_batch: Optional[Dict],
    ):
        self.opt_gen.zero_grad()
        # --- TIMER 1: IL PROCESSO DI GENERAZIONE ---
        #t_start_gen = time.time()

        fake_out, fake_disc = self._generate_fake(batch_size, real_irr=real_irr)
        #dt_inference = time.time() - t_start_gen

        # --- TIMER 2: IL CALCOLO DELLE LOSS ---
        #t_start_loss = time.time()
        # ── WGAN generator loss ───────────────────────────────────────
        # Il disc_temporal usa fake_static come conditioning:
        # i gradienti devono fluire attraverso entrambi i path → no detach.
        d_fake_s = self.disc_static(fake_disc["static"])
        d_fake_t = self.disc_temporal(
            fake_disc["static"], fake_disc["temporal"], fake_disc["valid_flag"])
        loss_g = wgan_g_loss(d_fake_s, d_fake_t)
        loss_g = check_finite(loss_g, "generator")

        # ── Irreversibilità ───────────────────────────────────────────
        #t_sub = time.time()
        irr_loss = torch.tensor(0.0, device=self.device)
        if self.irreversible_idx and self.lambda_irr > 0:
            irr_states = self._extract_fake_irr(fake_out["temporal_cat"])
            for k in range(irr_states.shape[-1]):
                irr_loss = irr_loss + irreversibility_loss(
                    irr_states[..., k], fake_out["valid_flag"])
            irr_loss = check_finite(irr_loss, "irr_loss")
        #dt_irr = time.time() - t_sub
        
        # ── Followup supervision ──────────────────────────────────────
        #t_sub = time.time()
        fup_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_fup > 0
                and real_batch is not None
                and "followup_norm" in real_batch):
            fup_loss = followup_norm_loss(
                fake_out["followup_norm"],
                real_batch["followup_norm"].to(self.device))
            fup_loss = check_finite(fup_loss, "fup_loss")
        #dt_fup = time.time() - t_sub

        # ── N_visits supervision ──────────────────────────────────────
        #t_sub = time.time()
        nv_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_nv > 0
                and real_batch is not None
                and "n_visits" in real_batch):
            nv_loss = n_visits_loss(
                fake_out["n_visits_pred"],
                real_batch["n_visits"].float().to(self.device))
            nv_loss = check_finite(nv_loss, "nv_loss")
        #dt_nv = time.time() - t_sub
        # ── Static categorical marginal ───────────────────────────────
        #t_sub = time.time()
        scat_loss = torch.tensor(0.0, device=self.device)
        fake_soft = fake_out.get("static_cat_soft") or {}
        if self.lambda_scat > 0 and self.target_probs_static and fake_soft:
            scat_loss = static_cat_marginal_loss(fake_soft, self.target_probs_static)
            scat_loss = check_finite(scat_loss, "scat_loss")
        #dt_cat = time.time() - t_sub

        # ── Feature matching ──────────────────────────────────────────
        #t_sub = time.time()
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fm > 0 and hasattr(self.disc_static, "get_features"):
            feat_real = self.disc_static.get_features(
                real_disc["static"]).detach()
            feat_fake = self.disc_static.get_features(fake_disc["static"])
            fm_loss   = feature_matching_loss(feat_real, feat_fake)
            fm_loss   = check_finite(fm_loss, "fm_loss")
        #dt_fm = time.time() - t_sub

        # ── Varianza intra-paziente ───────────────────────────────────
        #t_sub = time.time()
        var_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_var > 0
                and real_batch is not None
                and "temporal_cont" in real_batch
                and fake_out["temporal_cont"].shape[-1] > 0):
            var_loss = intra_patient_variance_loss(
                fake_out["temporal_cont"],
                real_batch["temporal_cont"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
            )
            var_loss = check_finite(var_loss, "var_loss")
        #dt_intra = time.time() - t_sub

        # ── Delta distribution loss ───────────────────────────────────
        #t_sub = time.time()
        delta_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_delta > 0
                and real_batch is not None
                and "visit_time" in real_batch
                and "deltas" in fake_out):
            delta_loss = delta_distribution_loss(
                fake_out["deltas"],
                real_batch["visit_time"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
            )
            delta_loss = check_finite(delta_loss, "delta_loss")
        #dt_delta = time.time() - t_sub

        # ── Autocorrelation loss ──────────────────────────────────────
        #t_sub = time.time()
        autocorr_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_autocorr > 0
                and real_batch is not None
                and "temporal_cont" in real_batch
                and fake_out["temporal_cont"].shape[-1] > 0):
            autocorr_loss = autocorrelation_loss(
                fake_out["temporal_cont"],
                real_batch["temporal_cont"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
                max_lag = self.autocorr_max_lag,
            )
            autocorr_loss = check_finite(autocorr_loss, "autocorr_loss")
        #dt_autocorr = time.time() - t_sub

        # ── Irreversible prevalence loss ───────────────────────────────────────
        #t_sub = time.time()
        irr_prev_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_irr_prev > 0
                and self.target_prevalence_temporal
                and fake_out.get("temporal_cat")):
            irr_prev_loss = temporal_irr_prevalence_loss(
                fake_temporal_cat = fake_out["temporal_cat"],
                target_prevalence = self.target_prevalence_temporal,
                valid_flag        = fake_out["valid_flag"],
            )
            irr_prev_loss = check_finite(irr_prev_loss, "irr_prev_loss")
        #dt_irrp = time.time() - t_sub
        #dt_all_losses = time.time() - t_start_loss

        # --- TIMER 3: BACKPROPAGATION ---
        #t_start_back = time.time()
        # ── Total ─────────────────────────────────────────────────────
        total = (loss_g
                 + self.lambda_irr      * irr_loss
                 + self.lambda_fup      * fup_loss
                 + self.lambda_nv       * nv_loss
                 + self.lambda_scat     * scat_loss
                 + self.lambda_fm       * fm_loss
                 + self.lambda_var      * var_loss
                 + self.lambda_delta    * delta_loss
                 + self.lambda_autocorr * autocorr_loss
                 + self.lambda_irr_prev * irr_prev_loss)

        total = check_finite(total, "total_generator")
        total.backward()

        all_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            all_params += list(self.preprocessor.embeddings.parameters())
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, self.model_config.grad_clip)
        self.opt_gen.step()
        self._update_ema()

        #dt_backprop = time.time() - t_start_back
        #print(f"\n[GEN PROFILE] Inf: {dt_inference:.2f}s | Losses: {dt_all_losses:.2f}s | Backprop: {dt_backprop:.2f}s")
        #print(f"\n              AcL: {dt_autocorr:.2f}s | Irr: {dt_irr:.2f}s | Fup: {dt_fup:.2f}s | Cat: {dt_cat:.2f}s | Nv: {dt_nv:.2f}s")
        #print(f"\n              IntraVar: {dt_intra:.2f}s | Fm: {dt_fm:.2f}s | Delta: {dt_delta:.2f}s | IrrP: {dt_nv:.2f}s")

        mean_nv = float(fake_out["n_visits"].detach().mean())

        stats = {}
        if fake_out["temporal_cont"].shape[-1] > 0:
            vf_f = fake_out["valid_flag"]
            mask = vf_f.unsqueeze(-1).expand_as(fake_out["temporal_cont"][:, :, :3])
            vals = fake_out["temporal_cont"][:, :, :3][mask]
            stats["fake_cont_mean"] = float(vals.mean()) if vals.numel() > 0 else 0.0
            stats["fake_cont_std"]  = float(vals.std())  if vals.numel() > 0 else 0.0
        else:
            stats["fake_cont_mean"] = 0.0
            stats["fake_cont_std"]  = 0.0

        return (loss_g.item(), irr_loss.item(), fup_loss.item(),
                nv_loss.item(), scat_loss.item(), fm_loss.item(),
                var_loss.item(), delta_loss.item(), autocorr_loss.item(),
                irr_prev_loss.item(), mean_nv, stats)

    # ─────────────────────────────────────────────────────────────────
    # LOADER
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_loader(tensors_dict: Dict, batch_size: int,
                      use_dp: bool, drop_last: bool, num_workers: int = 0, pin_memory: bool = False):
        tensors, keys = [], []

        for k in ["static_cont", "static_cat", "temporal_cont",
                  "valid_flag", "visit_time", "followup_norm", "n_visits"]:
            if k in tensors_dict and tensors_dict[k] is not None:
                tensors.append(tensors_dict[k])
                keys.append(k)

        for name, t in tensors_dict.get("temporal_cat", {}).items():
            tensors.append(t); keys.append(f"tcat::{name}")

        for name, t in tensors_dict.get("static_cat_embed", {}).items():
            tensors.append(t); keys.append(f"sce::{name}")

        if len(tensors) == 0:
            raise ValueError(
                "tensors_dict è vuoto o non contiene tensori validi. "
                "Verifica che fit_transform sia stato chiamato correttamente."
            )

        loader = DataLoader(
            TensorDataset(*tensors),
            batch_size = batch_size,
            shuffle    = True,
            drop_last  = drop_last,
            num_workers = num_workers,    # <-- Imposta i worker
            pin_memory  = pin_memory,     # <-- Accelera il trasferimento CPU -> GPU
        )

        if len(loader) == 0:
            raise ValueError(
                f"Il DataLoader è vuoto. Il dataset ha {len(tensors[0])} campioni "
                f"e batch_size={batch_size}. Riduci batch_size o aumenta il dataset."
            )

        return loader, keys

    @staticmethod
    def _reconstruct_batch(batch_tuple, keys: List[str]) -> Dict:
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if key.startswith("tcat::"):
                batch.setdefault("temporal_cat",    {})[key[6:]] = tensor
            elif key.startswith("sce::"):
                batch.setdefault("static_cat_embed", {})[key[5:]] = tensor
            else:
                batch[key] = tensor
        return batch

    # ─────────────────────────────────────────────────────────────────
    # TRAIN STEP
    # ─────────────────────────────────────────────────────────────────

    def train_step(self, loader, keys, critic_steps, critic_steps_t):
        batch_losses = []

        for batch_tuple in loader:
            #start_batch = time.time()
            # 1. Spostamento dati su GPU
            #t0 = time.time()
            batch = self._reconstruct_batch(batch_tuple, keys)
            batch = self._move(batch)
            #dt_move = time.time() - t0
            
            B     = batch["temporal_cont"].shape[0]

            real_disc     = prepare_discriminator_inputs(batch, self.preprocessor)
            real_irr      = self._extract_real_irr(batch)
            embed_targets = self._build_embed_targets(batch)

            n_steps    = max(critic_steps, critic_steps_t)
            ld_s_list, ld_t_list, aux_list = [], [], []

            # 2. Training Discriminatori
            #t0 = time.time()
            for step_idx in range(n_steps):
                ld_s, ld_t, aux = self._train_discriminators(
                    real_disc, embed_targets, B,
                    update_static   = (step_idx < critic_steps),
                    update_temporal = (step_idx < critic_steps_t),
                )
                ld_s_list.append(ld_s)
                ld_t_list.append(ld_t)
                aux_list.append(aux)

            #dt_disc = time.time() - t0
        
            # 3. Training Generatore (spesso il collo di bottiglia per le molteplici loss)
            #t0 = time.time()
            (lg, l_irr, l_fup, l_nv, l_scat, l_fm,
             l_var, l_delta, l_ac, l_irr_prev, mean_nv, stats) = self._train_generator(
                real_disc, B, real_irr, real_batch=batch)
            #dt_gen = time.time() - t0
            
            #print(f"\r[Profiling] Move: {dt_move:.4f}s | Disc: {dt_disc:.4f}s | Gen: {dt_gen:.4f}s", end="")
            
            batch_losses.append({
                "generator":      lg,
                "disc_static":    float(np.mean(ld_s_list)),
                "disc_temporal":  float(np.mean(ld_t_list)),
                "aux_loss":       float(np.mean(aux_list)),
                "irr_loss":       l_irr,
                "fup_loss":       l_fup,
                "nv_loss":        l_nv,
                "scat_loss":      l_scat,
                "fm_loss":        l_fm,
                "var_loss":       l_var,
                "delta_loss":     l_delta,
                "autocorr_loss":  l_ac,
                "irr_prev_loss":  l_irr_prev,
                "mean_n_visits":  mean_nv,
                "fake_cont_mean": stats["fake_cont_mean"],
                "fake_cont_std":  stats["fake_cont_std"],
            })

        return batch_losses

    # ─────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────


    # ─────────────────────────────────────────────────────────────────
    # TARGET PREVALENCE VARIABILI IRREVERSIBILI TEMPORALI
    # ─────────────────────────────────────────────────────────────────

    def _compute_irr_prevalence_targets(self, tensors_dict: Dict) -> None:
        """
        Calcola la prevalenza reale (frazione di pazienti con stato=1
        nell'ultima visita valida) per ogni variabile temporale binaria
        irreversibile. Salvata in self.target_prevalence_temporal.

        Chiamata una volta sola all'inizio del fit().
        """
        tcat = tensors_dict.get("temporal_cat")
        vf   = tensors_dict.get("valid_flag")
        if tcat is None or vf is None:
            return

        vf_bool = vf.bool() if isinstance(vf, torch.Tensor) else torch.tensor(vf).bool()

        irr_names = [
            self.data_config.temporal_cat[idx].name
            for idx in self.irreversible_idx
        ]

        for var in self.data_config.temporal_cat:
            if var.name not in irr_names:
                continue
            if var.name not in tcat:
                continue

            cat_tensor = tcat[var.name]   # [N, T, n_cat] or [N, T] (encoded int)
            N = cat_tensor.shape[0]
            n_events = 0

            for b in range(N):
                valid_idx = vf_bool[b].nonzero(as_tuple=True)[0]
                if len(valid_idx) == 0:
                    continue
                last_t = int(valid_idx[-1])
                if cat_tensor.dim() == 3:
                    # OHE: [N, T, 2] → stato=1 se argmax==1
                    state = int(torch.argmax(cat_tensor[b, last_t]))
                else:
                    # Encoded int: 1=stato_0, 2=stato_1 (mapping gretel)
                    state = int(cat_tensor[b, last_t]) - 1  # 0 o 1
                if state == 1:
                    n_events += 1

            prevalence = n_events / max(N, 1)
            self.target_prevalence_temporal[var.name] = prevalence
            print(f"  [irr_prev] {var.name}: prevalence={prevalence:.3f} "
                  f"({n_events}/{N} pazienti con stato=1)")

    def fit(self, tensors_dict: Dict, epochs: int = None):
        self.set_train()
        epochs = epochs or self.model_config.epochs

        print(f"\n{'='*70}")
        print(f"  DGAN [gretel-style v3 — LSTM]  —  {epochs} epoche")
        print(f"  Device: {self.device}  |  batch_size: {self.model_config.batch_size}")
        print(f"  z_s={self.model_config.z_static_dim}  "
              f"z_t={self.model_config.z_temporal_dim}  "
              f"noise_ar_rho={self.model_config.noise_ar_rho}  "
              f"min_visits={self.data_config.min_visits}")
        print(f"  ema_decay={self.ema_decay}")
        print(f"  λ_gp_s={self.lambda_gp_s}  λ_gp_t={self.lambda_gp_t}  "
              f"λ_fup={self.lambda_fup}  λ_nv={self.lambda_nv}  "
              f"λ_var={self.lambda_var}  λ_scat={self.lambda_scat}  "
              f"λ_delta={self.lambda_delta}  λ_autocorr={self.lambda_autocorr}")
        print(f"{'='*70}\n")

        # ── Channel min/max dal dataset reale ─────────────────────────
        self._compute_channel_bounds(tensors_dict)

        # ── Target prevalence variabili irreversibili ─────────────────
        print("Calcolo prevalenza target variabili irreversibili...")
        self._compute_irr_prevalence_targets(tensors_dict)

        loader, keys = self._build_loader(
            tensors_dict,
            self.model_config.batch_size,
            self.model_config.use_dp,
            self.model_config.dataloader_drop_last,
            num_workers = 2,
        )

        if self.model_config.use_dp and OPACUS_AVAILABLE:
            privacy_engine = PrivacyEngine()
            self.disc_static, self.opt_disc_static, loader = (
                privacy_engine.make_private(
                    module           = self.disc_static,
                    optimizer        = self.opt_disc_static,
                    data_loader      = loader,
                    noise_multiplier = self.model_config.noise_std,
                    max_grad_norm    = self.model_config.grad_clip))
        elif self.model_config.use_dp and not OPACUS_AVAILABLE:
            warnings.warn(
                "use_dp=true ma Opacus non è installato. "
                "Training senza privacy differenziale.",
                UserWarning,
            )

        # ── Target probs statiche ─────────────────────────────────────
        scat_tensor = tensors_dict.get("static_cat")
        if self.lambda_scat > 0 and scat_tensor is not None:
            print("Calcolo distribuzione marginale categoriche statiche...")
            offset = 0
            for var in self.data_config.static_cat:
                if var.name in self.preprocessor.embedding_configs:
                    continue
                n   = var.n_categories
                ohe = scat_tensor[:, offset:offset + n].float()
                freq = ohe.mean(dim=0)
                self.target_probs_static[var.name] = (
                    freq / freq.sum().clamp(min=1e-8)).to(self.device)
                offset += n

                # Warm-start: inizializza il bias della head categorica
                head = dict(self.generator.static_cat_heads).get(var.name)
                if head is not None:
                    p         = self.target_probs_static[var.name].cpu()
                    log_prior = torch.log(p.clamp(min=1e-6))
                    log_prior = log_prior - log_prior.mean()
                    last_layer = head
                    if isinstance(head, nn.Sequential):
                        for m in reversed(list(head.modules())):
                            if isinstance(m, nn.Linear):
                                last_layer = m; break
                    with torch.no_grad():
                        if (hasattr(last_layer, "bias")
                                and last_layer.bias is not None):
                            last_layer.bias.data.copy_(
                                log_prior.to(last_layer.bias.device))
            print(f"  -> {len(self.target_probs_static)} variabili")

        # ── Warm-start followup ───────────────────────────────────────
        fn_all = tensors_dict.get("followup_norm")
        if fn_all is not None:
            fn_mean  = float(fn_all.float().mean().clamp(0.02, 0.98))
            fn_logit = float(np.log(fn_mean / (1.0 - fn_mean)))
            with torch.no_grad():
                self.generator.followup_head[-2].bias.fill_(fn_logit)
            print(f"  followup warm-start: mean={fn_mean:.3f}  logit={fn_logit:.3f}")

        # ── Warm-start n_visits ───────────────────────────────────────
        nv_all = tensors_dict.get("n_visits")
        if nv_all is not None:
            nv_med = float(nv_all.float().median())
            min_v  = self.data_config.min_visits
            nv_med = max(nv_med, float(min_v))
            target = max(nv_med - 1.0, 0.1)
            nv_bias = float(np.log(np.exp(target) - 1.0 + 1e-6))
            with torch.no_grad():
                self.generator.n_visits_head[-1].bias.fill_(nv_bias)
            print(f"  n_visits warm-start: median={nv_med:.1f}  min_visits={min_v}")

        critic_steps   = self.model_config.critic_steps
        critic_steps_t = self.model_config.critic_steps_temporal
        best_loss, patience_counter = float("inf"), 0

        # ── Training loop ─────────────────────────────────────────────
        for epoch in range(epochs):
            self.current_temperature = max(
                self.temperature_min,
                self.current_temperature * 0.995)

            batch_losses = self.train_step(
                loader, keys, critic_steps, critic_steps_t)

            avg = {k: float(np.mean([b[k] for b in batch_losses]))
                   for k in batch_losses[0]}
            for k, v in avg.items():
                if k in self.loss_history:
                    self.loss_history[k].append(v)

            print(
                f"[Ep {epoch+1:4d}/{epochs}]  "
                f"G={avg['generator']:+7.3f}  "
                f"D_s={avg['disc_static']:+7.3f}  "
                f"D_t={avg['disc_temporal']:+7.3f}"
            )
            print(
                f"              "
                f"Nv={avg['mean_n_visits']:4.1f}  "
                f"NvL={avg['nv_loss']:.3f}  "
                f"Fup={avg['fup_loss']:.3f}  "
                f"Scat={avg['scat_loss']:.3f}  "
                f"Fm={avg['fm_loss']:.3f}  "
                f"VarL={avg['var_loss']:.3f}  "
                f"dL={avg['delta_loss']:.3f}  "
                f"AcL={avg['autocorr_loss']:.3f}  "
                f"IrrP={avg['irr_prev_loss']:.3f}  "
                f"Irr={avg['irr_loss']:.3f}"
            )
            print(
                f"              "
                f"Cont(fake) mu={avg['fake_cont_mean']:+.3f}  "
                f"s={avg['fake_cont_std']:.3f}  "
                f"T={self.current_temperature:.3f}",
                flush=True,
            )

            disc_loss = avg["disc_static"] + avg["disc_temporal"]
            if disc_loss < best_loss:
                best_loss = disc_loss; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    print(f"\nEarly stopping — epoch {epoch+1} "
                          f"(best disc_loss={best_loss:.4f})")
                    break

    # ─────────────────────────────────────────────────────────────────
    # GENERATE
    # ─────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n_samples: int, temperature: float = 0.5,
                 return_dataframe: bool = True):
        self.set_eval()
        gen = self.ema_generator if self.ema_generator is not None else self.generator
        if self.ema_generator is not None:
            self.ema_generator.eval()

        all_outputs, remaining = [], n_samples

        while remaining > 0:
            bs       = min(self.model_config.batch_size, remaining)
            z_s, z_t = gen.sample_noise(bs, torch.device(self.device))
            out      = gen(z_s, z_t, temperature=temperature)

            all_outputs.append({
                k: (v.cpu().numpy() if torch.is_tensor(v)
                    else {n: t.cpu().numpy() for n, t in v.items()}
                    if isinstance(v, dict) else v)
                for k, v in out.items() if v is not None
            })
            remaining -= bs

        final: Dict = {}
        for k in all_outputs[0]:
            if isinstance(all_outputs[0][k], dict):
                final[k] = {
                    n: np.concatenate([o[k][n] for o in all_outputs], axis=0)[:n_samples]
                    for n in all_outputs[0][k]
                }
            elif isinstance(all_outputs[0][k], np.ndarray):
                final[k] = np.concatenate(
                    [o[k] for o in all_outputs], axis=0)[:n_samples]

        if "valid_flag" not in final:
            vf_arrays = [o["valid_flag"] for o in all_outputs if "valid_flag" in o]
            if vf_arrays:
                final["valid_flag"] = np.concatenate(vf_arrays, axis=0)[:n_samples]

        if "static_cat_embed" in final and final["static_cat_embed"]:
            decoded = self.preprocessor.decode_embeddings({
                n: torch.tensor(v, device=self.device)
                for n, v in final["static_cat_embed"].items()})
            final["static_cat_embed_decoded"] = {
                n: idx.cpu().numpy() for n, idx in decoded.items()}
            del final["static_cat_embed"]

        if not return_dataframe:
            self.set_train()
            return final

        synth = {
            "temporal_cont": torch.tensor(final["temporal_cont"]),
            "temporal_cat":  {n: torch.tensor(v)
                              for n, v in final["temporal_cat"].items()},
            "valid_flag":    torch.tensor(final["valid_flag"]),
            "visit_times":   torch.tensor(final["visit_times"]),
        }
        if "followup_norm" in final:
            synth["followup_norm"] = torch.tensor(final["followup_norm"])
        if "static_cont" in final:
            synth["static_cont"] = torch.tensor(final["static_cont"])
        if "static_cat_embed_decoded" in final:
            synth["static_cat_embed_decoded"] = {
                n: torch.tensor(v)
                for n, v in final["static_cat_embed_decoded"].items()}
        if "static_cat" in final and final["static_cat"]:
            sc_names = [v.name for v in self.data_config.static_cat
                        if v.name not in self.preprocessor.embedding_configs]
            arrays   = [final["static_cat"][k] for k in sc_names
                        if k in final["static_cat"]]
            if arrays:
                synth["static_cat"] = torch.from_numpy(
                    np.concatenate(arrays, axis=1)).float()

        self.set_train()
        return self.preprocessor.inverse_transform(synth)

    # ─────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────

    def save(self, filepath: str):
        state = {
            "generator_state":         self.generator.state_dict(),
            "disc_static_state":       self.disc_static.state_dict(),
            "disc_temporal_state":     self.disc_temporal.state_dict(),
            "opt_gen_state":           self.opt_gen.state_dict(),
            "opt_disc_static_state":   self.opt_disc_static.state_dict(),
            "opt_disc_temporal_state": self.opt_disc_temporal.state_dict(),
            "loss_history":            self.loss_history,
            "current_temperature":     self.current_temperature,
            "embedding_configs":       self.preprocessor.embedding_configs,
            "scalers_cont":            self.preprocessor.scalers_cont,
            "inverse_maps":            self.preprocessor.inverse_maps,
            "global_time_max":         self.preprocessor.global_time_max,
            # [NUOVO] channel bounds salvati nel checkpoint
            "channel_min":             self.generator.channel_min.cpu(),
            "channel_max":             self.generator.channel_max.cpu(),
        }
        if self.preprocessor.embeddings:
            state["embedding_state"] = {
                name: layer.state_dict()
                for name, layer in self.preprocessor.embeddings.items()}
        if self.ema_generator is not None:
            state["ema_generator_state"] = self.ema_generator.state_dict()
        torch.save(state, filepath)
        logger.info(f"Model saved → {filepath}")
        print(f"  Modello salvato → {filepath}")

    @classmethod
    def load(cls, filepath: str, data_config, model_config, preprocessor,
             device=None):
        state = torch.load(filepath, map_location=device or "cpu")
        dgan  = cls(data_config, model_config, preprocessor, device=device)
        dgan.generator.load_state_dict(state["generator_state"])
        dgan.disc_static.load_state_dict(state["disc_static_state"])
        dgan.disc_temporal.load_state_dict(state["disc_temporal_state"])
        dgan.opt_gen.load_state_dict(state["opt_gen_state"])
        dgan.opt_disc_static.load_state_dict(state["opt_disc_static_state"])
        dgan.opt_disc_temporal.load_state_dict(state["opt_disc_temporal_state"])
        dgan.loss_history               = state["loss_history"]
        dgan.current_temperature        = state["current_temperature"]
        dgan.preprocessor.embedding_configs = state["embedding_configs"]
        dgan.preprocessor.scalers_cont  = state["scalers_cont"]
        dgan.preprocessor.inverse_maps  = state["inverse_maps"]
        dgan.preprocessor.global_time_max = state["global_time_max"]

        # Ripristina channel bounds nel generatore
        if "channel_min" in state and "channel_max" in state:
            dgan.generator.set_channel_bounds(
                state["channel_min"], state["channel_max"])

        if "embedding_state" in state:
            for name, emb_state in state["embedding_state"].items():
                if name in dgan.preprocessor.embeddings:
                    dgan.preprocessor.embeddings[name].load_state_dict(emb_state)
        if "ema_generator_state" in state and dgan.ema_generator is not None:
            dgan.ema_generator.load_state_dict(state["ema_generator_state"])

        logger.info(f"Model loaded ← {filepath}")
        print(f"  Modello caricato ← {filepath}")
        return dgan