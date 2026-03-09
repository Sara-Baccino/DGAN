"""
model/dgan.py  [v9]
================================================================================
Modifiche rispetto a v8:

  1. StructuredTemporalNoise integrato:
     DGAN._generate_fake() usa generator.sample_noise() invece di
     torch.randn(...) per z_temporal. I parametri rho, sigma_* del
     noise_model sono nell'optimizer del generatore e vengono logghati.

  2. Curriculum learning sulla visit_mask:
     DGAN.fit() calcola curriculum_p ad ogni epoca e chiama
     generator.set_curriculum_p(p). Schedule:
       - Epoche [0, warmup_mask]:        p=0.0 (sempre real n_visits)
       - Epoche [warmup_mask, T-fine]:   p cresce linearmente da 0 a 1
       - Epoche [T-fine, T]:             p=1.0 (sempre pred n_visits)
     Controllato da:
       warmup_mask_frac:  frazione epoche con p=0 (default 0.20)
       finetune_mask_frac: frazione epoche con p=1 (default 0.15)

  3. Logging arricchito:
     Ogni N epoche stampa i parametri del noise_model:
       rho, w_global, w_ar, w_episod, curriculum_p

  4. z_static_dim coerente:
     sample_noise() usa model_config.z_static_dim (non più da data_config).

  5. Tutto il resto invariato rispetto a v8.
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple, List
import numpy as np
import logging

from model.generator     import HierarchicalGenerator
from model.discriminator import (
    StaticDiscriminator, TemporalDiscriminator,
    prepare_discriminator_inputs,
)
from utils.losses import (
    wgan_discriminator_loss, wgan_generator_loss,
    gradient_penalty, irreversibility_loss,
    compute_category_weights,
    categorical_frequency_loss_generator,
    categorical_frequency_loss_discriminator,
    followup_norm_loss,
    feature_matching_loss,
    n_visits_supervision_loss,
    static_cat_marginal_loss,
    static_cont_dist_loss,
    inter_visit_interval_loss,
    inter_visit_uniformity_loss,
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available — DP disabled.")

logger = logging.getLogger(__name__)


class DGAN:

    def __init__(self, data_config, model_config, preprocessor, device=None):
        self.data_config  = data_config
        self.model_config = model_config
        self.preprocessor = preprocessor
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.static_dim   = self._calculate_static_dim()
        self.temporal_dim = (
            data_config.n_temp_cont
            + sum(
                2 if v.irreversible else len(v.mapping)
                for v in preprocessor.vars
                if not v.static and v.kind == "categorical"
            )
            + 1   # visit time feature
        )
        self.max_len          = data_config.max_len
        self.irreversible_idx = data_config.irreversible_idx

        self.embed_var_categories: Dict[str, int] = {}
        for var_name in preprocessor.embedding_configs:
            var = next((v for v in data_config.static_cat if v.name == var_name), None)
            if var is not None:
                self.embed_var_categories[var_name] = len(var.mapping)

        self._build_model()

        self.privacy_engine        = None
        self.current_temperature   = model_config.gumbel_temperature_start
        self.temperature_min       = model_config.temperature_min
        self.teacher_forcing_prob  = 1.0
        self.teacher_forcing_decay = 0.995
        self.alpha_irr_start = 0.1
        self.alpha_irr_max   = model_config.alpha_irr
        self.alpha_irr       = self.alpha_irr_start

        self.lambda_aux        = getattr(model_config, "lambda_aux",        0.2)
        self.lambda_gp_s       = getattr(model_config, "lambda_gp_s",       4.0)
        self.lambda_gp_t       = getattr(model_config, "lambda_gp_t",       3.0)
        self.lambda_freq_gen   = getattr(model_config, "lambda_freq_gen",    0.15)
        self.lambda_freq_disc  = getattr(model_config, "lambda_freq_disc",   0.05)
        self.freq_weight_power = getattr(model_config, "freq_weight_power",  1.0)
        self.cat_weights: Dict[str, torch.Tensor] = {}
        self.lambda_fm         = getattr(model_config, "lambda_fm",          0.0)
        self.lambda_fup        = getattr(model_config, "lambda_fup",         2.0) or 2.0
        self.lambda_nv         = getattr(model_config, "lambda_nv",          2.0) or 2.0
        self.lambda_coverage   = getattr(model_config, "lambda_coverage",    5.0) or 5.0
        self.lambda_static_cat = getattr(model_config, "lambda_static_cat", 12.0) or 12.0
        self.lambda_ivi        = getattr(model_config, "lambda_ivi",         18.0) or 18.0
        self.lambda_uniformity = getattr(model_config, "lambda_uniformity",   5.0) or 5.0

        # [v9] Curriculum schedule per la visit_mask
        self.warmup_mask_frac   = getattr(model_config, "warmup_mask_frac",   0.20)
        self.finetune_mask_frac = getattr(model_config, "finetune_mask_frac", 0.15)
        self._current_curriculum_p = 0.0

        self.target_probs_static: Dict[str, torch.Tensor] = {}
        self.real_intervals: Optional[torch.Tensor] = None
        self._warned_no_soft = False

        self.loss_history = {
            "generator":[], "disc_static":[], "disc_temporal":[],
            "irreversibility":[], "gp_static":[], "gp_temporal":[],
            "aux_embed":[], "alpha_irr":[], "epsilon":[],
            "freq_gen":[], "freq_disc":[], "mean_n_visits":[],
            "nv_loss":[], "fm_loss":[], "fup_loss":[],
            "static_cat_loss":[], "ivi_loss":[], "coverage_loss":[],
            "uniformity_loss":[], "curriculum_p":[],
            "noise_rho":[], "noise_w_global":[], "noise_w_ar":[], "noise_w_episod":[],
        }

    def _calculate_static_dim(self) -> int:
        dim = self.data_config.n_static_cont
        for i, k in enumerate(self.data_config.n_static_cat):
            if self.data_config.static_cat[i].name not in self.preprocessor.embedding_configs:
                dim += k
        for var_name, embed_dim in self.preprocessor.embedding_configs.items():
            var = next((v for v in self.data_config.static_cat if v.name == var_name), None)
            if var and var.static:
                dim += embed_dim
        return dim

    def _build_model(self):
        self.generator = HierarchicalGenerator(
            data_config=self.data_config, preprocessor=self.preprocessor,
            z_static_dim=self.model_config.z_static_dim,
            z_temporal_dim=self.model_config.z_temporal_dim,
            hidden_dim=self.model_config.generator.hidden_dim,
            gru_layers=self.model_config.generator.gru_layers,
            dropout=self.model_config.generator.dropout,
            n_visits_sharpness=getattr(self.model_config, "n_visits_sharpness", 10.0),
            n_transformer_layers=self.model_config.generator.n_transformer_layers,
            n_heads=self.model_config.generator.n_heads,
            pe_frequencies=self.model_config.generator.pe_frequencies,
        ).to(self.device)

        self.disc_static = StaticDiscriminator(
            input_dim=self.static_dim,
            hidden=self.model_config.static_discriminator.mlp_hidden_dim,
            static_layers=self.model_config.static_discriminator.static_layers,
            dropout=self.model_config.static_discriminator.dropout,
            embed_var_categories=self.embed_var_categories,
        ).to(self.device)

        self.disc_temporal = TemporalDiscriminator(
            static_dim=self.static_dim, temporal_dim=self.temporal_dim,
            model_config=self.model_config,
        ).to(self.device)

        # ── Optimizer a gruppi [v8+v9] ────────────────────────────────
        # noise_model: parametri con lr standard (ottimizzati end-to-end)
        # followup_head, n_visits_head: lr * 0.1 (protetti)
        # time_encoder: lr * 2.0 (amplificato)
        fup_params  = list(self.generator.followup_head.parameters())
        nv_params   = list(self.generator.n_visits_head.parameters())
        te_params   = list(self.generator.time_encoder.parameters())
        nm_params   = list(self.generator.noise_model.parameters())   # [v9]

        fup_ids = {id(p) for p in fup_params}
        nv_ids  = {id(p) for p in nv_params}
        te_ids  = {id(p) for p in te_params}
        nm_ids  = {id(p) for p in nm_params}

        other_params = [p for p in self.generator.parameters()
                        if id(p) not in fup_ids | nv_ids | te_ids | nm_ids]

        if self.preprocessor.embeddings:
            self.preprocessor.embeddings = self.preprocessor.embeddings.to(self.device)
            other_params += list(self.preprocessor.embeddings.parameters())

        lr_g = self.model_config.lr_g
        self.opt_gen = torch.optim.Adam([
            {"params": other_params, "lr": lr_g},
            {"params": fup_params,   "lr": lr_g * 0.1},   # protetto
            {"params": nv_params,    "lr": lr_g * 0.1},   # protetto
            {"params": te_params,    "lr": lr_g * 2.0},   # amplificato
            {"params": nm_params,    "lr": lr_g * 0.5},   # noise model: lr moderato
        ], betas=(0.5, 0.9))

        lr_d_t = getattr(self.model_config, "lr_d_t",
                         self.model_config.lr_d_s * 0.3)
        self.opt_disc_static = torch.optim.Adam(
            self.disc_static.parameters(),   lr=self.model_config.lr_d_s, betas=(0.5, 0.9))
        self.opt_disc_temporal = torch.optim.Adam(
            self.disc_temporal.parameters(), lr=lr_d_t, betas=(0.5, 0.9))

    def set_train(self):
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()

    def _move(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {n: t.to(self.device) for n, t in v.items()}
            else:
                out[k] = v
        return out

    def _extract_real_irr(self, batch):
        if not self.irreversible_idx:
            return None
        return torch.stack([
            batch["temporal_cat"][self.data_config.temporal_cat[idx].name][:, :, 1]
            for idx in self.irreversible_idx
        ], dim=-1)

    def _extract_fake_irr(self, fake_cat_dict):
        return torch.stack([
            fake_cat_dict[self.data_config.temporal_cat[idx].name][:, :, 1]
            for idx in self.irreversible_idx
        ], dim=-1)

    def _build_embed_targets(self, batch):
        targets = {}
        if "static_cat_embed" not in batch or not batch["static_cat_embed"]:
            return targets
        for var_name in self.embed_var_categories:
            if var_name in batch["static_cat_embed"]:
                payload = batch["static_cat_embed"][var_name]
                if payload.dim() == 1:
                    targets[var_name] = payload
        return targets

    def _generate_fake(self, batch_size, use_tf=False, real_irr=None, real_batch=None):
        """
        [v9] Usa generator.sample_noise() per z_temporal strutturato.
        """
        # [v9] Campiona z_static e z_temporal strutturato dal noise_model
        z_s = torch.randn(batch_size, self.model_config.z_static_dim, device=self.device)
        z_t = self.generator.noise_model(batch_size, self.max_len, self.device)

        fixed_visits = None
        if real_batch is not None and "n_visits" in real_batch:
            fixed_visits = real_batch["n_visits"].to(self.device)

        fake_out = self.generator(
            z_s, z_t, temperature=self.current_temperature,
            teacher_forcing=use_tf, real_irr=real_irr,
            hard_mask=False, fixed_visits=fixed_visits,
        )
        fake_disc = prepare_discriminator_inputs(fake_out, self.preprocessor)
        self._last_fake_cat = {k: v.detach() for k, v in fake_out["temporal_cat"].items()}
        return fake_out, fake_disc

    def _train_discriminators(self, real_disc, embed_targets, batch_size,
                               real_cat_dict, update_static=True, update_temporal=True):
        with torch.no_grad():
            _, fake_disc = self._generate_fake(batch_size)

        real_s = real_disc["static"].detach()
        real_t = real_disc["temporal"].detach()
        fake_s = fake_disc["static"].detach()
        fake_t = fake_disc["temporal"].detach()

        # Static discriminator
        d_real_s = self.disc_static(real_s)
        d_fake_s = self.disc_static(fake_s)
        gp_s     = gradient_penalty(lambda x: self.disc_static(x), real_s, fake_s, self.device)
        aux_loss = self.disc_static.auxiliary_loss(real_s, embed_targets)
        loss_d_s = (wgan_discriminator_loss(d_real_s, d_fake_s)
                    + self.lambda_gp_s * gp_s
                    + self.lambda_aux  * aux_loss)
        if update_static:
            self.opt_disc_static.zero_grad()
            loss_d_s.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.disc_static.parameters(), self.model_config.grad_clip)
            self.opt_disc_static.step()

        # Temporal discriminator
        d_real_t = self.disc_temporal(real_s, real_t, real_disc["visit_mask"], real_disc["temporal_mask"])
        d_fake_t = self.disc_temporal(fake_s, fake_t, fake_disc["visit_mask"], fake_disc["temporal_mask"])
        gp_t     = gradient_penalty(
            lambda x: self.disc_temporal(real_s, x, real_disc["visit_mask"], real_disc["temporal_mask"]),
            real_t, fake_t, self.device)

        freq_loss_disc = torch.tensor(0.0, device=self.device)
        if self.lambda_freq_disc > 0 and self.cat_weights:
            freq_loss_disc = categorical_frequency_loss_discriminator(
                real_cat_dict=real_cat_dict, fake_cat_dict=self._last_fake_cat,
                cat_weights=self.cat_weights, visit_mask=real_disc["visit_mask"])

        loss_d_t = (wgan_discriminator_loss(d_real_t, d_fake_t)
                    + self.lambda_gp_t     * gp_t
                    + self.lambda_freq_disc * freq_loss_disc)
        if update_temporal:
            self.opt_disc_temporal.zero_grad()
            loss_d_t.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.disc_temporal.parameters(), self.model_config.grad_clip)
            self.opt_disc_temporal.step()

        return (loss_d_s.item(), loss_d_t.item(), gp_s.item(), gp_t.item(),
                aux_loss.item(), freq_loss_disc.item())

    def _train_generator(self, real_disc, batch_size, real_irr, real_cat_dict, real_batch=None):
        self.opt_gen.zero_grad()
        use_tf = (real_irr is not None and torch.rand(1).item() < self.teacher_forcing_prob)
        fake_out, fake_disc = self._generate_fake(
            batch_size, use_tf=use_tf, real_irr=real_irr, real_batch=real_batch)

        d_fake_s = self.disc_static(fake_disc["static"])
        d_fake_t = self.disc_temporal(
            fake_disc["static"], fake_disc["temporal"],
            fake_disc["visit_mask"], fake_disc["temporal_mask"])
        loss_g = wgan_generator_loss(d_fake_s, d_fake_t)

        irr_loss = torch.tensor(0.0, device=self.device)
        if self.irreversible_idx:
            irr_states = self._extract_fake_irr(fake_out["temporal_cat"])
            irr_loss   = irreversibility_loss(irr_states, fake_disc["visit_mask"])

        freq_loss_gen = torch.tensor(0.0, device=self.device)
        if self.lambda_freq_gen > 0 and self.cat_weights:
            freq_loss_gen = categorical_frequency_loss_generator(
                fake_cat_dict=fake_out["temporal_cat"],
                cat_weights=self.cat_weights,
                visit_mask=fake_disc["visit_mask"])

        fm_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fm > 0 and hasattr(self.disc_static, "get_features"):
            feat_real = self.disc_static.get_features(real_disc["static"]).detach()
            feat_fake = self.disc_static.get_features(fake_disc["static"])
            fm_loss   = feature_matching_loss(feat_real, feat_fake)

        fup_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fup > 0 and real_batch is not None and "followup_norm" in real_batch:
            fup_loss = followup_norm_loss(
                fake_out["followup_norm"], real_batch["followup_norm"].to(self.device))

        nv_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_nv > 0 and real_batch is not None and "n_visits" in real_batch:
            nv_loss = n_visits_supervision_loss(
                fake_out["n_visits_pred"], real_batch["n_visits"].to(self.device))

        # Coverage loss (sanity check: con v8 TimeEncoder sarà ~0)
        coverage_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_coverage > 0 and fake_out.get("visit_times_months") is not None:
            vm_c     = fake_out["visit_mask"].squeeze(-1)
            t_months = fake_out["visit_times_months"]
            B_c      = vm_c.shape[0]
            last_idx = (vm_c * torch.arange(
                vm_c.shape[1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)).argmax(dim=1)
            t_last_m  = t_months[torch.arange(B_c, device=self.device), last_idx]
            gtm       = float(getattr(self.preprocessor, "global_time_max", 400.0) or 400.0)
            d3_fup_m  = (fake_out["followup_norm"].detach() * gtm).clamp(min=1.0)
            has_multi = (vm_c.sum(dim=1) > 1).float()
            rel_err   = ((t_last_m - d3_fup_m) / d3_fup_m) ** 2
            coverage_loss = (rel_err * has_multi).mean()

        # IVI loss
        ivi_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_ivi > 0 and self.real_intervals is not None \
                and fake_out.get("delta_months") is not None:
            ivi_loss = inter_visit_interval_loss(
                visit_times=fake_out["delta_months"],
                visit_mask=fake_out["visit_mask"],
                real_intervals=self.real_intervals)

        # Uniformity loss
        uniformity_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_uniformity > 0 and fake_out.get("delta_months") is not None:
            uniformity_loss = inter_visit_uniformity_loss(
                delta_months=fake_out["delta_months"],
                visit_mask=fake_out["visit_mask"])

        # Static cat loss: Focal CE + KL pesata
        scat_loss      = torch.tensor(0.0, device=self.device)
        fake_scat_soft = fake_out.get("static_cat_soft") or {}
        fake_scat_hard = fake_out.get("static_cat")      or {}
        if self.lambda_static_cat > 0 and self.target_probs_static:
            scat_terms = []
            for var_name, target_p in self.target_probs_static.items():
                src    = fake_scat_soft if var_name in fake_scat_soft else fake_scat_hard
                if var_name not in src:
                    continue
                soft_p        = src[var_name].float()
                fake_marginal = soft_p.mean(dim=0)
                fake_marginal = fake_marginal / fake_marginal.sum().clamp(min=1e-8)
                target_p_dev  = target_p.to(self.device).float()

                inv_freq    = (1.0 / target_p_dev.clamp(min=0.05)).clamp(max=20.0)
                inv_freq    = inv_freq / inv_freq.mean()
                kl          = target_p_dev * (
                    torch.log(target_p_dev.clamp(min=1e-8))
                    - torch.log(fake_marginal.clamp(min=1e-8)))
                kl_weighted = (kl * inv_freq).sum()

                log_soft      = torch.log(soft_p.clamp(min=1e-8))
                p_t           = (target_p_dev.unsqueeze(0) * soft_p).sum(dim=-1).clamp(min=1e-8)
                focal_weight  = (1.0 - p_t) ** 2
                ce_per_sample = -(target_p_dev.unsqueeze(0) * log_soft).sum(dim=-1)
                focal_ce_loss = (focal_weight * ce_per_sample).mean()

                scat_terms.append(0.7 * kl_weighted + 0.3 * focal_ce_loss)
            if scat_terms:
                scat_loss = torch.stack(scat_terms).mean()

        sc_var_loss   = torch.tensor(0.0, device=self.device)
        lambda_sc_var = getattr(self.model_config, "lambda_sc_var", 1.0) or 1.0
        if (lambda_sc_var > 0
                and fake_out.get("static_cont") is not None
                and real_batch is not None
                and "static_cont" in real_batch):
            sc_var_loss = static_cont_dist_loss(
                fake_out["static_cont"].float(),
                real_batch["static_cont"].float().to(self.device))

        total = (loss_g
                 + self.alpha_irr        * irr_loss
                 + self.lambda_freq_gen  * freq_loss_gen
                 + self.lambda_fm        * fm_loss
                 + self.lambda_fup       * fup_loss
                 + self.lambda_nv        * nv_loss
                 + self.lambda_coverage  * coverage_loss
                 + self.lambda_static_cat * scat_loss
                 + lambda_sc_var         * sc_var_loss
                 + self.lambda_ivi       * ivi_loss
                 + self.lambda_uniformity * uniformity_loss)
        total.backward()

        all_gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            all_gen_params += list(self.preprocessor.embeddings.parameters())
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_gen_params, self.model_config.grad_clip)
        self.opt_gen.step()

        mean_nv = float(fake_out["n_visits"].mean().item())
        return (loss_g.item(), irr_loss.item(), freq_loss_gen.item(), mean_nv,
                nv_loss.item(), coverage_loss.item(), fm_loss.item(),
                scat_loss.item(), fup_loss.item(), ivi_loss.item(),
                uniformity_loss.item())

    def _compute_curriculum_p(self, epoch: int, total_epochs: int) -> float:
        """
        Schedule lineare per il curriculum learning della visit_mask.

        Phase 1 [0, warmup_end):         p=0.0  → usa sempre real n_visits
        Phase 2 [warmup_end, fine_start): p cresce linearmente 0→1
        Phase 3 [fine_start, total):      p=1.0  → usa sempre pred n_visits
        """
        warmup_end  = int(total_epochs * self.warmup_mask_frac)
        fine_start  = int(total_epochs * (1.0 - self.finetune_mask_frac))

        if epoch < warmup_end:
            return 0.0
        elif epoch >= fine_start:
            return 1.0
        else:
            window = max(fine_start - warmup_end, 1)
            return float(epoch - warmup_end) / window

    def _update_alpha_irr(self, epoch, total_epochs):
        warmup_end = int(total_epochs * 0.30)
        if epoch < warmup_end:
            t = epoch / max(warmup_end - 1, 1)
            self.alpha_irr = self.alpha_irr_start + t * (self.alpha_irr_max - self.alpha_irr_start)
        else:
            self.alpha_irr = self.alpha_irr_max

    @staticmethod
    def _build_loader(tensors_dict, batch_size, use_dp):
        tensors, keys = [], []
        for k in ["static_cont","static_cat","temporal_cont","visit_mask","visit_time",
                  "followup_norm","n_visits","static_cont_mask","static_cat_mask",
                  "temporal_cont_mask"]:
            if k in tensors_dict and tensors_dict[k] is not None:
                tensors.append(tensors_dict[k]); keys.append(k)
        for name, t in tensors_dict.get("temporal_cat", {}).items():
            tensors.append(t); keys.append(f"tcat::{name}")
        for name, t in tensors_dict.get("temporal_cat_mask", {}).items():
            tensors.append(t); keys.append(f"tcatm::{name}")
        for name, t in tensors_dict.get("static_cat_embed", {}).items():
            tensors.append(t); keys.append(f"sce::{name}")
        for name, t in tensors_dict.get("static_cat_embed_mask", {}).items():
            tensors.append(t); keys.append(f"scem::{name}")
        return (DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                           shuffle=True, drop_last=not use_dp), keys)

    @staticmethod
    def _reconstruct_batch(batch_tuple, keys):
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if   key.startswith("tcat::"):   batch.setdefault("temporal_cat",{})[key[6:]]      = tensor
            elif key.startswith("tcatm::"):  batch.setdefault("temporal_cat_mask",{})[key[7:]] = tensor
            elif key.startswith("sce::"):    batch.setdefault("static_cat_embed",{})[key[5:]]  = tensor
            elif key.startswith("scem::"):   batch.setdefault("static_cat_embed_mask",{})[key[6:]] = tensor
            else: batch[key] = tensor
        return batch

    def fit(self, tensors_dict, epochs=None):
        self.set_train()
        epochs = epochs or self.model_config.epochs
        print("Inizio addestramento DGAN [v9]")
        loader, keys = self._build_loader(
            tensors_dict, self.model_config.batch_size, self.model_config.use_dp)

        if self.model_config.use_dp and OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            self.disc_static, self.opt_disc_static, loader = self.privacy_engine.make_private(
                module=self.disc_static, optimizer=self.opt_disc_static, data_loader=loader,
                noise_multiplier=self.model_config.noise_std,
                max_grad_norm=self.model_config.grad_clip)

        # Categorical frequency weights
        if self.lambda_freq_gen > 0 or self.lambda_freq_disc > 0:
            print("Calcolo pesi per categorical frequency regularization...")
            full_visit_mask = tensors_dict.get("visit_mask")
            if full_visit_mask is not None:
                self.cat_weights = compute_category_weights(
                    real_cat_dict=dict(tensors_dict.get("temporal_cat", {})),
                    visit_mask=full_visit_mask,
                    smoothing=1e-3, power=self.freq_weight_power)
                print(f"  -> Pesi calcolati per {len(self.cat_weights)} variabili.")

        # Static cat target probs
        if self.lambda_static_cat > 0:
            print("Calcolo distribuzione marginale categoriche statiche...")
            scat_tensor = tensors_dict.get("static_cat")
            if scat_tensor is not None and hasattr(self.data_config, "static_cat"):
                offset = 0
                for var in self.data_config.static_cat:
                    if var.name in self.preprocessor.embedding_configs:
                        continue
                    n   = var.n_categories
                    ohe = scat_tensor[:, offset: offset + n].float()
                    freq= ohe.mean(dim=0)
                    self.target_probs_static[var.name] = (
                        freq / freq.sum().clamp(min=1e-8)).to(self.device)
                    offset += n

            if self.target_probs_static:
                print(f"  -> Distribuzione calcolata per {len(self.target_probs_static)} variabili.")
                for var in self.data_config.static_cat:
                    if var.name not in self.target_probs_static:
                        continue
                    head = dict(self.generator.static_cat_heads).get(var.name)
                    if head is None:
                        continue
                    p         = self.target_probs_static[var.name].cpu()
                    log_prior = torch.log(p.clamp(min=1e-6)) - torch.log(p.clamp(min=1e-6)).mean()
                    last_layer = head
                    if isinstance(head, nn.Sequential):
                        for m in reversed(list(head.modules())):
                            if isinstance(m, nn.Linear): last_layer = m; break
                    with torch.no_grad():
                        if hasattr(last_layer,'bias') and last_layer.bias is not None:
                            last_layer.bias.data.copy_(log_prior.to(last_layer.bias.device))

        # Followup warm start
        if "followup_norm" in tensors_dict and tensors_dict["followup_norm"] is not None:
            fn_all  = tensors_dict["followup_norm"].float()
            fn_mean = float(fn_all.mean().clamp(0.02, 0.98))
            fn_logit= float(torch.log(torch.tensor(fn_mean / (1.0 - fn_mean))))
            with torch.no_grad():
                self.generator.followup_head[-2].bias.fill_(fn_logit)
            print(f"  -> followup_head warm start: mean={fn_mean:.3f}  logit={fn_logit:.3f}")

        # n_visits warm start
        if "n_visits" in tensors_dict and tensors_dict["n_visits"] is not None:
            nv_all   = tensors_dict["n_visits"].float()
            nv_median= float(nv_all.median())
            target   = max(nv_median - 1.0, 0.1)
            nv_bias  = float(torch.log(torch.tensor(target).exp() - 1.0 + 1e-6))
            with torch.no_grad():
                self.generator.n_visits_head[-1].bias.fill_(nv_bias)
            print(f"  -> n_visits_head warm start: median={nv_median:.1f}  bias={nv_bias:.3f}")

        # Real intervals per IVI loss
        if (self.lambda_ivi > 0
                and "visit_time" in tensors_dict
                and "visit_mask"  in tensors_dict):
            gtm    = float(getattr(self.preprocessor, "global_time_max", 400.0) or 400.0)
            vt_all = tensors_dict["visit_time"].float()
            vm_all = tensors_dict["visit_mask"].float()
            if vm_all.dim() == 3: vm_all = vm_all.squeeze(-1)

            if "followup_norm" in tensors_dict and tensors_dict["followup_norm"] is not None:
                fup_months = tensors_dict["followup_norm"].float().clamp(0.001, 1.0) * gtm
                dt_real    = (vt_all[:, 1:] - vt_all[:, :-1]) * vm_all[:, 1:] * fup_months.unsqueeze(1)
            else:
                dt_real = (vt_all[:, 1:] - vt_all[:, :-1]) * vm_all[:, 1:] * gtm

            valid_dt = dt_real[dt_real > 0.1]
            if len(valid_dt) > 0:
                self.real_intervals = valid_dt.detach().cpu()
                print(f"  -> real_intervals: n={len(valid_dt)}, "
                      f"mean={float(valid_dt.mean()):.2f} mo  "
                      f"median={float(valid_dt.median()):.2f} mo")

        print(f"\n  Curriculum schedule:")
        warmup_end = int(epochs * self.warmup_mask_frac)
        fine_start = int(epochs * (1.0 - self.finetune_mask_frac))
        print(f"    Epoche   0-{warmup_end}: p=0.0 (real n_visits)")
        print(f"    Epoche {warmup_end}-{fine_start}: p cresce 0→1 (mix)")
        print(f"    Epoche {fine_start}-{epochs}: p=1.0 (pred n_visits)")

        best_disc_loss, patience_counter = float("inf"), 0

        for epoch in range(epochs):
            self._update_alpha_irr(epoch, epochs)

            # [v9] Aggiorna curriculum_p e lo segnala al generator
            curriculum_p = self._compute_curriculum_p(epoch, epochs)
            self._current_curriculum_p = curriculum_p
            self.generator.set_curriculum_p(curriculum_p)

            batch_losses = []

            for batch_tuple in loader:
                batch = self._reconstruct_batch(batch_tuple, keys)
                batch = self._move(batch)
                B     = batch["temporal_cont"].shape[0]

                real_disc           = prepare_discriminator_inputs(batch, self.preprocessor)
                real_irr            = self._extract_real_irr(batch) if self.irreversible_idx else None
                embed_targets       = self._build_embed_targets(batch)
                real_cat_dict_batch = dict(batch.get("temporal_cat", {}))

                lds_list, ldt_list, gps_list, gpt_list, aux_list, fdisc_list = [], [], [], [], [], []
                critic_steps_s = self.model_config.critic_steps
                critic_steps_t = getattr(self.model_config, "critic_steps_temporal", critic_steps_s)
                n_steps        = max(critic_steps_s, critic_steps_t)

                for step_idx in range(n_steps):
                    lds, ldt, gps, gpt, aux, fdisc = self._train_discriminators(
                        real_disc, embed_targets, B,
                        real_cat_dict=real_cat_dict_batch,
                        update_static=(step_idx  < critic_steps_s),
                        update_temporal=(step_idx < critic_steps_t),
                    )
                    lds_list.append(lds); ldt_list.append(ldt)
                    gps_list.append(gps); gpt_list.append(gpt)
                    aux_list.append(aux); fdisc_list.append(fdisc)

                (lg, lirr, lfreq_gen, mean_nv, lnv, lcov, lfm,
                 lscat, lfup, livi, lunif) = self._train_generator(
                    real_disc, B, real_irr,
                    real_cat_dict=real_cat_dict_batch, real_batch=batch)

                batch_losses.append({
                    "generator":       lg,
                    "disc_static":     float(np.mean(lds_list)),
                    "disc_temporal":   float(np.mean(ldt_list)),
                    "irreversibility": lirr,
                    "gp_static":       float(np.mean(gps_list)),
                    "gp_temporal":     float(np.mean(gpt_list)),
                    "aux_embed":       float(np.mean(aux_list)),
                    "freq_gen":        lfreq_gen,
                    "freq_disc":       float(np.mean(fdisc_list)),
                    "mean_n_visits":   mean_nv,
                    "nv_loss":         lnv,
                    "coverage_loss":   lcov,
                    "fm_loss":         lfm,
                    "static_cat_loss": lscat,
                    "fup_loss":        lfup,
                    "ivi_loss":        livi,
                    "uniformity_loss": lunif,
                })

            avg = {k: float(np.mean([b[k] for b in batch_losses]))
                   for k in batch_losses[0]}
            for k, v in avg.items():
                self.loss_history[k].append(v)
            self.loss_history["alpha_irr"].append(self.alpha_irr)
            self.loss_history["curriculum_p"].append(curriculum_p)

            # [v9] Log parametri noise model
            noise_stats = self.generator.noise_model.get_stats()
            self.loss_history["noise_rho"].append(noise_stats["rho"])
            self.loss_history["noise_w_global"].append(noise_stats["w_global"])
            self.loss_history["noise_w_ar"].append(noise_stats["w_ar"])
            self.loss_history["noise_w_episod"].append(noise_stats["w_episod"])

            self.current_temperature = max(
                self.temperature_min, self.current_temperature * 0.995)
            self.teacher_forcing_prob = max(
                0.0, self.teacher_forcing_prob * self.teacher_forcing_decay)

            if self.privacy_engine:
                try:
                    self.loss_history["epsilon"].append(
                        self.privacy_engine.get_epsilon(delta=1e-5))
                except Exception:
                    pass

            # Log ogni epoca con info curriculum + noise_model ogni 20 epoche
            print(
                f"[Epoch {epoch+1}/{epochs}]  G={avg['generator']:.3f}  "
                f"D_s={avg['disc_static']:.3f}  D_t={avg['disc_temporal']:.3f}  "
                f"| IVI={avg['ivi_loss']:.4f}  Cov={avg['coverage_loss']:.4f}  "
                f"Unif={avg['uniformity_loss']:.4f}  Fup={avg['fup_loss']:.4f}  "
                f"| Nv={avg['mean_n_visits']:.1f}  NvL={avg['nv_loss']:.3f}  "
                f"| Scat={avg['static_cat_loss']:.4f}  Fm={avg['fm_loss']:.4f}  "
                f"| Fgen={avg['freq_gen']:.4f}  Irr={avg['irreversibility']:.4f}  "
                f"T={self.current_temperature:.3f}  Cur={curriculum_p:.2f}", flush=True)

            if (epoch + 1) % 20 == 0:
                print(
                    f"  [Noise]  rho={noise_stats['rho']:.3f}  "
                    f"w_global={noise_stats['w_global']:.3f}  "
                    f"w_ar={noise_stats['w_ar']:.3f}  "
                    f"w_episod={noise_stats['w_episod']:.3f}", flush=True)

            disc_loss = avg["disc_static"] + avg["disc_temporal"]
            if disc_loss < best_disc_loss:
                best_disc_loss = disc_loss; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    print(f"Early stopping — epoch {epoch+1}"); break

    @torch.no_grad()
    def generate(self, n_samples, temperature=0.5, return_dataframe=True):
        self.set_eval()
        all_outputs, remaining = [], n_samples
        while remaining > 0:
            bs  = min(self.model_config.batch_size, remaining)
            z_s = torch.randn(bs, self.model_config.z_static_dim, device=self.device)
            z_t = self.generator.noise_model(bs, self.max_len, self.device)
            out = self.generator(
                z_s, z_t, temperature=temperature, teacher_forcing=False,
                fixed_visits=None, hard_mask=True)
            all_outputs.append({
                k: (v.cpu().numpy() if torch.is_tensor(v)
                    else {n: t.cpu().numpy() for n, t in v.items()} if isinstance(v, dict)
                    else v)
                for k, v in out.items() if v is not None
            })
            remaining -= bs

        final = {}
        for k in all_outputs[0]:
            if isinstance(all_outputs[0][k], dict):
                final[k] = {
                    n: np.concatenate([o[k][n] for o in all_outputs], axis=0)[:n_samples]
                    for n in all_outputs[0][k]
                }
            elif isinstance(all_outputs[0][k], np.ndarray):
                final[k] = np.concatenate([o[k] for o in all_outputs], axis=0)[:n_samples]

        if "static_cat_embed" in final and final["static_cat_embed"]:
            decoded = self.preprocessor.decode_embeddings(
                {n: torch.tensor(v, device=self.device)
                 for n, v in final["static_cat_embed"].items()})
            final["static_cat_embed_decoded"] = {
                n: idx.cpu().numpy() for n, idx in decoded.items()}
            del final["static_cat_embed"]

        if return_dataframe:
            synth = {
                "temporal_cont": torch.tensor(final["temporal_cont"]),
                "temporal_cat":  {n: torch.tensor(v) for n, v in final["temporal_cat"].items()},
                "visit_mask":    torch.tensor(final["visit_mask"]),
                "visit_times":   torch.tensor(final["visit_times"]),
            }
            if "followup_norm"           in final:
                synth["followup_norm"]            = torch.tensor(final["followup_norm"])
            if "static_cont"             in final:
                synth["static_cont"]              = torch.tensor(final["static_cont"])
            if "static_cat_embed_decoded" in final:
                synth["static_cat_embed_decoded"] = {
                    n: torch.tensor(v)
                    for n, v in final["static_cat_embed_decoded"].items()}
            if "static_cat" in final and final["static_cat"]:
                static_cat_var_names = [
                    v.name for v in self.data_config.static_cat
                    if v.name not in self.preprocessor.embedding_configs]
                arrays = [final["static_cat"][k]
                          for k in static_cat_var_names if k in final["static_cat"]]
                if arrays:
                    synth["static_cat"] = torch.from_numpy(
                        np.concatenate(arrays, axis=1)).float()
            self.set_train()
            return self.preprocessor.inverse_transform(synth, complete_followup=False)

        self.set_train()
        return final

    def save(self, filepath):
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
        }
        if self.preprocessor.embeddings:
            state["embedding_state"] = {
                name: layer.state_dict()
                for name, layer in self.preprocessor.embeddings.items()}
        torch.save(state, filepath)
        logger.info(f"Model saved → {filepath}")

    @classmethod
    def load(cls, filepath, data_config, model_config, preprocessor, device=None):
        state = torch.load(filepath, map_location=device or "cpu")
        dgan  = cls(data_config, model_config, preprocessor, device=device)
        dgan.generator.load_state_dict(state["generator_state"])
        dgan.disc_static.load_state_dict(state["disc_static_state"])
        dgan.disc_temporal.load_state_dict(state["disc_temporal_state"])
        dgan.opt_gen.load_state_dict(state["opt_gen_state"])
        dgan.opt_disc_static.load_state_dict(state["opt_disc_static_state"])
        dgan.opt_disc_temporal.load_state_dict(state["opt_disc_temporal_state"])
        dgan.loss_history                     = state["loss_history"]
        dgan.current_temperature              = state["current_temperature"]
        dgan.preprocessor.embedding_configs   = state["embedding_configs"]
        dgan.preprocessor.scalers_cont        = state["scalers_cont"]
        dgan.preprocessor.inverse_maps        = state["inverse_maps"]
        dgan.preprocessor.global_time_max     = state["global_time_max"]
        if "embedding_state" in state:
            for name, emb_state in state["embedding_state"].items():
                if name in dgan.preprocessor.embeddings:
                    dgan.preprocessor.embeddings[name].load_state_dict(emb_state)
        logger.info(f"Model loaded ← {filepath}")
        return dgan