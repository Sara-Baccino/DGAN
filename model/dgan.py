"""
model/dgan.py
================================================================================
Modifiche rispetto alla versione precedente:

  [v4] temporal_dim += 1 per followup_norm.
       prepare_discriminator_inputs ora aggiunge followup_norm come feature
       temporale. Il TemporalDiscriminator deve ricevere temporal_dim corretto.

  [v4] visit_length_loss RIMOSSA.
       Il generatore predice n_visits direttamente con n_visits_head.
       Non serve più la loss di supervisione sulla lunghezza.
       Log: "Len=" rimosso, aggiunto "Nv=" (media n_visits generati).

  [v4] followup_norm passato come chiave nel batch al discriminatore.
       Nel _build_loader: "followup_norm" aggiunto alle chiavi.
       Nel _reconstruct_batch: gestito con la chiave "fn".
       In _move: tensore spostato su device.

  [v4] _generate_fake: followup_norm dal generatore passato nel batch fake
       per prepare_discriminator_inputs.

  [v2] lambda_gp_s e lambda_gp_t separati.
  [v2] temperature_min = 0.5.
  [v2] lr_d_t separato da lr_d_s.
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
    followup_constraint_loss,
    static_cat_marginal_loss,      # [v6.1] loss diretta su categoriche statiche
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available — DP disabled.")

logger = logging.getLogger(__name__)


class DGAN:

    def __init__(self, data_config, model_config, preprocessor, device: Optional[str] = None):
        self.data_config  = data_config
        self.model_config = model_config
        self.preprocessor = preprocessor
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.static_dim   = self._calculate_static_dim()

        # [v4] +1 per followup_norm nel temporal
        self.temporal_dim = (
            data_config.n_temp_cont
            + sum(
                2 if v.irreversible else len(v.mapping)
                for v in preprocessor.vars
                if not v.static and v.kind == "categorical"
            )
            + 1   # followup_norm broadcastato come feature temporale
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
        self.teacher_forcing_prob  = 1.0
        self.teacher_forcing_decay = 0.995

        # Curriculum alpha_irr
        self.alpha_irr_start = 0.1
        self.alpha_irr_max   = model_config.alpha_irr
        self.alpha_irr       = self.alpha_irr_start

        # Loss weights
        self.lambda_aux = getattr(model_config, "lambda_aux", 0.2)
        self.lambda_gp_s = getattr(model_config, "lambda_gp_s",
                                   getattr(model_config, "lambda_gp", 4.0))
        self.lambda_gp_t = getattr(model_config, "lambda_gp_t",
                                   getattr(model_config, "lambda_gp", 4.0) * 1.5)

        # Categorical frequency regularization
        self.lambda_freq_gen   = getattr(model_config, "lambda_freq_gen",   0.15)
        self.lambda_freq_disc  = getattr(model_config, "lambda_freq_disc",  0.05)
        self.freq_weight_power = getattr(model_config, "freq_weight_power", 1.0)
        self.cat_weights: Dict[str, torch.Tensor] = {}

        # [v5] Feature matching loss (discriminatore statico)
        # Dà gradiente denso al generatore sulle categoriche statiche.
        # Disabilitato di default (0): aggiungi "lambda_fm": 2.0 nel JSON.
        self.lambda_fm = getattr(model_config, "lambda_fm", 0.0)

        # [v5] Followup norm loss: vincola la durata del follow-up sintetico
        # a replicare la distribuzione empirica del batch reale.
        self.lambda_fup = getattr(model_config, "lambda_fup", 0.5)

        # [v5.1] n_visits supervision: supervisione diretta su n_visits_head.
        # In training, n_visits reale viene campionato dalla distribuzione empirica
        # e passato come fixed_visits al generatore. L_nv = SmoothL1(n_v_pred, n_v_real).
        # Risolve il collasso di n_visits sulla media.
        self.lambda_nv = getattr(model_config, "lambda_nv", 0.5)

        # [v5.1] Followup constraint: MSE followup_norm + penalità t_norm_last < 1.
        # Forza il TimeEncoder a distribuire le visite sull'intero range [0,1]
        # invece di concentrarle all'inizio.
        self.lambda_fc = getattr(model_config, "lambda_fc", 0.3)

        # [v6.1] Static categorical marginal loss
        # Supervisione diretta sulla distribuzione marginale delle categoriche statiche.
        # Risolve il collapse V=1.0 su SEX, ETHNICC, ALCOHOL, SMOKING, ecc.
        # target_probs_static viene calcolato in fit() una volta dal training set.
        self.lambda_static_cat:   float = getattr(model_config, "lambda_static_cat", 3.0)
        self.target_probs_static: Dict[str, torch.Tensor] = {}

        # Temperature
        self.temperature_min = getattr(model_config, "temperature_min", 0.5)

        self.loss_history = {
            "generator":       [], "disc_static":  [], "disc_temporal": [],
            "irreversibility": [], "gp_static":    [], "gp_temporal":   [],
            "aux_embed":       [], "alpha_irr":    [], "epsilon":        [],
            "freq_gen":        [], "freq_disc":    [],
            "mean_n_visits":   [],
            "nv_loss":         [],
            "fc_loss":         [],
            "fm_loss":         [],
            "fup_loss":        [],   # [v6.2] followup_head supervision
            "static_cat_loss": [],
        }

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    @staticmethod
    def _init_cat_logit_biases(generator, cat_weights: Dict[str, torch.Tensor]):
        """
        Inizializza i bias degli output head categorici con log(p_reale).
        Porta i logit iniziali a campionare dalla distribuzione marginale
        reale invece che da uniform, evitando il collapse verso la classe
        maggioritaria.
        """
        for name, head in generator.static_cat_heads.items():
            if name not in cat_weights:
                continue
            w = cat_weights[name]
            log_prior = -torch.log(w + 1e-8)
            log_prior = log_prior - log_prior.mean()
            with torch.no_grad():
                if hasattr(head, 'bias') and head.bias is not None:
                    head.bias.data.copy_(log_prior.to(head.bias.device))

        for name, head in generator.temporal_cat_heads.items():
            if name not in cat_weights or generator.temporal_cat_irrev.get(name, False):
                continue
            w = cat_weights[name]
            log_prior = -torch.log(w + 1e-8)
            log_prior = log_prior - log_prior.mean()
            with torch.no_grad():
                if hasattr(head, 'bias') and head.bias is not None:
                    head.bias.data.copy_(log_prior.to(head.bias.device))

    # ------------------------------------------------------------------
    def _build_model(self):
        n_visits_sharpness = getattr(self.model_config, "n_visits_sharpness", 10.0)

        self.generator = HierarchicalGenerator(
            data_config=self.data_config,
            preprocessor=self.preprocessor,
            z_static_dim=self.model_config.z_static_dim,
            z_temporal_dim=self.model_config.z_temporal_dim,
            hidden_dim=self.model_config.generator.hidden_dim,
            gru_layers=self.model_config.generator.gru_layers,
            dropout=self.model_config.generator.dropout,
            n_visits_sharpness=n_visits_sharpness,
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
            static_dim   = self.static_dim,
            temporal_dim = self.temporal_dim,
            model_config = self.model_config,
        ).to(self.device)

        gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            self.preprocessor.embeddings = self.preprocessor.embeddings.to(self.device)
            gen_params += list(self.preprocessor.embeddings.parameters())

        lr_d_t = getattr(self.model_config, "lr_d_t", self.model_config.lr_d_s / 3.0)

        self.opt_gen = torch.optim.Adam(
            gen_params, lr=self.model_config.lr_g, betas=(0.5, 0.9)
        )
        self.opt_disc_static = torch.optim.Adam(
            self.disc_static.parameters(),
            lr=self.model_config.lr_d_s, betas=(0.5, 0.9),
        )
        self.opt_disc_temporal = torch.optim.Adam(
            self.disc_temporal.parameters(),
            lr=lr_d_t, betas=(0.5, 0.9),
        )

    def set_train(self):
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()

    # ------------------------------------------------------------------
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

    def _extract_fake_irr(self, fake_cat_dict: Dict) -> torch.Tensor:
        return torch.stack([
            fake_cat_dict[self.data_config.temporal_cat[idx].name][:, :, 1]
            for idx in self.irreversible_idx
        ], dim=-1)

    # ------------------------------------------------------------------
    def _build_embed_targets(self, batch: Dict) -> Dict[str, torch.Tensor]:
        targets = {}
        if "static_cat_embed" not in batch or not batch["static_cat_embed"]:
            return targets
        for var_name in self.embed_var_categories:
            if var_name in batch["static_cat_embed"]:
                payload = batch["static_cat_embed"][var_name]
                if payload.dim() == 1:
                    targets[var_name] = payload
        return targets

    # ------------------------------------------------------------------
    def _generate_fake(
        self,
        batch_size:  int,
        use_tf:      bool = False,
        real_irr:    Optional[torch.Tensor] = None,
        real_batch:  Optional[Dict]         = None,
    ) -> Tuple[Dict, Dict]:
        """
        [v5] real_batch: se fornito, campiona n_visits e followup_norm reali
        per condizionare il generatore sulla distribuzione empirica.

        MECCANISMO:
          Il DataLoader usa shuffle=True → real_batch e z_s/z_t non hanno
          corrispondenza paziente-per-paziente, ma la distribuzione marginale
          di n_visits e followup_norm del batch sintetico = distribuzione reale.
          Questo risolve il collasso di n_visits senza modificare il WGAN.
        """
        z_s = torch.randn(batch_size, self.model_config.z_static_dim, device=self.device)
        z_t = torch.randn(
            batch_size, self.max_len, self.model_config.z_temporal_dim,
            device=self.device,
        )

        # [v6.2] real_followup_norm NON viene più passato al forward():
        # followup_head è ora sempre attivo in training (non bypassato).
        # real_followup_norm rimane disponibile in real_batch per fup_loss.
        fixed_visits = None
        if real_batch is not None and "n_visits" in real_batch:
            fixed_visits = real_batch["n_visits"].to(self.device)

        fake_out = self.generator(
            z_s, z_t,
            temperature=self.current_temperature,
            teacher_forcing=use_tf,
            real_irr=real_irr,
            hard_mask=False,
            fixed_visits=fixed_visits,
            real_followup_norm=None,   # [v6.2] sempre None — followup_head libero
        )

        fake_batch = dict(fake_out)
        fake_batch["followup_norm"] = fake_out["followup_norm"]
        fake_disc = prepare_discriminator_inputs(fake_batch, self.preprocessor)

        self._last_fake_cat = {
            k: v.detach() for k, v in fake_out["temporal_cat"].items()
        }
        return fake_out, fake_disc

    # ------------------------------------------------------------------
    def _train_discriminators(
        self,
        real_disc:      Dict,
        embed_targets:  Dict[str, torch.Tensor],
        batch_size:     int,
        real_cat_dict:  Dict[str, torch.Tensor],
        update_static:  bool = True,
        update_temporal: bool = True,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        [v5.1] update_static / update_temporal: flag per aggiornamenti asimmetrici.
        Permette critic_steps_temporal < critic_steps per ridurre la dominanza di D_t.
        Se update_X=False, il forward pass del discriminatore X viene eseguito
        (per calcolare la loss da loggare) ma il backward/step viene saltato.
        """
        with torch.no_grad():
            _, fake_disc = self._generate_fake(batch_size)

        real_s = real_disc["static"].detach()
        real_t = real_disc["temporal"].detach()
        fake_s = fake_disc["static"].detach()
        fake_t = fake_disc["temporal"].detach()

        # ---- Static ----
        d_real_s = self.disc_static(real_s)
        d_fake_s = self.disc_static(fake_s)
        gp_s     = gradient_penalty(
            lambda x: self.disc_static(x), real_s, fake_s, self.device
        )
        aux_loss = self.disc_static.auxiliary_loss(real_s, embed_targets)
        loss_d_s = (
            wgan_discriminator_loss(d_real_s, d_fake_s)
            + self.lambda_gp_s * gp_s
            + self.lambda_aux  * aux_loss
        )
        if update_static:
            self.opt_disc_static.zero_grad()
            loss_d_s.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.disc_static.parameters(), self.model_config.grad_clip)
            self.opt_disc_static.step()

        # ---- Temporal ----
        d_real_t = self.disc_temporal(
            real_s, real_t, real_disc["visit_mask"], real_disc["temporal_mask"]
        )
        d_fake_t = self.disc_temporal(
            fake_s, fake_t, fake_disc["visit_mask"], fake_disc["temporal_mask"]
        )
        gp_t = gradient_penalty(
            lambda x: self.disc_temporal(
                real_s, x, real_disc["visit_mask"], real_disc["temporal_mask"]
            ),
            real_t, fake_t, self.device,
        )

        freq_loss_disc = torch.tensor(0.0, device=self.device)
        if self.lambda_freq_disc > 0 and self.cat_weights:
            freq_loss_disc = categorical_frequency_loss_discriminator(
                real_cat_dict=real_cat_dict,
                fake_cat_dict=self._last_fake_cat,
                cat_weights=self.cat_weights,
                visit_mask=real_disc["visit_mask"],
            )

        loss_d_t = (
            wgan_discriminator_loss(d_real_t, d_fake_t)
            + self.lambda_gp_t      * gp_t
            + self.lambda_freq_disc * freq_loss_disc
        )
        if update_temporal:
            self.opt_disc_temporal.zero_grad()
            loss_d_t.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.disc_temporal.parameters(), self.model_config.grad_clip)
            self.opt_disc_temporal.step()

        return (
            loss_d_s.item(), loss_d_t.item(),
            gp_s.item(), gp_t.item(),
            aux_loss.item(), freq_loss_disc.item(),
        )

    # ------------------------------------------------------------------
    def _train_generator(
        self,
        real_disc:     Dict,
        batch_size:    int,
        real_irr:      Optional[torch.Tensor],
        real_cat_dict: Dict[str, torch.Tensor],
        real_batch:    Optional[Dict] = None,
    ) -> Tuple[float, float, float, float]:
        """
        [v5] Aggiunge:
          - real_batch: passato a _generate_fake per conditioning su
            n_visits e followup_norm reali (risolve collasso n_visits)
          - feature_matching_loss: gradiente denso sulle categoriche statiche
          - followup_norm_loss: vincola la durata del follow-up sintetico

        Returns: (loss_g, irr_loss, freq_loss_gen, mean_n_visits)
        """
        self.opt_gen.zero_grad()

        use_tf      = (real_irr is not None
                       and torch.rand(1).item() < self.teacher_forcing_prob)

        # [v5] Passa real_batch per conditioning su distribuzione reale
        fake_out, fake_disc = self._generate_fake(
            batch_size, use_tf=use_tf, real_irr=real_irr, real_batch=real_batch
        )

        d_fake_s = self.disc_static(fake_disc["static"])
        d_fake_t = self.disc_temporal(
            fake_disc["static"], fake_disc["temporal"],
            fake_disc["visit_mask"], fake_disc["temporal_mask"],
        )
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
                visit_mask=fake_disc["visit_mask"],
            )

        # [v5] Feature matching loss: gradiente denso dal discriminatore statico
        # sulle distribuzioni categoriche (SEX, ETHNICC, ALCOHOL, SMOKING...)
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fm > 0 and hasattr(self.disc_static, "get_features"):
            feat_real = self.disc_static.get_features(real_disc["static"]).detach()
            feat_fake = self.disc_static.get_features(fake_disc["static"])
            fm_loss   = feature_matching_loss(feat_real, feat_fake)

        # [v6.2] Followup norm loss: MSE tra followup_head(z_static) e followup reale.
        # followup_head è ora SEMPRE attivo in training (non bypassato dal valore reale),
        # quindi questa loss dà gradiente diretto a followup_head in ogni step.
        fup_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fup > 0 and real_batch is not None and "followup_norm" in real_batch:
            pred_fup = fake_out["followup_norm"]                         # [B] da followup_head
            real_fup = real_batch["followup_norm"].to(self.device)       # [B] reale (shuffled)
            fup_loss = followup_norm_loss(pred_fup, real_fup)

        # [v5.1] n_visits supervision: forza n_visits_head a predire il valore reale.
        nv_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_nv > 0 and real_batch is not None and "n_visits" in real_batch:
            nv_loss = n_visits_supervision_loss(
                fake_out["n_visits_pred"],
                real_batch["n_visits"].to(self.device),
            )

        # [v6.2] Followup constraint: rimuoviamo il termine L_fn (già coperto da fup_loss)
        # e manteniamo solo L_sl (t_norm_last ≈ 1.0) per forzare la distribuzione
        # temporale sull'intero range — non più MSE ridondante.
        fc_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fc > 0 and fake_out.get("visit_times") is not None:
            vm      = fake_out["visit_mask"].squeeze(-1)        # [B, T]
            B_fc    = vm.shape[0]
            last_idx = (vm * torch.arange(
                vm.shape[1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)).argmax(dim=1)
            t_last  = fake_out["visit_times"][
                torch.arange(B_fc, device=self.device), last_idx
            ]
            has_multi = (vm.sum(dim=1) > 1).float()
            fc_loss   = ((t_last - 1.0) ** 2 * has_multi).mean()

        # [v6.1/v6.2] Static categorical marginal loss
        # Supervisione diretta su SEX, ETHNICC, ALCOHOL, SMOKING, INRPT ecc.
        # Corregge il collapse Cramér's V=1.0 sulle variabili statiche binarie.
        # Nota: fake_out["static_cat"] è None se tutte le var sono in embedding_configs.
        scat_loss = torch.tensor(0.0, device=self.device)
        fake_scat = fake_out.get("static_cat")
        if (self.lambda_static_cat > 0
                and self.target_probs_static
                and fake_scat is not None
                and len(fake_scat) > 0):
            scat_loss = static_cat_marginal_loss(
                fake_static_cat=fake_scat,
                target_probs=self.target_probs_static,
            )

        total = (
            loss_g
            + self.alpha_irr          * irr_loss
            + self.lambda_freq_gen    * freq_loss_gen
            + self.lambda_fm          * fm_loss
            + self.lambda_fup         * fup_loss
            + self.lambda_nv          * nv_loss
            + self.lambda_fc          * fc_loss
            + self.lambda_static_cat  * scat_loss   # [v6.1]
        )
        total.backward()

        all_gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            all_gen_params += list(self.preprocessor.embeddings.parameters())
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_gen_params, self.model_config.grad_clip)
        self.opt_gen.step()

        with torch.no_grad():
            mean_nv = float(fake_out["n_visits"].mean().item())

        return loss_g.item(), irr_loss.item(), freq_loss_gen.item(), mean_nv, nv_loss.item(), fc_loss.item(), fm_loss.item(), scat_loss.item(), fup_loss.item()

    # ------------------------------------------------------------------
    def _update_alpha_irr(self, epoch: int, total_epochs: int):
        warmup_end = int(total_epochs * 0.30)
        if epoch < warmup_end:
            t = epoch / max(warmup_end - 1, 1)
            self.alpha_irr = (
                self.alpha_irr_start
                + t * (self.alpha_irr_max - self.alpha_irr_start)
            )
        else:
            self.alpha_irr = self.alpha_irr_max

    # ------------------------------------------------------------------
    @staticmethod
    def _build_loader(tensors_dict: Dict, batch_size: int, use_dp: bool):
        tensors, keys = [], []
        for k in [
            "static_cont", "static_cat", "temporal_cont", "visit_mask",
            "visit_time", "followup_norm",   # [v4]
            "n_visits",                      # [v5] per conditioning su distribuzione reale
            "static_cont_mask", "static_cat_mask", "temporal_cont_mask",
        ]:
            if k in tensors_dict and tensors_dict[k] is not None:
                tensors.append(tensors_dict[k])
                keys.append(k)

        for name, t in tensors_dict.get("temporal_cat",          {}).items():
            tensors.append(t); keys.append(f"tcat::{name}")
        for name, t in tensors_dict.get("temporal_cat_mask",     {}).items():
            tensors.append(t); keys.append(f"tcatm::{name}")
        for name, t in tensors_dict.get("static_cat_embed",      {}).items():
            tensors.append(t); keys.append(f"sce::{name}")
        for name, t in tensors_dict.get("static_cat_embed_mask", {}).items():
            tensors.append(t); keys.append(f"scem::{name}")

        return (
            DataLoader(
                TensorDataset(*tensors),
                batch_size=batch_size,
                shuffle=True,
                drop_last=not use_dp,
            ),
            keys,
        )

    @staticmethod
    def _reconstruct_batch(batch_tuple, keys):
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if   key.startswith("tcat::"):  batch.setdefault("temporal_cat",         {})[key[6:]]  = tensor
            elif key.startswith("tcatm::"): batch.setdefault("temporal_cat_mask",    {})[key[7:]]  = tensor
            elif key.startswith("sce::"):   batch.setdefault("static_cat_embed",     {})[key[5:]]  = tensor
            elif key.startswith("scem::"): batch.setdefault("static_cat_embed_mask", {})[key[6:]]  = tensor
            else: batch[key] = tensor
        return batch

    # ------------------------------------------------------------------
    def fit(self, tensors_dict: Dict, epochs: Optional[int] = None):
        self.set_train()
        epochs = epochs or self.model_config.epochs
        print("Inizio addestramento DGAN")

        loader, keys = self._build_loader(
            tensors_dict, self.model_config.batch_size, self.model_config.use_dp
        )

        if self.model_config.use_dp and OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            self.disc_static, self.opt_disc_static, loader = \
                self.privacy_engine.make_private(
                    module=self.disc_static,
                    optimizer=self.opt_disc_static,
                    data_loader=loader,
                    noise_multiplier=self.model_config.noise_std,
                    max_grad_norm=self.model_config.grad_clip,
                )

        # Categorical frequency weights (una volta prima del loop)
        if self.lambda_freq_gen > 0 or self.lambda_freq_disc > 0:
            print("Calcolo pesi per categorical frequency regularization...")
            combined_cat     = dict(tensors_dict.get("temporal_cat", {}))
            full_visit_mask  = tensors_dict.get("visit_mask")
            if full_visit_mask is not None:
                self.cat_weights = compute_category_weights(
                    real_cat_dict=combined_cat,
                    visit_mask=full_visit_mask,
                    smoothing=1e-3,
                    power=self.freq_weight_power,
                )
                self._init_cat_logit_biases(self.generator, self.cat_weights)
                print("  -> Bias logit inizializzati con log-prior reale.")
                print(f"  -> Pesi calcolati per {len(self.cat_weights)} variabili.")
            else:
                print("  [WARN] visit_mask non trovato.")

        # [v6.1] Distribuzione marginale reale delle categoriche statiche
        # Usata da static_cat_marginal_loss per supervisionare SEX, ETHNICC, ecc.
        if self.lambda_static_cat > 0:
            print("Calcolo distribuzione marginale categoriche statiche...")
            scat_tensor = tensors_dict.get("static_cat")   # [N, sum_n_cats] one-hot flat
            if scat_tensor is not None and hasattr(self.data_config, "static_cat"):
                offset = 0
                for var in self.data_config.static_cat:
                    if var.name in self.preprocessor.embedding_configs:
                        continue   # CENTRE gestita da aux_loss, non qui
                    n    = var.n_categories
                    ohe  = scat_tensor[:, offset: offset + n].float()
                    freq = ohe.mean(dim=0)
                    self.target_probs_static[var.name] = (
                        freq / freq.sum().clamp(min=1e-8)
                    ).to(self.device)
                    offset += n
            if self.target_probs_static:
                print(f"  -> Distribuzione calcolata per {len(self.target_probs_static)} variabili statiche.")
            else:
                print("  [WARN] static_cat non trovato — static_cat_marginal_loss disabilitata.")

        # [v6.2] Warm start followup_head con la media empirica di followup_norm
        # PROBLEMA: followup_head[-2].bias=-0.62 (valore fisso) non corrisponde
        # alla distribuzione reale dei dati. Se il follow-up medio reale è 0.15
        # (pazienti con follow-up breve), il bias deve essere logit(0.15)=-1.73.
        # Senza questo, followup_head parte lontano dalla distribuzione reale
        # e FcL rimane alto per decine di epoche.
        if "followup_norm" in tensors_dict and tensors_dict["followup_norm"] is not None:
            fn_all   = tensors_dict["followup_norm"].float()
            fn_mean  = float(fn_all.mean().clamp(0.02, 0.98))
            fn_logit = float(torch.log(torch.tensor(fn_mean / (1.0 - fn_mean))))
            with torch.no_grad():
                self.generator.followup_head[-2].bias.fill_(fn_logit)
            print(f"  -> followup_head warm start: mean={fn_mean:.3f}  logit={fn_logit:.3f}")

        best_disc_loss   = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self._update_alpha_irr(epoch, epochs)
            batch_losses = []

            for batch_tuple in loader:
                batch     = self._reconstruct_batch(batch_tuple, keys)
                batch     = self._move(batch)
                B         = batch["temporal_cont"].shape[0]

                # [v4] Costruisce il batch reale con followup_norm
                # per prepare_discriminator_inputs
                real_disc = prepare_discriminator_inputs(batch, self.preprocessor)
                real_irr  = self._extract_real_irr(batch) if self.irreversible_idx else None
                embed_targets     = self._build_embed_targets(batch)
                real_cat_dict_batch = dict(batch.get("temporal_cat", {}))

                lds_list, ldt_list = [], []
                gps_list, gpt_list = [], []
                aux_list, fdisc_list = [], []

                # [v5.1] critic_steps separati per D_s e D_t.
                # D_t era troppo dominante (D_t≈-20 vs D_s≈-0.3) perché
                # entrambi i discriminatori ricevevano n_critic aggiornamenti.
                # critic_steps_temporal < critic_steps rallenta D_t rispetto a D_s.
                # Configurabile via "critic_steps_temporal" nel JSON (default=critic_steps).
                critic_steps_s = self.model_config.critic_steps
                critic_steps_t = getattr(
                    self.model_config, "critic_steps_temporal", critic_steps_s
                )
                n_steps = max(critic_steps_s, critic_steps_t)

                for step_idx in range(n_steps):
                    update_s = step_idx < critic_steps_s
                    update_t = step_idx < critic_steps_t
                    lds, ldt, gps, gpt, aux, fdisc = self._train_discriminators(
                        real_disc, embed_targets, B,
                        real_cat_dict=real_cat_dict_batch,
                        update_static=update_s,
                        update_temporal=update_t,
                    )
                    lds_list.append(lds);    ldt_list.append(ldt)
                    gps_list.append(gps);    gpt_list.append(gpt)
                    aux_list.append(aux);    fdisc_list.append(fdisc)
                    lds_list.append(lds);    ldt_list.append(ldt)
                    gps_list.append(gps);    gpt_list.append(gpt)
                    aux_list.append(aux);    fdisc_list.append(fdisc)

                lg, lirr, lfreq_gen, mean_nv, lnv, lfc, lfm, lscat, lfup = self._train_generator(
                    real_disc, B, real_irr,
                    real_cat_dict=real_cat_dict_batch,
                    real_batch=batch,
                )

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
                    "fc_loss":         lfc,
                    "fm_loss":         lfm,
                    "static_cat_loss": lscat,
                    "fup_loss":        lfup,   # [v6.2]
                })

            avg = {
                k: float(np.mean([b[k] for b in batch_losses]))
                for k in batch_losses[0]
            }
            for k, v in avg.items():
                self.loss_history[k].append(v)
            self.loss_history["alpha_irr"].append(self.alpha_irr)

            self.current_temperature = max(
                self.temperature_min, self.current_temperature * 0.995
            )
            self.teacher_forcing_prob = max(
                0.0, self.teacher_forcing_prob * self.teacher_forcing_decay
            )

            if self.privacy_engine:
                try:
                    self.loss_history["epsilon"].append(
                        self.privacy_engine.get_epsilon(delta=1e-5)
                    )
                except Exception:
                    pass

            print(
                f"[Epoch {epoch+1}/{epochs}]  "
                f"G={avg['generator']:.3f}  "
                f"D_s={avg['disc_static']:.3f}  D_t={avg['disc_temporal']:.3f}  "
                f"| Fup={avg['fup_loss']:.4f}  Fc={avg['fc_loss']:.4f}  "
                f"| Nv={avg['mean_n_visits']:.1f}  NvL={avg['nv_loss']:.3f}  "
                f"| Scat={avg['static_cat_loss']:.4f}  Fm={avg['fm_loss']:.4f}  "
                f"| Fgen={avg['freq_gen']:.4f}  Irr={avg['irreversibility']:.4f}  "
                f"| Aux={avg['aux_embed']:.3f}  "
                f"T={self.current_temperature:.3f}",
                flush=True,
            )

            disc_loss = avg["disc_static"] + avg["disc_temporal"]
            if disc_loss < best_disc_loss:
                best_disc_loss   = disc_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    print(f"Early stopping — epoch {epoch+1}")
                    break

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        n_samples:        int,
        temperature:      float = 0.5,
        return_dataframe: bool  = True,
    ):
        self.set_eval()
        all_outputs, remaining = [], n_samples

        while remaining > 0:
            bs  = min(self.model_config.batch_size, remaining)
            z_s = torch.randn(bs, self.model_config.z_static_dim, device=self.device)
            z_t = torch.randn(
                bs, self.max_len, self.model_config.z_temporal_dim, device=self.device
            )
            out = self.generator(
                z_s, z_t,
                temperature=temperature,
                teacher_forcing=False,
                fixed_visits=self.model_config.fixed_visits,
                hard_mask=True,   # maschera esattamente 0/1 in inference
            )
            all_outputs.append({
                k: v.cpu().numpy() if torch.is_tensor(v)
                   else {n: t.cpu().numpy() for n, t in v.items()}
                   if isinstance(v, dict) else v
                for k, v in out.items()
                if v is not None
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
                final[k] = np.concatenate(
                    [o[k] for o in all_outputs], axis=0
                )[:n_samples]

        if "static_cat_embed" in final and final["static_cat_embed"]:
            decoded = self.preprocessor.decode_embeddings({
                n: torch.tensor(v, device=self.device)
                for n, v in final["static_cat_embed"].items()
            })
            final["static_cat_embed_decoded"] = {
                n: idx.cpu().numpy() for n, idx in decoded.items()
            }
            del final["static_cat_embed"]

        if return_dataframe:
            synth = {
                "temporal_cont": torch.tensor(final["temporal_cont"]),
                "temporal_cat":  {n: torch.tensor(v) for n, v in final["temporal_cat"].items()},
                "visit_mask":    torch.tensor(final["visit_mask"]),
                "visit_times":   torch.tensor(final["visit_times"]),
            }
            if "followup_norm" in final:
                synth["followup_norm"] = torch.tensor(final["followup_norm"])
            if "static_cont" in final:
                synth["static_cont"] = torch.tensor(final["static_cont"])
            if "static_cat_embed_decoded" in final:
                synth["static_cat_embed_decoded"] = {
                    n: torch.tensor(v)
                    for n, v in final["static_cat_embed_decoded"].items()
                }
            # [FIX BUG 5] Ordine da self.vars, non sorted()
            if "static_cat" in final and final["static_cat"]:
                static_cat_var_names = [
                    v.name
                    for v in self.data_config.static_cat
                    if v.name not in self.preprocessor.embedding_configs
                ]
                arrays = [
                    final["static_cat"][k]
                    for k in static_cat_var_names
                    if k in final["static_cat"]
                ]
                if arrays:
                    synth["static_cat"] = torch.from_numpy(
                        np.concatenate(arrays, axis=1)
                    ).float()

            self.set_train()
            return self.preprocessor.inverse_transform(synth, complete_followup=False)

        self.set_train()
        return final

    # ------------------------------------------------------------------
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
        }
        if self.preprocessor.embeddings:
            state["embedding_state"] = {
                name: layer.state_dict()
                for name, layer in self.preprocessor.embeddings.items()
            }
        torch.save(state, filepath)
        logger.info(f"Model saved → {filepath}")

    @classmethod
    def load(cls, filepath: str, data_config, model_config, preprocessor, device=None):
        state = torch.load(filepath, map_location=device or "cpu")
        dgan  = cls(data_config, model_config, preprocessor, device=device)
        dgan.generator.load_state_dict(state["generator_state"])
        dgan.disc_static.load_state_dict(state["disc_static_state"])
        dgan.disc_temporal.load_state_dict(state["disc_temporal_state"])
        dgan.opt_gen.load_state_dict(state["opt_gen_state"])
        dgan.opt_disc_static.load_state_dict(state["opt_disc_static_state"])
        dgan.opt_disc_temporal.load_state_dict(state["opt_disc_temporal_state"])
        dgan.loss_history        = state["loss_history"]
        dgan.current_temperature = state["current_temperature"]
        dgan.preprocessor.embedding_configs = state["embedding_configs"]
        dgan.preprocessor.scalers_cont      = state["scalers_cont"]
        dgan.preprocessor.inverse_maps      = state["inverse_maps"]
        dgan.preprocessor.global_time_max   = state["global_time_max"]
        if "embedding_state" in state:
            for name, emb_state in state["embedding_state"].items():
                if name in dgan.preprocessor.embeddings:
                    dgan.preprocessor.embeddings[name].load_state_dict(emb_state)
        logger.info(f"Model loaded ← {filepath}")
        return dgan