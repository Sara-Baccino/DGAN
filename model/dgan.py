"""
================================================================================
DGAN — DoppelGANger: training + generation
================================================================================
Fix rispetto alla versione precedente:
  - Embedding weights (preprocessor.embeddings) aggiunti all'optimizer del generatore
  - Opacus: drop_last=False nel DataLoader base (Opacus lo gestisce internamente)
  - wgan_generator_loss chiamata con entrambi i discriminatori
  - GP: usa gradient_penalty dalla losses.py corretta (gestisce 2D e 3D)
  - irreversibility_loss: ora effettiva (non più stub)
================================================================================
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Tuple
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
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available — DP disabled.")

logger = logging.getLogger(__name__)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


class DGAN:
    """
    DoppelGANger con WGAN-GP.

    Modifiche rispetto alla versione originale:
      1. TimeEncoder separato nel generatore (no dipendenza circolare)
      2. t_norm + delta_t come feature GRU
      3. cummax hard constraint per irreversibili
      4. Auxiliary loss per embedding (ottimizza il mapping insieme)
      5. alpha_irr curriculum annealing
      6. critic_steps corretto: fake generati con no_grad
      7. Early stopping su loss discriminatore (piu stabile)
      8. betas Adam WGAN-GP raccomandati (0.5, 0.9)
      9. teacher_forcing_decay piu conservativo (0.999)
    """

    def __init__(self, data_config, model_config, preprocessor, device: Optional[str] = None):
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
        )
        self.max_len          = data_config.max_len
        self.irreversible_idx = data_config.irreversible_idx

        # Mappa {var_name: n_cats} per le auxiliary heads del discriminatore statico
        # Costruita dal preprocessor: include solo le variabili con embedding
        self.embed_var_categories: Dict[str, int] = {}
        for var_name in preprocessor.embedding_configs:
            var = next((v for v in data_config.static_cat if v.name == var_name), None)
            if var is not None:
                self.embed_var_categories[var_name] = len(var.mapping)

        self._build_model()

        self.privacy_engine        = None
        self.current_temperature   = model_config.gumbel_temperature_start
        self.teacher_forcing_prob  = 1.0
        self.teacher_forcing_decay = 0.999

        # Curriculum alpha_irr
        self.alpha_irr_start = 0.1
        self.alpha_irr_max   = model_config.alpha_irr
        self.alpha_irr       = self.alpha_irr_start

        # Peso auxiliary loss: bilancia con WGAN loss
        # 0.1 e un buon punto di partenza; aumenta se gli embedding collassano
        self.lambda_aux = getattr(model_config, "lambda_aux", 0.1)
        self.lambda_gp  = model_config.lambda_gp

        self.loss_history = {
            "generator": [], "disc_static": [], "disc_temporal": [],
            "irreversibility": [], "gp_static": [], "gp_temporal": [],
            "aux_embed": [], "alpha_irr": [], "epsilon": [],
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
    def _build_model(self):
        self.generator = HierarchicalGenerator(
            data_config=self.data_config,
            preprocessor=self.preprocessor,
            z_static_dim=self.model_config.z_static_dim,
            z_temporal_dim=self.model_config.z_temporal_dim,
            hidden_dim=self.model_config.generator.hidden_dim,
            gru_layers=self.model_config.generator.gru_layers,
            dropout=self.model_config.generator.dropout,
        ).to(self.device)

        self.disc_static = StaticDiscriminator(
            input_dim=self.static_dim,
            hidden=self.model_config.static_discriminator.mlp_hidden_dim,
            static_layers=self.model_config.static_discriminator.static_layers,
            dropout=self.model_config.static_discriminator.dropout,
            embed_var_categories=self.embed_var_categories,
        ).to(self.device)

        self.disc_temporal = TemporalDiscriminator(
            static_dim=self.static_dim,
            temporal_dim=self.temporal_dim,
            mlp_hidden_dim=self.model_config.temporal_discriminator.mlp_hidden_dim,
            gru_hidden_dim=self.model_config.temporal_discriminator.gru_hidden_dim,
            gru_layers=self.model_config.temporal_discriminator.gru_layers,
            mlp_layers=self.model_config.temporal_discriminator.mlp_layers,
            dropout=self.model_config.temporal_discriminator.dropout,
        ).to(self.device)

        gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            self.preprocessor.embeddings = self.preprocessor.embeddings.to(self.device)
            gen_params += list(self.preprocessor.embeddings.parameters())

        # betas (0.5, 0.9): raccomandati per WGAN-GP
        self.opt_gen = torch.optim.Adam(
            gen_params, lr=self.model_config.lr_g, betas=(0.5, 0.9)
        )
        self.opt_disc_static = torch.optim.Adam(
            self.disc_static.parameters(), lr=self.model_config.lr_d_s, betas=(0.5, 0.9)
        )
        self.opt_disc_temporal = torch.optim.Adam(
            self.disc_temporal.parameters(), lr=self.model_config.lr_d_t, betas=(0.5, 0.9)
        )

    def set_train(self):
        self.generator.train(); self.disc_static.train(); self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval(); self.disc_static.eval(); self.disc_temporal.eval()

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
        """
        Estrae gli indici categorici interi per le variabili embedded,
        da usare come target nella auxiliary loss del discriminatore.

        Per il batch reale, "static_cat_embed" contiene indici [B] (long).
        """
        targets = {}
        if "static_cat_embed" not in batch or not batch["static_cat_embed"]:
            return targets
        for var_name in self.embed_var_categories:
            if var_name in batch["static_cat_embed"]:
                payload = batch["static_cat_embed"][var_name]
                # Batch reale: indici [B] (long)
                if payload.dim() == 1:
                    targets[var_name] = payload
                # Non disponibile dal batch fake (vettori continui): skip
        return targets

    # ------------------------------------------------------------------
    def _generate_fake(
        self,
        batch_size: int,
        use_tf:     bool = False,
        real_irr:   Optional[torch.Tensor] = None,
    ) -> Tuple[Dict, Dict]:
        """Genera fake e restituisce (fake_out, fake_disc_input)."""
        z_s = torch.randn(batch_size, self.model_config.z_static_dim, device=self.device)
        z_t = torch.randn(batch_size, self.max_len, self.model_config.z_temporal_dim,
                          device=self.device)
        fake_out  = self.generator(
            z_s, z_t, temperature=self.current_temperature,
            teacher_forcing=use_tf, real_irr=real_irr,
        )
        fake_disc = prepare_discriminator_inputs(fake_out, self.preprocessor)
        return fake_out, fake_disc

    # ------------------------------------------------------------------
    def _train_discriminators(
        self,
        real_disc:     Dict,
        embed_targets: Dict[str, torch.Tensor],
        batch_size:    int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Un passo di aggiornamento dei discriminatori.
        I fake vengono generati con no_grad: il generatore non accumula gradienti.
        """
        with torch.no_grad():
            _, fake_disc = self._generate_fake(batch_size)

        real_s = real_disc["static"].detach()
        real_t = real_disc["temporal"].detach()
        fake_s = fake_disc["static"].detach()
        fake_t = fake_disc["temporal"].detach()

        # ---- Static discriminator ----
        self.opt_disc_static.zero_grad()
        d_real_s = self.disc_static(real_s)
        d_fake_s = self.disc_static(fake_s)
        gp_s     = gradient_penalty(lambda x: self.disc_static(x), real_s, fake_s, self.device)

        # Auxiliary loss: il discriminatore deve predire la categoria embedded
        # dai dati reali. Questo segnale backpropaga anche sugli embedding
        # del preprocessor, che sono nello stesso ottimizzatore del generatore...
        # ATTENZIONE: qui vogliamo solo addestrare il discriminatore, quindi
        # usiamo real_s.detach() gia fatto sopra.
        aux_loss  = self.disc_static.auxiliary_loss(real_s, embed_targets)
        loss_d_s  = (wgan_discriminator_loss(d_real_s, d_fake_s)
                     + self.lambda_gp * gp_s
                     + self.lambda_aux * aux_loss)
        loss_d_s.backward()
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.disc_static.parameters(), self.model_config.grad_clip)
        self.opt_disc_static.step()

        # ---- Temporal discriminator ----
        self.opt_disc_temporal.zero_grad()
        d_real_t = self.disc_temporal(real_s, real_t, real_disc["visit_mask"], real_disc["temporal_mask"])
        d_fake_t = self.disc_temporal(fake_s, fake_t, fake_disc["visit_mask"], fake_disc["temporal_mask"])
        gp_t     = gradient_penalty(
            lambda x: self.disc_temporal(real_s, x, real_disc["visit_mask"], real_disc["temporal_mask"]),
            real_t, fake_t, self.device,
        )
        loss_d_t = wgan_discriminator_loss(d_real_t, d_fake_t) + self.lambda_gp * gp_t
        loss_d_t.backward()
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.disc_temporal.parameters(), self.model_config.grad_clip)
        self.opt_disc_temporal.step()

        return loss_d_s.item(), loss_d_t.item(), gp_s.item(), gp_t.item(), aux_loss.item()

    # ------------------------------------------------------------------
    def _train_generator(
        self,
        real_disc:  Dict,
        batch_size: int,
        real_irr:   Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        """
        Un passo di aggiornamento del generatore.
        La auxiliary loss del generatore e' calcolata separatamente:
        vogliamo che il generatore produca embedding riconoscibili dal
        discriminatore come categorie reali (segnale inverso rispetto
        all'auxiliary del discriminatore).
        """
        self.opt_gen.zero_grad()

        use_tf   = (real_irr is not None and torch.rand(1).item() < self.teacher_forcing_prob)
        fake_out, fake_disc = self._generate_fake(batch_size, use_tf=use_tf, real_irr=real_irr)

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

        total = loss_g + self.alpha_irr * irr_loss
        total.backward()

        all_gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            all_gen_params += list(self.preprocessor.embeddings.parameters())
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_gen_params, self.model_config.grad_clip)
        self.opt_gen.step()
        return loss_g.item(), irr_loss.item()

    # ------------------------------------------------------------------
    def _update_alpha_irr(self, epoch: int, total_epochs: int):
        """Warm-up lineare: 0.1 -> alpha_irr_max nel primo 30% delle epoche."""
        warmup_end = int(total_epochs * 0.30)
        if epoch < warmup_end:
            t = epoch / max(warmup_end - 1, 1)
            self.alpha_irr = self.alpha_irr_start + t * (self.alpha_irr_max - self.alpha_irr_start)
        else:
            self.alpha_irr = self.alpha_irr_max

    # ------------------------------------------------------------------
    @staticmethod
    def _build_loader(tensors_dict: Dict, batch_size: int, use_dp: bool):
        tensors, keys = [], []
        for k in ["static_cont", "static_cat", "temporal_cont", "visit_mask", "visit_time",
                  "static_cont_mask", "static_cat_mask", "temporal_cont_mask"]:
            if k in tensors_dict and tensors_dict[k] is not None:
                tensors.append(tensors_dict[k]); keys.append(k)
        for name, t in tensors_dict.get("temporal_cat",      {}).items():
            tensors.append(t); keys.append(f"tcat::{name}")
        for name, t in tensors_dict.get("temporal_cat_mask", {}).items():
            tensors.append(t); keys.append(f"tcatm::{name}")
        for name, t in tensors_dict.get("static_cat_embed",  {}).items():
            tensors.append(t); keys.append(f"sce::{name}")
        for name, t in tensors_dict.get("static_cat_embed_mask", {}).items():
            tensors.append(t); keys.append(f"scem::{name}")
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                          shuffle=True, drop_last=not use_dp), keys

    @staticmethod
    def _reconstruct_batch(batch_tuple, keys):
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if   key.startswith("tcat::"):  batch.setdefault("temporal_cat",          {})[key[6:]]  = tensor
            elif key.startswith("tcatm::"): batch.setdefault("temporal_cat_mask",     {})[key[7:]]  = tensor
            elif key.startswith("sce::"):   batch.setdefault("static_cat_embed",      {})[key[5:]]  = tensor
            elif key.startswith("scem::"): batch.setdefault("static_cat_embed_mask",  {})[key[6:]]  = tensor
            else: batch[key] = tensor
        return batch

    # ------------------------------------------------------------------
    def fit(self, tensors_dict: Dict, epochs: Optional[int] = None):
        self.set_train()
        epochs = epochs or self.model_config.epochs
        print("Inizio addestramento DGAN")

        loader, keys = self._build_loader(tensors_dict, self.model_config.batch_size,
                                          self.model_config.use_dp)

        if self.model_config.use_dp and OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            self.disc_static, self.opt_disc_static, loader = \
                self.privacy_engine.make_private(
                    module=self.disc_static, optimizer=self.opt_disc_static,
                    data_loader=loader, noise_multiplier=self.model_config.noise_std,
                    max_grad_norm=self.model_config.grad_clip,
                )

        best_disc_loss   = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            self._update_alpha_irr(epoch, epochs)
            batch_losses = []

            for batch_tuple in loader:
                batch     = self._reconstruct_batch(batch_tuple, keys)
                batch     = self._move(batch)
                B         = batch["temporal_cont"].shape[0]
                real_disc = prepare_discriminator_inputs(batch, self.preprocessor)
                real_irr  = self._extract_real_irr(batch) if self.irreversible_idx else None

                # Targets per auxiliary loss (indici categorici embedded)
                embed_targets = self._build_embed_targets(batch)

                # critic_steps passi discriminatore (generatore frozen via no_grad)
                lds_list, ldt_list, gps_list, gpt_list, aux_list = [], [], [], [], []
                for _ in range(self.model_config.critic_steps):
                    lds, ldt, gps, gpt, aux = self._train_discriminators(
                        real_disc, embed_targets, B
                    )
                    lds_list.append(lds); ldt_list.append(ldt)
                    gps_list.append(gps); gpt_list.append(gpt)
                    aux_list.append(aux)

                # un passo generatore
                lg, lirr = self._train_generator(real_disc, B, real_irr)

                batch_losses.append({
                    "generator":       lg,
                    "disc_static":     float(np.mean(lds_list)),
                    "disc_temporal":   float(np.mean(ldt_list)),
                    "irreversibility": lirr,
                    "gp_static":       float(np.mean(gps_list)),
                    "gp_temporal":     float(np.mean(gpt_list)),
                    "aux_embed":       float(np.mean(aux_list)),
                })

            avg = {k: float(np.mean([b[k] for b in batch_losses])) for k in batch_losses[0]}
            for k, v in avg.items():
                self.loss_history[k].append(v)
            self.loss_history["alpha_irr"].append(self.alpha_irr)

            self.current_temperature  = max(0.1, self.current_temperature * 0.995)
            self.teacher_forcing_prob = max(0.0, self.teacher_forcing_prob * self.teacher_forcing_decay)

            if self.privacy_engine:
                try:
                    self.loss_history["epsilon"].append(
                        self.privacy_engine.get_epsilon(delta=1e-5)
                    )
                except Exception:
                    pass

            print(
                f"[Epoch {epoch+1}/{epochs}]  "
                f"G={avg['generator']:.4f}  "
                f"D_s={avg['disc_static']:.4f}  D_t={avg['disc_temporal']:.4f}  "
                f"Irr={avg['irreversibility']:.5f}  "
                f"Aux={avg['aux_embed']:.4f}  "
                f"alpha_irr={self.alpha_irr:.3f}  "
                f"T={self.current_temperature:.3f}"
            )

            # Early stopping su loss discriminatore combinata
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
    def generate(self, n_samples: int, temperature: float = 0.5, return_dataframe: bool = True):
        self.set_eval()
        all_outputs, remaining = [], n_samples
        while remaining > 0:
            bs  = min(self.model_config.batch_size, remaining)
            z_s = torch.randn(bs, self.model_config.z_static_dim, device=self.device)
            z_t = torch.randn(bs, self.max_len, self.model_config.z_temporal_dim, device=self.device)
            out = self.generator(z_s, z_t, temperature=temperature, teacher_forcing=False,
                                 fixed_visits=self.model_config.fixed_visits)
            all_outputs.append({
                k: v.cpu().numpy() if torch.is_tensor(v)
                   else {n: t.cpu().numpy() for n, t in v.items()}
                for k, v in out.items()
            })
            remaining -= bs

        final = {}
        for k in all_outputs[0]:
            if isinstance(all_outputs[0][k], dict):
                final[k] = {n: np.concatenate([o[k][n] for o in all_outputs], axis=0)[:n_samples]
                            for n in all_outputs[0][k]}
            else:
                final[k] = np.concatenate([o[k] for o in all_outputs], axis=0)[:n_samples]

        if "static_cat_embed" in final and final["static_cat_embed"]:
            decoded = self.preprocessor.decode_embeddings({
                n: torch.tensor(v, device=self.device)
                for n, v in final["static_cat_embed"].items()
            })
            final["static_cat_embed_decoded"] = {n: idx.cpu().numpy() for n, idx in decoded.items()}
            del final["static_cat_embed"]

        if return_dataframe:
            synth = {
                "temporal_cont": torch.tensor(final["temporal_cont"]),
                "temporal_cat":  {n: torch.tensor(v) for n, v in final["temporal_cat"].items()},
                "visit_mask":    torch.tensor(final["visit_mask"]),
                "visit_times":   torch.tensor(final["visit_times"]),
            }
            if "static_cont" in final:
                synth["static_cont"] = torch.tensor(final["static_cont"])
            if "static_cat_embed_decoded" in final:
                synth["static_cat_embed_decoded"] = {
                    n: torch.tensor(v) for n, v in final["static_cat_embed_decoded"].items()
                }
            if "static_cat" in final:
                static_cat = np.concatenate(
                    [final["static_cat"][k] for k in sorted(final["static_cat"].keys())], axis=1
                )
                synth["static_cat"] = torch.from_numpy(static_cat).float()

            self.set_train()
            return self.preprocessor.inverse_transform(synth, complete_followup=True)

        self.set_train()
        return final
    
    # ------------------------------------------------------------------
    # save / load
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
            # preprocessor state
            "embedding_configs":       self.preprocessor.embedding_configs,
            "scalers_cont":            self.preprocessor.scalers_cont,
            "inverse_maps":            self.preprocessor.inverse_maps,
            "global_time_max":         self.preprocessor.global_time_max,
        }
        # pesi embedding
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