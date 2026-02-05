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
    StaticDiscriminator,
    TemporalDiscriminator,
    prepare_discriminator_inputs,
)
from utils.losses import (
    wgan_discriminator_loss,
    wgan_generator_loss,
    gradient_penalty,
    irreversibility_loss,
)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    logging.warning("Opacus not available — DP disabled.")

logger = logging.getLogger(__name__)


# ==================================================================
class DGAN:
    """
    DoppelGANger con WGAN-GP, maschere missing, teacher forcing
    irreversibili, embedding ad alta cardinalità, DP opzionale.
    """

    def __init__(self, data_config, model_config, preprocessor, device: Optional[str] = None):
        self.data_config  = data_config
        self.model_config = model_config
        self.preprocessor = preprocessor
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.static_dim   = self._calculate_static_dim()
        #self.temporal_dim = self._calculate_temporal_dim()
        #self.temporal_dim = self._infer_temporal_dim()
        self.temporal_dim = (self.data_config.n_temp_cont
                        + sum(2 if v.irreversible else len(v.mapping)
                            for v in self.preprocessor.vars
                            if not v.static and v.kind == "categorical")
                        )

        self.max_len      = data_config.max_len
        self.irreversible_idx = data_config.irreversible_idx

        self._build_model()

        self.privacy_engine      = None
        self.current_temperature = model_config.gumbel_temperature_start
        self._debug_printed = True

        self.loss_history = {
            "generator": [], "disc_static": [], "disc_temporal": [],
            "irreversibility": [], "gp_static": [], "gp_temporal": [], "epsilon": [],
        }

    # ------------------------------------------------------------------
    # dimensioni
    # ------------------------------------------------------------------
    def _infer_temporal_dim(self):
        dim = self.data_config.n_temp_cont
        for v in self.preprocessor.vars:
            if not v.static and v.kind == "categorical":
                dim += len(v.mapping)  # SEMPRE
        return dim

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

    def _calculate_temporal_dim(self) -> int:
        #return self.data_config.n_temp_cont + sum(self.data_config.n_temp_cat)
        dim = self.data_config.n_temp_cont
        for v in self.preprocessor.vars:
            if not v.static and v.kind == "categorical":
                dim += v.n_categories
        return dim

    # ------------------------------------------------------------------
    # costruzione modelli + optimizer
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
            cond_dim=0,
        ).to(self.device)

        self.disc_static = StaticDiscriminator(
            input_dim=self.static_dim,
            hidden=self.model_config.static_discriminator.mlp_hidden_dim,
            static_layers=self.model_config.static_discriminator.static_layers,
            dropout=self.model_config.static_discriminator.dropout,
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

        # ---- optimizer generatore ----
        # IMPORTANTE: includi anche i pesi degli embedding layer del preprocessor.
        # Senza di essi i pesi dell'Embedding(48,6) non vengono mai aggiornati
        # e il nearest-neighbor in decode_embeddings usa pesi casuali.
        gen_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            # Sposta gli embedding sul device corretto
            self.preprocessor.embeddings = self.preprocessor.embeddings.to(self.device)
            gen_params += list(self.preprocessor.embeddings.parameters())

        self.opt_gen           = torch.optim.Adam(gen_params,                            lr=self.model_config.lr)
        self.opt_disc_static   = torch.optim.Adam(self.disc_static.parameters(),         lr=self.model_config.lr)
        self.opt_disc_temporal = torch.optim.Adam(self.disc_temporal.parameters(),       lr=self.model_config.lr)

    def set_train(self):
        self.generator.train()
        self.disc_static.train()
        self.disc_temporal.train()

    def set_eval(self):
        self.generator.eval()
        self.disc_static.eval()
        self.disc_temporal.eval()

    # ------------------------------------------------------------------
    # batch utils
    # ------------------------------------------------------------------
    def _move(self, batch: Dict) -> Dict:
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
                logger.info(f"{k}: {tuple(v.shape)}")
            elif isinstance(v, dict):
                out[k] = {n: t.to(self.device) for n, t in v.items()}
                for n, t in v.items():
                    logger.info(f"{k}[{n}]: {tuple(t.shape)}")
            else:
                out[k] = v
        return out

    def _extract_real_irr(self, batch: Dict) -> Optional[torch.Tensor]:
        """Estrae [B, T, n_irr] dal batch reale per teacher forcing."""
        if not self.irreversible_idx:
            return None
        slices = []
        for idx in self.irreversible_idx:
            name = self.data_config.temporal_cat[idx].name
            slices.append(batch["temporal_cat"][name][:, :, 1])   # stato "attivo"
        return torch.stack(slices, dim=-1)

    def _extract_fake_irr(self, fake_cat_dict: Dict) -> torch.Tensor:
        """Estrae [B, T, n_irr] dall'output del generatore."""
        slices = []
        for idx in self.irreversible_idx:
            name = self.data_config.temporal_cat[idx].name
            slices.append(fake_cat_dict[name][:, :, 1])
        return torch.stack(slices, dim=-1)

    # ------------------------------------------------------------------
    # train discriminatori
    # ------------------------------------------------------------------
    def _train_discriminators(self, batch: Dict, batch_size: int) -> Tuple[float, float, float, float]:
        
        # input reali concatenati (maschere applicate internamente)
        real = prepare_discriminator_inputs(batch, self.preprocessor)
        

        # genera fake
        z_s = torch.randn(batch_size, self.model_config.z_static_dim,                 device=self.device)
        z_t = torch.randn(batch_size, self.max_len, self.model_config.z_temporal_dim, device=self.device)
        with torch.no_grad():
            fake_out = self.generator(z_s, z_t, temperature=self.current_temperature)

        # input fake concatenati (nessuna maschera → prepare crea maschere di 1)
        #for k, v in fake_out.get("static_cat_embed", {}).items():
            #print(k, type(v), v.shape if torch.is_tensor(v) else v.keys())

        fake = prepare_discriminator_inputs(fake_out, self.preprocessor)
        assert fake["static"].numel() > 0, "Fake static input is empty!"
        assert real["temporal"].shape == fake["temporal"].shape
        assert real["visit_mask"].shape == fake["visit_mask"].shape
        assert real["temporal_mask"].shape == fake["temporal_mask"].shape
        sce = fake_out.get("static_cat_embed")
        if sce:
            for k, v in sce.items():
                assert torch.is_tensor(v), f"static_cat_embed[{k}] is not Tensor"
                assert v.dim() == 2, f"static_cat_embed[{k}] must be [B, D]"



        #print("REAL temporal:", real["temporal"].shape)
        #print("FAKE temporal:", fake["temporal"].shape)
        #print("GRU expects:", self.disc_temporal.gru.input_size)


        # ========== STATIC ==========
        self.opt_disc_static.zero_grad()

        d_real_s = self.disc_static(real["static"].detach())
        d_fake_s = self.disc_static(fake["static"].detach())

        gp_s = gradient_penalty(
            lambda x: self.disc_static(x),
            real["static"].detach(),
            fake["static"].detach(),
            self.device,
        )

        loss_d_s = wgan_discriminator_loss(d_real_s, d_fake_s) + 10.0 * gp_s
        loss_d_s.backward()
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.disc_static.parameters(), self.model_config.grad_clip)
        self.opt_disc_static.step()

        # ========== TEMPORAL ==========
        self.opt_disc_temporal.zero_grad()

        d_real_t = self.disc_temporal(
            real["static"].detach(), real["temporal"], real["visit_mask"], real["temporal_mask"]
        )
        d_fake_t = self.disc_temporal(
            fake["static"].detach(), fake["temporal"].detach(),
            fake["visit_mask"], fake["temporal_mask"]
        )

        # GP temporale: interpolazione solo sul tensore temporale [B,T,D],
        # static tenuto fisso (real, detached) — standard practice WGAN-GP
        gp_t = gradient_penalty(
            lambda x: self.disc_temporal(
                real["static"].detach(), x, real["visit_mask"], real["temporal_mask"]
            ),
            real["temporal"].detach(),
            fake["temporal"].detach(),
            self.device,
        )

        loss_d_t = wgan_discriminator_loss(d_real_t, d_fake_t) + 10.0 * gp_t
        loss_d_t.backward()
        # -------- WGAN loss --------
        #loss_d_t = wgan_discriminator_loss(d_real_t, d_fake_t)
        #loss_d_t.backward()
        # -------- GP loss (separata) --------
        #(10.0 * gp_t).backward()

        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.disc_temporal.parameters(), self.model_config.grad_clip)
        
        self.opt_disc_temporal.step()

        return loss_d_s.item(), loss_d_t.item(), gp_s.item(), gp_t.item()

    # ------------------------------------------------------------------
    # train generatore
    def _train_generator(self, batch: Dict, batch_size: int) -> Tuple[float, float]:

        self.opt_gen.zero_grad()

        z_s = torch.randn(batch_size, self.model_config.z_static_dim,                 device=self.device)
        z_t = torch.randn(batch_size, self.max_len, self.model_config.z_temporal_dim, device=self.device)

        # teacher forcing: forza stati irreversibili reali
        real_irr = self._extract_real_irr(batch) if self.irreversible_idx else None

        fake_out = self.generator(
            z_s, z_t,
            temperature=self.current_temperature,
            teacher_forcing=(real_irr is not None),
            real_irr=real_irr,
        )

        fake = prepare_discriminator_inputs(fake_out, self.preprocessor)

        d_fake_s = self.disc_static(fake["static"])
        d_fake_t = self.disc_temporal(
            fake["static"], fake["temporal"], fake["visit_mask"], fake["temporal_mask"]
        )

        # loss WGAN generatore: entrambi i discriminatori
        loss_g = wgan_generator_loss(d_fake_s, d_fake_t)

        # loss irreversibilità: penalizza transizioni 1→0
        irr_loss = torch.tensor(0.0, device=self.device)
        if self.irreversible_idx:
            irr_states = self._extract_fake_irr(fake_out["temporal_cat"])  # [B, T, n_irr]
            irr_loss   = irreversibility_loss(irr_states, fake["visit_mask"])

        total = loss_g + irr_loss
        total.backward()

        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.generator.parameters()) +
                (list(self.preprocessor.embeddings.parameters()) if self.preprocessor.embeddings else []),
                self.model_config.grad_clip,
            )
        self.opt_gen.step()

        return loss_g.item(), irr_loss.item()

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_loader(tensors_dict: Dict, batch_size: int, use_dp: bool) -> Tuple[DataLoader, list]:
        """
        Converte il dict di tensori in DataLoader + lista chiavi.

        Separatore '::' nelle chiavi → nessun rischio di collisione con nomi variabili.
        drop_last=False quando si usa DP (Opacus lo gestisce internamente e
        stampa un warning se è True).
        """
        tensors = []
        keys    = []

        for k in ["static_cont", "static_cat", "temporal_cont",
                  "visit_mask", "visit_time",
                  "static_cont_mask", "static_cat_mask", "temporal_cont_mask"]:
            if k in tensors_dict and tensors_dict[k] is not None:
                tensors.append(tensors_dict[k])
                keys.append(k)

        for name, t in tensors_dict.get("temporal_cat", {}).items():
            tensors.append(t);  keys.append(f"tcat::{name}")

        for name, t in tensors_dict.get("temporal_cat_mask", {}).items():
            tensors.append(t);  keys.append(f"tcatm::{name}")

        for name, t in tensors_dict.get("static_cat_embed", {}).items():
            tensors.append(t);  keys.append(f"sce::{name}")

        for name, t in tensors_dict.get("static_cat_embed_mask", {}).items():
            tensors.append(t);  keys.append(f"scem::{name}")

        dataset = TensorDataset(*tensors)
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=not use_dp,   # False con DP: Opacus non sopporta drop_last=True
        )
        return loader, keys

    @staticmethod
    def _reconstruct_batch(batch_tuple: tuple, keys: list) -> Dict:
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if   key.startswith("tcat::"):
                batch.setdefault("temporal_cat",      {})[key[len("tcat::"):]]  = tensor
            elif key.startswith("tcatm::"):
                batch.setdefault("temporal_cat_mask", {})[key[len("tcatm::"):]] = tensor
            elif key.startswith("sce::"):
                batch.setdefault("static_cat_embed",  {})[key[len("sce::"):]]   = tensor
            elif key.startswith("scem::"):
                batch.setdefault("static_cat_embed_mask", {})[key[len("scem::"):]] = tensor
            else:
                batch[key] = tensor
        return batch

    # ------------------------------------------------------------------
    # fit (training loop principale)
    # ------------------------------------------------------------------
    def fit(self, tensors_dict: Dict, epochs: Optional[int] = None):
        self.set_train()
        epochs = epochs or self.model_config.epochs

        print("Inizio addestramento:")

        loader, keys = self._build_loader(
            tensors_dict, self.model_config.batch_size, self.model_config.use_dp
        )

        # ---- Differential Privacy ----
        if self.model_config.use_dp:
            
            if OPACUS_AVAILABLE:
                logger.info("Enabling DP via Opacus")
                self.privacy_engine = PrivacyEngine()
                # Opacus wrappa solo disc_static (come nel paper DoppelGANger)
                self.disc_static, self.opt_disc_static, loader = self.privacy_engine.make_private(
                    module=self.disc_static,
                    optimizer=self.opt_disc_static,
                    data_loader=loader,
                    noise_multiplier=self.model_config.noise_std,
                    max_grad_norm=self.model_config.grad_clip,
                )
            
            else:
                logger.warning("DP richiesto ma Opacus non disponibile — training senza DP.")

        # ---- loop ----
        best_loss        = float("inf")
        patience_counter = 0

        logger.info("=" * 80)
        logger.info("INIZIO TRAINING DGAN")
        logger.info(f"Device: {self.device}")
        logger.info(f"Static dim: {self.static_dim}")
        logger.info(f"Temporal dim: {self.temporal_dim}")
        logger.info(f"Max len: {self.max_len}")
        logger.info(f"Use DP: {self.model_config.use_dp}")
        logger.info("=" * 80)

        

        for epoch in range(epochs):
            batch_losses = []

            for batch_tuple in loader:
                batch = self._reconstruct_batch(batch_tuple, keys)
                batch = self._move(batch)
                B     = batch["temporal_cont"].shape[0]

                # discriminatori
                for _ in range(self.model_config.critic_steps):
                    lds, ldt, gps, gpt = self._train_discriminators(batch, B)

                # generatore
                lg, lirr = self._train_generator(batch, B)

                batch_losses.append({
                    "generator": lg, "disc_static": lds, "disc_temporal": ldt,
                    "irreversibility": lirr, "gp_static": gps, "gp_temporal": gpt,
                })

            # ---- epoch summary ----
            avg = {k: np.mean([b[k] for b in batch_losses]) for k in batch_losses[0]}
            for k, v in avg.items():
                self.loss_history[k].append(v)

            self.current_temperature = max(0.1, self.current_temperature * 0.95)

            if self.privacy_engine:
                try:
                    eps = self.privacy_engine.get_epsilon(delta=1e-5)
                    self.loss_history["epsilon"].append(eps)
                    logger.info(f"[Epoch {epoch+1}/{epochs}] ε = {eps:.2f}")
                except Exception:
                    logger.warning("Could not compute epsilon")
            
            print(f"[Epoch {epoch+1}/{epochs}] "
                f"G={avg['generator']:.4f}  D_s={avg['disc_static']:.4f}  "
                f"D_t={avg['disc_temporal']:.4f}  Irr={avg['irreversibility']:.4f}  "
                f"Temp={self.current_temperature:.3f}")
            
            logger.info(
                f"[Epoch {epoch+1}/{epochs}] "
                f"G={avg['generator']:.4f}  D_s={avg['disc_static']:.4f}  "
                f"D_t={avg['disc_temporal']:.4f}  Irr={avg['irreversibility']:.4f}  "
                f"Temp={self.current_temperature:.3f}"
            )

            # early stopping
            if avg["generator"] < best_loss:
                best_loss        = avg["generator"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.model_config.patience:
                    print(f"Early stopping — epoch {epoch+1}")
                    break

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        n_samples:        int,
        temperature:      float = 0.5,
        return_dataframe: bool  = True,
    ):
        """
        Genera dati sintetici.

        Returns:
            pd.DataFrame long (con Delta_t, statiche ripetute, nessun missing)
            se return_dataframe=True; altrimenti dict di numpy array.
        """
        self.set_eval()
        
        print(f"INIZIO GENERAZIONE: n_samples={n_samples}, temp={temperature}")

        all_outputs = []
        remaining   = n_samples

        while remaining > 0:
            bs = min(self.model_config.batch_size, remaining)
            z_s = torch.randn(bs, self.model_config.z_static_dim,                 device=self.device)
            z_t = torch.randn(bs, self.max_len, self.model_config.z_temporal_dim, device=self.device)

            out = self.generator(z_s, z_t, temperature=temperature, teacher_forcing=False)

            out_np = {}
            logger.info("DEBUG GENERAZIONE (primo batch)")
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    out_np[k] = v.cpu().numpy()
                    logger.info(f"gen[{k}]: {tuple(v.shape)}")
                elif isinstance(v, dict):
                    out_np[k] = {n: t.cpu().numpy() for n, t in v.items()}
            all_outputs.append(out_np)
            remaining -= bs

        # concatena batches
        final = {}
        for k in all_outputs[0]:
            if isinstance(all_outputs[0][k], dict):
                final[k] = {
                    n: np.concatenate([o[k][n] for o in all_outputs], axis=0)[:n_samples]
                    for n in all_outputs[0][k]
                }
            else:
                final[k] = np.concatenate([o[k] for o in all_outputs], axis=0)[:n_samples]

        # decodifica embedding statici → nearest neighbor
        if "static_cat_embed" in final and final["static_cat_embed"]:
            embedded_torch = {
                n: torch.tensor(v, device=self.device)
                for n, v in final["static_cat_embed"].items()
            }
            decoded = self.preprocessor.decode_embeddings(embedded_torch)
            final["static_cat_embed_decoded"] = {
                n: idx.cpu().numpy() for n, idx in decoded.items()
            }
            del final["static_cat_embed"]

        # inverse transform → DataFrame
        if return_dataframe:
            synth = {
                "temporal_cont": torch.tensor(final["temporal_cont"]),
                "temporal_cat":  {n: torch.tensor(v) for n, v in final["temporal_cat"].items()},
                "visit_mask":    torch.tensor(final["visit_mask"]),
                "visit_times":   torch.tensor(final["visit_times"]),
            }
            if "static_cont" in final:
                synth["static_cont"] = torch.tensor(final["static_cont"])
            if "static_cat" in final:
                #print(type(final["static_cat"]))
                #print(final["static_cat"])
                #synth["static_cat"]  = torch.tensor(final["static_cat"])
                static_cat = np.concatenate(
                    [final["static_cat"][k] for k in sorted(final["static_cat"].keys())],
                    axis=1)

                synth["static_cat"] = torch.from_numpy(static_cat).float()

            if "static_cat_embed_decoded" in final:
                synth["static_cat_embed_decoded"] = {
                    n: torch.tensor(v) for n, v in final["static_cat_embed_decoded"].items()
                }

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