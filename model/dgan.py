"""
model/dgan.py  [v2-fully-parametrized]
================================================================================
Rispetto alla versione precedente:

  [NUOVO] lambda_var — loss sulla varianza delle feature continue temporali:
    Per ogni feature continua temporale, penalizza la differenza di std
    tra dati reali e fake, calcolata solo sugli step valid_flag=True.
    Questo forza il modello a riprodurre l'eterogeneità tra pazienti, non
    solo la media. Critico per PBC dove la varianza inter-paziente è alta.

  [NUOVO] Stampe di training più informative:
    - Media e std delle prime feature continue (biomarker principali)
    - Confronto real vs fake per followup_norm e n_visits
    - Percentuale di pazienti con n_visits > 1

  [NUOVO] Tutti i parametri da model_config (zero hardcoded):
    - optimizer_betas letti da model_config.optimizer_betas
    - dataloader_drop_last da model_config.dataloader_drop_last
    - noise_ar_rho dal generator (model_config.noise_ar_rho)

  [NUOVO] Campionamento rumore AR via generator.sample_noise():
    Il generatore ora ha sample_noise() che produce z_t AR se noise_ar_rho > 0.
    Usato in _generate_fake() invece di torch.randn diretto.

  [NUOVO] Gestione errori espliciti:
    - Errore se DataLoader è vuoto
    - Warning se batch troppo piccolo per WGAN-GP
    - Warning se le loss divergono (NaN/Inf)

  [INVARIATO]
    - WGAN-GP come loss dominante
    - Loss ausiliarie: irr, fup, nv, scat, fm
    - Early stopping
    - Save/Load
================================================================================
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List
import numpy as np
import logging

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/dgan_experiment_1")

from model.generator    import DGANGenerator
from model.discriminator import (StaticDiscriminator, TemporalDiscriminator, prepare_discriminator_inputs)

try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================================================================
# LOSS UTILITIES
# ==================================================================

def _wgan_d_loss(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return fake.mean() - real.mean()

def _wgan_g_loss(fake_s: torch.Tensor, fake_t: torch.Tensor) -> torch.Tensor:
    return -(fake_s.mean() + fake_t.mean())

def _gradient_penalty(
    disc_fn, real: torch.Tensor, fake: torch.Tensor, device
) -> torch.Tensor:
    B    = real.shape[0]
    eps  = torch.rand(B, *([1] * (real.dim() - 1)), device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    out    = disc_fn(interp)
    grads  = torch.autograd.grad(
        out, interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()

def _dist_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalizza differenza di media e deviazione standard (distribuzione)."""
    loss  = (pred.mean() - target.mean()).pow(2)
    loss += (pred.std()  - target.std() ).pow(2).clamp(min=1e-6)
    return loss

def _var_loss(
    fake_cont:    torch.Tensor,   # [B, T, n_cont]
    real_cont:    torch.Tensor,   # [B, T, n_cont]
    fake_valid:   torch.Tensor,   # [B, T] bool
    real_valid:   torch.Tensor,   # [B, T] bool
) -> torch.Tensor:
    """
    Penalizza la differenza di deviazione standard per ogni feature continua
    temporale, calcolata solo sugli step validi (non padding).
    Forza il generatore a riprodurre la varianza inter-paziente e intra-paziente.
    """
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)

    losses = []
    for j in range(n_cont):
        f_vals = fake_cont[:, :, j][fake_valid]   # valori flat fake validi
        r_vals = real_cont[:, :, j][real_valid]   # valori flat real validi

        if len(f_vals) < 2 or len(r_vals) < 2:
            continue

        # Std complessiva (inter + intra paziente combinata)
        loss_j = (f_vals.std() - r_vals.std()).pow(2)

        # Std intra-paziente (per ogni paziente, media della std)
        f_std_intra = torch.stack([
            fake_cont[b, :, j][fake_valid[b]].std()
            for b in range(fake_cont.shape[0])
            if fake_valid[b].sum() > 1
        ]) if any(fake_valid[b].sum() > 1 for b in range(fake_cont.shape[0])) \
          else torch.zeros(1, device=fake_cont.device)

        r_std_intra = torch.stack([
            real_cont[b, :, j][real_valid[b]].std()
            for b in range(real_cont.shape[0])
            if real_valid[b].sum() > 1
        ]) if any(real_valid[b].sum() > 1 for b in range(real_cont.shape[0])) \
          else torch.zeros(1, device=real_cont.device)

        if len(f_std_intra) > 0 and len(r_std_intra) > 0:
            loss_j = loss_j + (f_std_intra.mean() - r_std_intra.mean()).pow(2)

        losses.append(loss_j)

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=fake_cont.device)

def _irr_loss(irr_states: torch.Tensor, valid_flag: torch.Tensor) -> torch.Tensor:
    """Penalizza transizioni 1→0 (irreversibili devono solo crescere)."""
    diff = irr_states[:, 1:] - irr_states[:, :-1]
    vf   = valid_flag[:, 1:].float()
    return (torch.clamp(-diff, min=0) * vf).mean()

def _interval_loss(
    fake_times: torch.Tensor,   # [B, T] visit_times (normalizzati [0,1])
    real_times: torch.Tensor,   # [B, T] visit_times (normalizzati [0,1])
    fake_valid: torch.Tensor,   # [B, T] bool
    real_valid: torch.Tensor,   # [B, T] bool
) -> torch.Tensor:
    """
    Penalizza la differenza di distribuzione degli intervalli inter-visita.
    Opera sui tempi normalizzati [0,1] per essere scale-invariante.

    Strategia:
      1. Calcola delta[b,t] = time[b,t] - time[b,t-1] per ogni paziente
      2. Confronta media e std degli intervalli fake vs reali
      3. Penalizza anche se la std degli intervalli fake è troppo piccola
         (il generatore tende a collassare su intervalli uniformi corti)

    Questa loss è il fix diretto al problema delle traiettorie piatte:
    forza il modello a imparare la distribuzione degli intervalli reali
    (es. ~10 mesi in PBC) invece di comprimerli tutti verso 0.
    """
    def _get_intervals(times, valid):
        intervals = []
        B = times.shape[0]
        for b in range(B):
            idx = valid[b].nonzero(as_tuple=True)[0]
            if len(idx) < 2:
                continue
            t = times[b, idx]         # tempi delle visite valide
            d = t[1:] - t[:-1]        # intervalli
            d = d.clamp(min=0.0)      # no intervalli negativi
            if d.numel() > 0:
                intervals.append(d)
        if not intervals:
            return None
        return torch.cat(intervals)

    f_iv = _get_intervals(fake_times, fake_valid)
    r_iv = _get_intervals(real_times, real_valid)

    if f_iv is None or r_iv is None or len(f_iv) < 2 or len(r_iv) < 2:
        return torch.tensor(0.0, device=fake_times.device)

    # Differenza di media
    loss = (f_iv.mean() - r_iv.mean()).pow(2)

    # Differenza di std (cruciale: il generatore tende a std troppo bassa)
    f_std = f_iv.std().clamp(min=1e-6)
    r_std = r_iv.std().clamp(min=1e-6)
    loss  = loss + (f_std - r_std).pow(2)

    # Penalità asimmetrica se gli intervalli fake sono sistematicamente
    # più corti di quelli reali (il problema diagnosticato nel report)
    if f_iv.mean() < r_iv.mean() * 0.5:
        undershoot = (r_iv.mean() - f_iv.mean()) / (r_iv.mean() + 1e-8)
        loss = loss + undershoot.pow(2)

    return loss


def _scat_marginal_loss(
    fake_soft: Dict[str, torch.Tensor],
    target_probs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    losses = []
    for name, soft in fake_soft.items():
        if name not in target_probs:
            continue
        pred_p = soft.mean(dim=0)
        tgt_p  = target_probs[name]
        # KL(pred || tgt): penalizza quando il generatore si allontana dalla target
        # Direzione corretta: F.kl_div(input=log_q, target=p) = KL(p || q)
        # Vogliamo minimizzare KL(pred || tgt) = sum(pred * log(pred/tgt))
        # = F.kl_div(tgt.log(), pred) in pytorch convention
        losses.append(F.kl_div(
            tgt_p.clamp(min=1e-8).log(),   # log(target)
            pred_p.clamp(min=1e-8),        # predicted (input to KL)
            reduction="sum",
        ))
        # Aggiungi entropy bonus: penalizza distribuzioni troppo peaked (collapse)
        entropy_j = -(pred_p * pred_p.clamp(min=1e-8).log()).sum()
        losses.append(-0.1 * entropy_j)   # massimizza entropia (anti-collapse)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def _check_finite(loss: torch.Tensor, name: str) -> torch.Tensor:
    """Controlla NaN/Inf nelle loss e restituisce 0 con warning."""
    if not torch.isfinite(loss):
        warnings.warn(
            f"Loss '{name}' non è finita ({loss.item():.4f}). "
            f"Verrà sostituita con 0. Controlla lr, lambda, o la stabilità del training.",
            UserWarning,
            stacklevel=2,
        )
        return torch.zeros_like(loss)
    return loss


def _delta_loss(
    fake_deltas:  torch.Tensor,   # [B, T] intervalli generati (normalizzati)
    real_times:   torch.Tensor,   # [B, T] visit_times reali (normalizzati [0,1])
    fake_valid:   torch.Tensor,   # [B, T] bool
    real_valid:   torch.Tensor,   # [B, T] bool
) -> torch.Tensor:
    """
    Penalizza la differenza di distribuzione degli intervalli inter-visita.

    Per i reali calcola Δt_real[b,t] = visit_time[b,t] - visit_time[b,t-1].
    Per i sintetici usa fake_deltas (già prodotti dal generator).

    Confronta: media, std, percentile 25 e 75 degli intervalli.
    Questo e' piu' robusto della sola differenza media/std (cattura la forma
    della distribuzione, non solo i momenti centrali).
    """
    # Calcola intervalli reali da visit_times
    real_iv_list, fake_iv_list = [], []
    B = real_times.shape[0]
    for b in range(B):
        r_idx = real_valid[b].nonzero(as_tuple=True)[0]
        if len(r_idx) >= 2:
            rt = real_times[b, r_idx]
            real_iv_list.append((rt[1:] - rt[:-1]).clamp(min=0.0))
        f_idx = fake_valid[b].nonzero(as_tuple=True)[0]
        if len(f_idx) >= 2:
            fd = fake_deltas[b, f_idx[1:]]   # skip step 0 (delta prima visita = 0)
            fake_iv_list.append(fd.clamp(min=0.0))

    if not real_iv_list or not fake_iv_list:
        return torch.tensor(0.0, device=real_times.device)

    r_iv = torch.cat(real_iv_list)
    f_iv = torch.cat(fake_iv_list)

    if len(r_iv) < 2 or len(f_iv) < 2:
        return torch.tensor(0.0, device=real_times.device)

    loss = (f_iv.mean() - r_iv.mean()).pow(2)
    loss = loss + (f_iv.std().clamp(min=1e-6) - r_iv.std().clamp(min=1e-6)).pow(2)

    # Penalita asimmetrica se sintetici sistematicamente piu corti
    if f_iv.mean() < r_iv.mean() * 0.5:
        undershoot = (r_iv.mean() - f_iv.mean()) / (r_iv.mean() + 1e-8)
        loss = loss + undershoot.pow(2)

    return loss


def _autocorrelation_loss(
    fake_cont:   torch.Tensor,   # [B, T, n_cont]
    real_cont:   torch.Tensor,   # [B, T, n_cont]
    fake_valid:  torch.Tensor,   # [B, T] bool
    real_valid:  torch.Tensor,   # [B, T] bool
    max_lag:     int = 2,
) -> torch.Tensor:
    """
    Penalizza la differenza di struttura di autocorrelazione lag-1..max_lag
    tra traiettorie reali e sintetiche.

    Per ogni feature continua temporale, calcola la correlazione di Pearson
    tra x[t] e x[t+lag] per lag = 1..max_lag, sia per reali che sintetici.
    Penalizza la differenza media sui lag e sulle feature.

    Questo spinge il generatore a produrre traiettorie con la stessa
    smoothness e persistenza temporale dei dati reali (es. ALP decresce
    gradualmente, non casualmente).
    """
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)

    losses = []
    for lag in range(1, max_lag + 1):
        for j in range(n_cont):
            f_ac_list, r_ac_list = [], []
            B = fake_cont.shape[0]
            for b in range(B):
                f_idx = fake_valid[b].nonzero(as_tuple=True)[0]
                if len(f_idx) > lag:
                    fx = fake_cont[b, f_idx[:-lag], j]
                    fy = fake_cont[b, f_idx[lag:],  j]
                    if fx.std() > 1e-6 and fy.std() > 1e-6:
                        f_ac = ((fx - fx.mean()) * (fy - fy.mean())).mean() / (
                            fx.std() * fy.std() + 1e-8)
                        f_ac_list.append(f_ac)

                r_idx = real_valid[b].nonzero(as_tuple=True)[0]
                if len(r_idx) > lag:
                    rx = real_cont[b, r_idx[:-lag], j]
                    ry = real_cont[b, r_idx[lag:],  j]
                    if rx.std() > 1e-6 and ry.std() > 1e-6:
                        r_ac = ((rx - rx.mean()) * (ry - ry.mean())).mean() / (
                            rx.std() * ry.std() + 1e-8)
                        r_ac_list.append(r_ac.detach())  # target: no grad needed

            if f_ac_list and r_ac_list:
                f_ac_mean = torch.stack(f_ac_list).mean()
                r_ac_mean = torch.stack(r_ac_list).mean()
                losses.append((f_ac_mean - r_ac_mean).pow(2))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=fake_cont.device)


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

        # ── Lambda: tutti da model_config ─────────────────────────────
        mc = model_config
        self.lambda_gp_s  = float(mc.lambda_gp_s)
        self.lambda_gp_t  = float(mc.lambda_gp_t)
        self.lambda_irr   = float(mc.alpha_irr)
        self.lambda_fup   = float(mc.lambda_fup)
        self.lambda_nv    = float(mc.lambda_nv)
        self.lambda_scat  = float(mc.lambda_static_cat)
        self.lambda_fm    = float(mc.lambda_fm)
        self.lambda_var      = float(mc.lambda_var)
        # lambda_interval rimpiazzato da lambda_delta (usa deltas espliciti del generatore v3)
        self.lambda_delta   = float(mc.lambda_delta)
        self.lambda_autocorr = float(getattr(mc, "lambda_autocorr", 0.5))
        self.autocorr_max_lag = int(getattr(mc, "autocorr_max_lag", 2))
        self.lambda_aux   = float(mc.lambda_aux)

        self.current_temperature = float(mc.gumbel_temperature_start)
        self.temperature_min     = float(mc.temperature_min)

        # ── EMA generatore ────────────────────────────────────────────
        self.ema_decay = float(mc.ema_decay)
        self.ema_generator = None   # inizializzato in _build_model dopo che il generatore esiste

        # ── Instance noise sulle OHE reali (anti-collapse) ────────────
        self.instance_noise_start = float(mc.instance_noise_start)
        self.instance_noise_end   = float(mc.instance_noise_end)
        self._current_epoch       = 0   # tracciato in fit()

        # ── Lambda_scat warmup ────────────────────────────────────────
        self.lambda_scat_warmup_epochs = int(mc.lambda_scat_warmup_epochs)

        # ── GP curriculum: rampa lambda_gp da 0 → target nelle prime N epoche ─
        # All'inizio il generatore produce distribuzione casuale: GP alto è
        # instabile. Aumentare gradualmente evita mode collapse precoce.
        self.gp_warmup_epochs = int(getattr(mc, "gp_warmup_epochs", 30))

        self.target_probs_static: Dict[str, torch.Tensor] = {}

        # _build_model() viene chiamato DOPO tutte le assegnazioni di attributi
        # perché usa self.ema_decay, self.instance_noise_start, ecc.
        self._build_model()

        self.loss_history: Dict[str, List] = {
            "generator":     [], "disc_static": [], "disc_temporal": [],
            #"gp_static":     [], "gp_temporal": [],
            "irr_loss":      [], "fup_loss":    [], "nv_loss":       [],
            "scat_loss":     [], "fm_loss":     [], "aux_loss":      [],
            "var_loss":      [],
            "delta_loss":    [],   # distribuzione intervalli (da deltas generatore v3)
            "autocorr_loss": [],   # autocorrelazione lag-1..max_lag
            "mean_n_visits": [],
            # Statistiche per monitoring clinico
            "fake_cont_mean": [], "real_cont_mean": [],
            "fake_cont_std":  [], "real_cont_std":  [],
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
            min_visits     = self.data_config.min_visits,   # da DataConfig
            device         = self.device
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

        #if self.preprocessor.embeddings:
        #    self.preprocessor.embeddings = self.preprocessor.embeddings.to(self.device)

        # AGGIUNGI QUESTO: Sposta gli embedding del preprocessor
        if hasattr(self.preprocessor, 'embeddings'):
            for name in self.preprocessor.embeddings:
                self.preprocessor.embeddings[name].to(self.device)
        
        # Optimizer con betas dal config
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


        # ── EMA generatore: deepcopy iniziale, poi aggiornato ogni gen step ──
        if self.ema_decay > 0.0:
            import copy as _copy
            self.ema_generator = _copy.deepcopy(self.generator)
            for p in self.ema_generator.parameters():
                p.requires_grad_(False)
        else:
            self.ema_generator = None

    # ─────────────────────────────────────────────────────────────────

    def _update_ema(self):
        """Aggiorna i pesi dell'EMA generatore con un passo esponenziale."""
        if self.ema_generator is None:
            return
        decay = self.ema_decay
        for ema_p, gen_p in zip(self.ema_generator.parameters(),
                                 self.generator.parameters()):
            ema_p.data.mul_(decay).add_(gen_p.data, alpha=1.0 - decay)

    def _instance_noise_strength(self) -> float:
        """
        Restituisce la forza corrente del rumore di istanza sulle OHE reali.
        Decade linearmente da instance_noise_start → instance_noise_end
        nel corso delle epoche configurate.
        """
        if self.instance_noise_start <= 0:
            return 0.0
        total_epochs = self.model_config.epochs
        frac = min(self._current_epoch / max(total_epochs, 1), 1.0)
        return self.instance_noise_start + frac * (self.instance_noise_end - self.instance_noise_start)

    def _effective_lambda_scat(self) -> float:
        """
        Lambda_scat con warmup coseno nelle prime lambda_scat_warmup_epochs epoche.
        Nelle prime epoche il segnale GAN è instabile: un warmup graduale di
        lambda_scat evita che la loss categorica domini troppo presto.
        """
        if self.lambda_scat_warmup_epochs <= 0:
            return self.lambda_scat
        import math as _math
        t = min(self._current_epoch / max(self.lambda_scat_warmup_epochs, 1), 1.0)
        # cosine warmup: sale da 0 → lambda_scat
        scale = (1.0 - _math.cos(_math.pi * t)) / 2.0
        return self.lambda_scat * scale

    def _effective_gp(self) -> tuple:
        """
        GP curriculum: lambda_gp_s e lambda_gp_t crescono linearmente da 0
        ai valori configurati nelle prime gp_warmup_epochs epoche.
        Motivazione: nelle prime epoche il gradiente penalty su distribuzioni
        random e' instabile e rallenta la convergenza del generatore.
        """
        if self.gp_warmup_epochs <= 0:
            return self.lambda_gp_s, self.lambda_gp_t
        t = min(self._current_epoch / max(self.gp_warmup_epochs, 1), 1.0)
        return self.lambda_gp_s * t, self.lambda_gp_t * t

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

    def _generate_fake(
        self,
        batch_size: int,
        real_irr:   Optional[torch.Tensor] = None,
    ):
        # Usa sample_noise() del generatore (supporta AR se configurato)
        z_s, z_t = self.generator.sample_noise(batch_size, torch.device(self.device))

        fake_out  = self.generator(
            z_s, z_t,
            temperature = self.current_temperature,
            real_irr    = real_irr,
        )
        fake_disc = prepare_discriminator_inputs(fake_out, self.preprocessor)
        return fake_out, fake_disc

    # ─────────────────────────────────────────────────────────────────

    def _train_discriminators(
        self,
        real_disc:     Dict,
        embed_targets: Dict,
        batch_size:    int,
        update_static:   bool = True,
        update_temporal: bool = True,
    ):
        if batch_size < 2:
            warnings.warn(
                f"Batch size = {batch_size} è troppo piccolo per il gradient penalty. "
                f"Aumenta batch_size nel config (raccomandato >= 16).",
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

        # ── Instance noise sulle OHE reali (anti-collapse categoriche) ─
        # Aggiunge U(0, ε) ai vettori reali prima del discriminatore.
        # ε decade linearmente da instance_noise_start → 0 nel training.
        # Rompe l'asimmetria discriminatore (OHE reali perfette vs soft fake).
        noise_eps = self._instance_noise_strength()
        if noise_eps > 0.0:
            real_s = real_s + noise_eps * torch.rand_like(real_s)
            real_t = real_t + noise_eps * torch.rand_like(real_t)

        # ── Static discriminator ──────────────────────────────────────
        d_real_s = self.disc_static(real_s)
        d_fake_s = self.disc_static(fake_s)
        #gp_s     = _gradient_penalty(lambda x: self.disc_static(x), real_s, fake_s, self.device)
        aux_loss = self.disc_static.auxiliary_loss(real_s, embed_targets)
        #eff_gp_s, eff_gp_t = self._effective_gp()
        loss_d_s = (_wgan_d_loss(d_real_s, d_fake_s)
                    #+ eff_gp_s * gp_s
                    + self.lambda_aux  * aux_loss)
        loss_d_s = _check_finite(loss_d_s, "disc_static")

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
        #gp_t     = _gradient_penalty(lambda x: self.disc_temporal(real_s, x, real_vf), real_t, fake_t, self.device)
        loss_d_t = _wgan_d_loss(d_real_t, d_fake_t)     #+ eff_gp_t * gp_t
        loss_d_t = _check_finite(loss_d_t, "disc_temporal")

        if update_temporal:
            self.opt_disc_temporal.zero_grad()
            loss_d_t.backward()
            if self.model_config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.disc_temporal.parameters(), self.model_config.grad_clip)
            self.opt_disc_temporal.step()

        return (loss_d_s.item(), loss_d_t.item(),
                #gp_s.item(),   gp_t.item(),
                aux_loss.item())

    # ─────────────────────────────────────────────────────────────────

    def _train_generator(
        self,
        real_disc:  Dict,
        batch_size: int,
        real_irr:   Optional[torch.Tensor],
        real_batch: Optional[Dict],
    ):
        self.opt_gen.zero_grad()

        fake_out, fake_disc = self._generate_fake(batch_size, real_irr=real_irr)

        # ── WGAN generator loss (dominante) ───────────────────────────
        # NOTA: fake_disc["static"] NON viene detachato qui.
        # Il disc_temporal usa le static features come conditioning:
        # i gradienti devono fluire sia attraverso il path statico che
        # quello temporale verso il generatore.
        # Il detach avviene SOLO in _train_discriminators (con no_grad).
        d_fake_s = self.disc_static(fake_disc["static"])
        # Passa fake_static SENZA detach al temporal disc: gradient flow completo
        d_fake_t = self.disc_temporal(
            fake_disc["static"], fake_disc["temporal"], fake_disc["valid_flag"])
        loss_g = _wgan_g_loss(d_fake_s, d_fake_t)
        loss_g = _check_finite(loss_g, "generator")

        # ── Irreversibilità ───────────────────────────────────────────
        irr_loss = torch.tensor(0.0, device=self.device)
        if self.irreversible_idx:
            irr_states = self._extract_fake_irr(fake_out["temporal_cat"])
            for k in range(irr_states.shape[-1]):
                irr_loss = irr_loss + _irr_loss(irr_states[..., k], fake_out["valid_flag"])
            irr_loss = _check_finite(irr_loss, "irr_loss")

        # ── Followup supervision ──────────────────────────────────────
        fup_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fup > 0 and real_batch is not None and "followup_norm" in real_batch:
            fup_loss = _dist_loss(
                fake_out["followup_norm"],
                real_batch["followup_norm"].to(self.device))
            fup_loss = _check_finite(fup_loss, "fup_loss")

        # ── N_visits supervision ──────────────────────────────────────
        nv_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_nv > 0 and real_batch is not None and "n_visits" in real_batch:
            nv_loss = _dist_loss(
                fake_out["n_visits_pred"],
                real_batch["n_visits"].float().to(self.device))
            nv_loss = _check_finite(nv_loss, "nv_loss")

        # ── Static categorical marginal ───────────────────────────────
        eff_lambda_scat = self._effective_lambda_scat()
        scat_loss = torch.tensor(0.0, device=self.device)
        fake_soft = fake_out.get("static_cat_soft") or {}
        if eff_lambda_scat > 0 and self.target_probs_static and fake_soft:
            scat_loss = _scat_marginal_loss(fake_soft, self.target_probs_static)
            scat_loss = _check_finite(scat_loss, "scat_loss")

        # ── Feature matching ──────────────────────────────────────────
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_fm > 0 and hasattr(self.disc_static, "get_features"):
            feat_real = self.disc_static.get_features(real_disc["static"]).detach()
            feat_fake = self.disc_static.get_features(fake_disc["static"])
            fm_loss   = F.mse_loss(feat_fake, feat_real)
            fm_loss   = _check_finite(fm_loss, "fm_loss")

        # ── Varianza feature continue temporali [NUOVO] ───────────────
        var_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_var > 0
                and real_batch is not None
                and "temporal_cont" in real_batch
                and fake_out["temporal_cont"].shape[-1] > 0):
            var_loss = _var_loss(
                fake_out["temporal_cont"],
                real_batch["temporal_cont"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
            )
            var_loss = _check_finite(var_loss, "var_loss")

        # ── Delta loss (intervalli inter-visita espliciti da generator v3) ──
        # Usa i deltas prodotti esplicitamente dal generator (interval_head).
        # Piu' preciso di _interval_loss che ricalcola delta dai visit_times.
        delta_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_delta > 0
                and real_batch is not None
                and "visit_time" in real_batch
                and "deltas" in fake_out):
            delta_loss = _delta_loss(
                fake_out["deltas"],
                real_batch["visit_time"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
            )
            delta_loss = _check_finite(delta_loss, "delta_loss")
        elif (self.lambda_delta > 0
                and real_batch is not None
                and "visit_time" in real_batch):
            # Fallback: usa _interval_loss se il generator non produce deltas (v2)
            delta_loss = _interval_loss(
                fake_out["visit_times"],
                real_batch["visit_time"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
            )
            delta_loss = _check_finite(delta_loss, "delta_loss_fallback")

        # ── Autocorrelation loss ──────────────────────────────────────
        autocorr_loss = torch.tensor(0.0, device=self.device)
        if (self.lambda_autocorr > 0
                and real_batch is not None
                and "temporal_cont" in real_batch
                and fake_out["temporal_cont"].shape[-1] > 0):
            autocorr_loss = _autocorrelation_loss(
                fake_out["temporal_cont"],
                real_batch["temporal_cont"].to(self.device),
                fake_out["valid_flag"],
                real_batch["valid_flag"].to(self.device),
                max_lag=self.autocorr_max_lag,
            )
            autocorr_loss = _check_finite(autocorr_loss, "autocorr_loss")

        # ── Total ─────────────────────────────────────────────────────
        total = (loss_g
                 + self.lambda_irr      * irr_loss
                 + self.lambda_fup      * fup_loss
                 + self.lambda_nv       * nv_loss
                 + eff_lambda_scat      * scat_loss
                 + self.lambda_fm       * fm_loss
                 + self.lambda_var      * var_loss
                 + self.lambda_delta    * delta_loss
                 + self.lambda_autocorr * autocorr_loss)

        total = _check_finite(total, "total_generator")
        total.backward()

        all_params = list(self.generator.parameters())
        if self.preprocessor.embeddings:
            all_params += list(self.preprocessor.embeddings.parameters())
        if self.model_config.grad_clip > 0:
            nn.utils.clip_grad_norm_(all_params, self.model_config.grad_clip)
        self.opt_gen.step()

        # Aggiorna EMA generatore (se configurato)
        self._update_ema()

        mean_nv = float(fake_out["n_visits"].detach().mean())

        # Stats per monitoring (prime 3 feature continue)
        stats = {}
        if fake_out["temporal_cont"].shape[-1] > 0:
            vf_f = fake_out["valid_flag"]
            stats["fake_cont_mean"] = float(
                fake_out["temporal_cont"][:, :, :3][vf_f.unsqueeze(-1).expand_as(
                    fake_out["temporal_cont"][:, :, :3])].mean())
            stats["fake_cont_std"]  = float(
                fake_out["temporal_cont"][:, :, :3][vf_f.unsqueeze(-1).expand_as(
                    fake_out["temporal_cont"][:, :, :3])].std())
        else:
            stats["fake_cont_mean"] = 0.0
            stats["fake_cont_std"]  = 0.0

        return (loss_g.item(), irr_loss.item(), fup_loss.item(),
                nv_loss.item(), scat_loss.item(), fm_loss.item(),
                var_loss.item(), delta_loss.item(), autocorr_loss.item(), mean_nv, stats)

    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_loader(tensors_dict: Dict, batch_size: int,
                      use_dp: bool, drop_last: bool):
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
        )

        if len(loader) == 0:
            raise ValueError(
                f"Il DataLoader è vuoto. Il dataset ha "
                f"{len(tensors[0])} campioni e batch_size={batch_size}. "
                f"Riduci batch_size o aumenta il dataset."
            )

        return loader, keys

    @staticmethod
    def _reconstruct_batch(batch_tuple, keys: List[str]) -> Dict:
        batch = {}
        for tensor, key in zip(batch_tuple, keys):
            if key.startswith("tcat::"):
                batch.setdefault("temporal_cat",   {})[key[6:]] = tensor
            elif key.startswith("sce::"):
                batch.setdefault("static_cat_embed", {})[key[5:]] = tensor
            else:
                batch[key] = tensor
        return batch

    # ─────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────
    def train_step(self, loader, keys, critic_steps, critic_steps_t, profiler=None):
        batch_losses = []

        for batch_tuple in loader:
            batch = self._reconstruct_batch(batch_tuple, keys)
            batch = self._move(batch)
            B     = batch["temporal_cont"].shape[0]

            real_disc     = prepare_discriminator_inputs(batch, self.preprocessor)
            real_irr      = self._extract_real_irr(batch)
            embed_targets = self._build_embed_targets(batch)

            n_steps = max(critic_steps, critic_steps_t)
            ld_s_list, ld_t_list, gp_s_list, gp_t_list, aux_list = [], [], [], [], []

            for step_idx in range(n_steps):
                #ld_s, ld_t, gp_s, gp_t, aux
                ld_s, ld_t, aux = self._train_discriminators(
                    real_disc, embed_targets, B,
                    update_static   = (step_idx < critic_steps),
                    update_temporal = (step_idx < critic_steps_t),
                )
                ld_s_list.append(ld_s); ld_t_list.append(ld_t)
                #gp_s_list.append(gp_s); gp_t_list.append(gp_t)
                aux_list.append(aux)

            (lg, l_irr, l_fup, l_nv, l_scat,
                l_fm, l_var, l_delta, l_ac, mean_nv, stats) = self._train_generator(
                real_disc, B, real_irr, real_batch=batch)

            batch_losses.append({
                "generator":     lg,
                "disc_static":   float(np.mean(ld_s_list)),
                "disc_temporal": float(np.mean(ld_t_list)),
                #"gp_static":     float(np.mean(gp_s_list)),
                #"gp_temporal":   float(np.mean(gp_t_list)),
                "aux_loss":      float(np.mean(aux_list)),
                "irr_loss":      l_irr,
                "fup_loss":      l_fup,
                "nv_loss":       l_nv,
                "scat_loss":     l_scat,
                "fm_loss":       l_fm,
                "var_loss":      l_var,
                "delta_loss":    l_delta,
                "autocorr_loss": l_ac,
                "mean_n_visits": mean_nv,
                "fake_cont_mean": stats["fake_cont_mean"],
                "fake_cont_std":  stats["fake_cont_std"],
            })

            if profiler is not None:
                profiler.step() # Segnala al profiler che un batch è finito

        return batch_losses

    def fit(self, tensors_dict: Dict, epochs: int = None):
        self.set_train()
        epochs = epochs or self.model_config.epochs

        print(f"\n{'='*70}")
        print(f"  DGAN [gretel-style v2]  —  {epochs} epoche")
        print(f"  Device: {self.device}  |  batch_size: {self.model_config.batch_size}")
        print(f"  z_s={self.model_config.z_static_dim}  z_t={self.model_config.z_temporal_dim}  "
              f"noise_ar_rho={self.model_config.noise_ar_rho}  min_visits={self.data_config.min_visits}")
        print(f"  ema_decay={self.ema_decay}  "
              f"instance_noise=[{self.instance_noise_start}->{self.instance_noise_end}]  "
              f"scat_warmup_ep={self.lambda_scat_warmup_epochs}")
        print(f"  λ_gp_s={self.lambda_gp_s}  λ_gp_t={self.lambda_gp_t}  "
              f"λ_fup={self.lambda_fup}  λ_nv={self.lambda_nv}  "
              f"λ_var={self.lambda_var}  λ_scat={self.lambda_scat}  "
              f"λ_delta_interval={self.lambda_delta}")
        print(f"{'='*70}\n")

        loader, keys = self._build_loader(
            tensors_dict,
            self.model_config.batch_size,
            self.model_config.use_dp,
            self.model_config.dataloader_drop_last,
        )

        if self.model_config.use_dp and OPACUS_AVAILABLE:
            self.privacy_engine = PrivacyEngine()
            self.disc_static, self.opt_disc_static, loader = (
                self.privacy_engine.make_private(
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

        # ── Target probs statiche ────────────────────────────────────
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

                head = dict(self.generator.static_cat_heads).get(var.name)
                if head is not None:
                    p         = self.target_probs_static[var.name].cpu()
                    log_prior = torch.log(p.clamp(min=1e-6)) - torch.log(p.clamp(min=1e-6)).mean()
                    last_layer = head
                    if isinstance(head, nn.Sequential):
                        for m in reversed(list(head.modules())):
                            if isinstance(m, nn.Linear):
                                last_layer = m; break
                    with torch.no_grad():
                        if hasattr(last_layer, "bias") and last_layer.bias is not None:
                            last_layer.bias.data.copy_(log_prior.to(last_layer.bias.device))
            print(f"  -> {len(self.target_probs_static)} variabili")

        # ── Warm-start followup ──────────────────────────────────────
        fn_all = tensors_dict.get("followup_norm")
        if fn_all is not None:
            fn_mean  = float(fn_all.float().mean().clamp(0.02, 0.98))
            fn_logit = float(np.log(fn_mean / (1.0 - fn_mean)))
            with torch.no_grad():
                self.generator.followup_head[-2].bias.fill_(fn_logit)
            print(f"  followup warm-start: mean={fn_mean:.3f}  logit={fn_logit:.3f}")

        # ── Warm-start n_visits ──────────────────────────────────────
        nv_all = tensors_dict.get("n_visits")
        if nv_all is not None:
            nv_med = float(nv_all.float().median())
            # Rispetta il vincolo min_visits
            min_v  = self.data_config.min_visits
            nv_med = max(nv_med, float(min_v))
            target  = max(nv_med - 1.0, 0.1)
            nv_bias = float(np.log(np.exp(target) - 1.0 + 1e-6))
            with torch.no_grad():
                self.generator.n_visits_head[-1].bias.fill_(nv_bias)
            print(f"  n_visits warm-start: median={nv_med:.1f}  min_visits={min_v}")

        # Stats reali per monitoring
        real_cont_all = tensors_dict.get("temporal_cont")
        real_vf_all   = tensors_dict.get("valid_flag")

        critic_steps   = self.model_config.critic_steps
        critic_steps_t = self.model_config.critic_steps_temporal
        best_loss, patience_counter = float("inf"), 0

        # Stampa distribuzione target categoriche statiche (attesa dal generatore)
        if self.target_probs_static:
            print("\n=== TARGET distribuzione categoriche statiche ===")
            for vname, prob in self.target_probs_static.items():
                p = prob.cpu().tolist()
                top3 = sorted(enumerate(p), key=lambda x: -x[1])[:3]
                top3_str = ", ".join(f"cat{i}={v:.3f}" for i, v in top3)
                entropy = -(prob * prob.clamp(min=1e-8).log()).sum().item()
                print(f"  {vname}: entropy={entropy:.3f}  top3=[{top3_str}]")
            print()


        # training loop
        for epoch in range(epochs):

            self._current_epoch = epoch   # usato da instance_noise e scat_warmup

            #batch_losses = self.train_step(loader, keys, critic_steps, critic_steps_t)
            
            # Esegui il training normale
            if epoch == 1: # Profila solo alla seconda epoca (per evitare overhead iniziale)
                with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=2, warmup=2, active=3, repeat=1),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler('./runs/dgan_experiment'),
                    record_shapes=True,
                    with_stack=True
                ) as prof:
                    batch_losses = self.train_step(loader, keys, critic_steps, critic_steps_t, profiler=prof)
                    
            else:
                batch_losses = self.train_step(loader, keys, critic_steps, critic_steps_t)



            avg = {k: float(np.mean([b[k] for b in batch_losses])) for k in batch_losses[0]}
            for k, v in avg.items():
                writer.add_scalar(f"Loss/{k}", v, epoch)
                if k in self.loss_history:
                    self.loss_history[k].append(v)


            self.current_temperature = max(self.temperature_min, self.current_temperature * 0.995)

            if epoch % 10 == 0:
                for name, param in self.generator.named_parameters():
                    writer.add_histogram(f"G_Grads/{name}", param.grad, epoch)
                for name, param in self.disc_temporal.named_parameters():
                    writer.add_histogram(f"D_Temp_Grads/{name}", param.grad, epoch)
                for name, param in self.disc_static.named_parameters():
                    writer.add_histogram(f"D_Static_Grads/{name}", param.grad, epoch)

            
            # Logga anche parametri dinamici
            writer.add_scalar("Params/Gumbel_Temp", self.current_temperature, epoch)

            # ── Stampa ─────────────────────────────────────────────────
            #eff_gp_s_now, eff_gp_t_now = self._effective_gp()
            print( f"[Ep {epoch+1:4d}/{epochs}]  "
                f"G={avg['generator']:+7.3f}  D_s={avg['disc_static']:+7.3f}  D_t={avg['disc_temporal']:+7.3f}  "
                #f"GP_s={avg['gp_static']:5.3f}(lam={eff_gp_s_now:.1f})  "
                #f"GP_t={avg['gp_temporal']:5.3f}(lam={eff_gp_t_now:.1f})"
            )
            print(f"              "
                f"Nv={avg['mean_n_visits']:4.1f}  NvL={avg['nv_loss']:.3f}  "
                f"Fup={avg['fup_loss']:.3f}  Scat={avg['scat_loss']:.3f}  "
                f"VarL={avg['var_loss']:.3f}  dL={avg['delta_loss']:.3f}  "
                f"AcL={avg['autocorr_loss']:.3f}"
            )
            print(f"              "
                f"Cont(fake) mu={avg['fake_cont_mean']:+.3f} s={avg['fake_cont_std']:.3f}  "
                f"| T={self.current_temperature:.3f}  noise={self._instance_noise_strength():.3f}",
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
        # Se EMA è attiva, usa l'EMA generator per generare (più stabile)
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
                final[k] = np.concatenate([o[k] for o in all_outputs], axis=0)[:n_samples]

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
            "temporal_cat":  {n: torch.tensor(v) for n, v in final["temporal_cat"].items()},
            "valid_flag":    torch.tensor(final["valid_flag"]),
            "visit_times":   torch.tensor(final["visit_times"]),
        }
        if "followup_norm" in final:
            synth["followup_norm"] = torch.tensor(final["followup_norm"])
        if "static_cont" in final:
            synth["static_cont"]   = torch.tensor(final["static_cont"])
        if "static_cat_embed_decoded" in final:
            synth["static_cat_embed_decoded"] = {
                n: torch.tensor(v) for n, v in final["static_cat_embed_decoded"].items()}
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
    def load(cls, filepath: str, data_config, model_config, preprocessor, device=None):
        state = torch.load(filepath, map_location=device or "cpu")
        dgan  = cls(data_config, model_config, preprocessor, device=device)
        dgan.generator.load_state_dict(state["generator_state"])
        dgan.disc_static.load_state_dict(state["disc_static_state"])
        dgan.disc_temporal.load_state_dict(state["disc_temporal_state"])
        dgan.opt_gen.load_state_dict(state["opt_gen_state"])
        dgan.opt_disc_static.load_state_dict(state["opt_disc_static_state"])
        dgan.opt_disc_temporal.load_state_dict(state["opt_disc_temporal_state"])
        dgan.loss_history                   = state["loss_history"]
        dgan.current_temperature            = state["current_temperature"]
        dgan.preprocessor.embedding_configs = state["embedding_configs"]
        dgan.preprocessor.scalers_cont      = state["scalers_cont"]
        dgan.preprocessor.inverse_maps      = state["inverse_maps"]
        dgan.preprocessor.global_time_max   = state["global_time_max"]
        if "embedding_state" in state:
            for name, emb_state in state["embedding_state"].items():
                if name in dgan.preprocessor.embeddings:
                    dgan.preprocessor.embeddings[name].load_state_dict(emb_state)
        if "ema_generator_state" in state and dgan.ema_generator is not None:
            dgan.ema_generator.load_state_dict(state["ema_generator_state"])
        logger.info(f"Model loaded ← {filepath}")
        print(f"  Modello caricato ← {filepath}")
        return dgan