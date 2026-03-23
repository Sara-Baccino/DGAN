"""
utils/losses.py  [v10]
================================================================================
Tutte le funzioni richieste da dgan.py sono presenti.

CAMBIAMENTI rispetto a v9:
  - static_cat_marginal_loss CORRETTA: p_t era calcolato su distribuzione target
    invece della probabilità predetta → gradiente sbagliato. Ora usa
    KL(target || fake_marginal) + CE standard (niente Focal CE senza ground truth).
  - coverage_loss RIMOSSA: sostituita da hard anchor in TimeEncoderV10.
  - inter_visit_interval_loss SEMPLIFICATA: rimosso l_mean_sq instabile.
  - cont_range_loss NUOVA (DoppelGANger-inspired): supervisiona min/max per paziente.
  - Lambda defaults ridotti in dgan.py (non qui): lambda_ivi 18→5, scat 12→5.
================================================================================
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


# ======================================================================
# WGAN LOSSES
# ======================================================================

def wgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return -(d_real.mean() - d_fake.mean())


def wgan_generator_loss(
    d_fake_static:   torch.Tensor,
    d_fake_temporal: torch.Tensor,
) -> torch.Tensor:
    return -(d_fake_static.mean() + d_fake_temporal.mean())


# ======================================================================
# GRADIENT PENALTY
# ======================================================================

def gradient_penalty(
    critic_fn,
    real:      torch.Tensor,
    fake:      torch.Tensor,
    device:    str,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    B         = real.size(0)
    eps_shape = (B,) + (1,) * (real.dim() - 1)
    eps       = torch.rand(eps_shape, device=device)
    interp    = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp  = critic_fn(interp)
    grads     = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads_flat = grads.reshape(B, -1)
    return ((grads_flat.norm(2, dim=1) - 1) ** 2).mean()


# ======================================================================
# IRREVERSIBILITY
# ======================================================================

def irreversibility_loss(
    irr_states: torch.Tensor,
    visit_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalizza hazard intermedi (~0.5) per variabili irreversibili."""
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    vm     = visit_mask.unsqueeze(-1).float()
    h      = irr_states.clamp(1e-6, 1 - 1e-6)
    ent    = -(h * h.log() + (1 - h) * (1 - h).log())
    masked = ent * vm
    denom  = vm.sum().clamp(min=1.0)
    return masked.sum() / denom


# ======================================================================
# CATEGORICAL FREQUENCY — utility
# ======================================================================

def compute_category_weights(
    real_cat_dict: Dict[str, torch.Tensor],
    visit_mask:    torch.Tensor,
    smoothing:     float = 1e-3,
    power:         float = 1.0,
) -> Dict[str, torch.Tensor]:
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    weights = {}
    for var_name, cat_tensor in real_cat_dict.items():
        device = cat_tensor.device
        if cat_tensor.dim() == 3:
            vm     = visit_mask.unsqueeze(-1).float().to(device)
            counts = (cat_tensor.float() * vm).sum(dim=(0, 1))
        elif cat_tensor.dim() == 2:
            counts = cat_tensor.float().sum(dim=0)
        else:
            continue
        total  = counts.sum().clamp(min=1.0)
        freq   = counts / total
        raw_w  = 1.0 / (freq + smoothing).pow(power)
        norm_w = raw_w / raw_w.sum()
        weights[var_name] = norm_w
    return weights


# ======================================================================
# CATEGORICAL FREQUENCY — GENERATOR
# ======================================================================

def categorical_frequency_loss_generator(
    fake_cat_dict: Dict[str, torch.Tensor],
    cat_weights:   Dict[str, torch.Tensor],
    visit_mask:    torch.Tensor,
    eps:           float = 1e-4,
) -> torch.Tensor:
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    losses = []
    for var_name, fake_tensor in fake_cat_dict.items():
        if var_name not in cat_weights:
            continue
        w = cat_weights[var_name].to(fake_tensor.device)
        if fake_tensor.dim() == 3:
            vm        = visit_mask.unsqueeze(-1).float().to(fake_tensor.device)
            n_visits  = vm.sum(dim=(0, 1)).clamp(min=1.0)
            fake_dist = (fake_tensor.float() * vm).sum(dim=(0, 1)) / n_visits
        elif fake_tensor.dim() == 2:
            fake_dist = fake_tensor.float().mean(dim=0)
        else:
            continue
        fake_dist = fake_dist.clamp(min=0.0)
        fake_dist = fake_dist / fake_dist.sum().clamp(min=1e-8)
        ce_loss   = -(w * (fake_dist + eps).log()).sum()
        losses.append(ce_loss)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


# ======================================================================
# CATEGORICAL FREQUENCY — DISCRIMINATOR
# ======================================================================

def categorical_frequency_loss_discriminator(
    real_cat_dict: Dict[str, torch.Tensor],
    fake_cat_dict: Dict[str, torch.Tensor],
    cat_weights:   Dict[str, torch.Tensor],
    visit_mask:    torch.Tensor,
) -> torch.Tensor:
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    losses = []
    for var_name in real_cat_dict:
        if var_name not in fake_cat_dict or var_name not in cat_weights:
            continue
        real_tensor = real_cat_dict[var_name]
        fake_tensor = fake_cat_dict[var_name]
        w           = cat_weights[var_name].to(real_tensor.device)
        if real_tensor.dim() == 3:
            vm        = visit_mask.unsqueeze(-1).float().to(real_tensor.device)
            n_visits  = vm.sum(dim=(0, 1)).clamp(min=1.0)
            real_dist = (real_tensor.float() * vm).sum(dim=(0, 1)) / n_visits
            fake_dist = (fake_tensor.float() * vm).sum(dim=(0, 1)) / n_visits
        elif real_tensor.dim() == 2:
            real_dist = real_tensor.float().mean(dim=0)
            fake_dist = fake_tensor.float().mean(dim=0)
        else:
            continue
        real_dist = real_dist.clamp(min=1e-8)
        real_dist = real_dist / real_dist.sum()
        fake_dist = fake_dist.clamp(min=1e-8)
        fake_dist = fake_dist / fake_dist.sum()
        m         = (0.5 * (real_dist + fake_dist)).clamp(min=1e-8)
        kl_real   = F.kl_div(m.log(), real_dist, reduction="none")
        kl_fake   = F.kl_div(m.log(), fake_dist, reduction="none")
        js        = (w * (0.5 * kl_real + 0.5 * kl_fake)).sum()
        losses.append(js)
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


# ======================================================================
# STATIC CATEGORICAL MARGINAL LOSS  [v10 — CORRETTA]
# ======================================================================

def static_cat_marginal_loss(
    fake_static_cat_soft: Dict[str, torch.Tensor],
    target_probs:         Dict[str, torch.Tensor],
    eps:                  float = 1e-6,
) -> torch.Tensor:
    """
    [v10] KL(target || fake_marginal) + CE standard per variabili rare.

    CORREZIONE rispetto a v9:
      p_t nella Focal CE era calcolato come somma pesata per target_probs
      (probabilità media pesata, non probabilità della classe predetta).
      Senza ground truth per campione non si può fare Focal CE corretta.

    NUOVO:
      1. KL(target || fake_marginal): penalizza fortemente classi rare ignorate.
      2. CE(fake_soft, target): supervisione campione-per-campione standard.
      3. Imbalance weight: amplifica variabili sbilanciate.
    """
    losses = []
    for name, p_soft in fake_static_cat_soft.items():
        if name not in target_probs:
            continue
        p_real = target_probs[name].to(p_soft.device).float()   # [n_cat]
        p_s    = p_soft.float()                                  # [B, n_cat]

        # 1. KL(target || fake_marginal)
        p_fake_marginal = p_s.mean(dim=0).clamp(min=eps)
        p_fake_marginal = p_fake_marginal / p_fake_marginal.sum()
        p_real_safe     = p_real.clamp(min=eps)
        kl = (p_real_safe * (p_real_safe / p_fake_marginal).log()).sum()

        # 2. CE standard campione-per-campione
        log_soft  = torch.log(p_s.clamp(min=eps))                 # [B, n_cat]
        ce        = -(p_real.unsqueeze(0) * log_soft).sum(dim=-1) # [B]
        ce_mean   = ce.mean()

        # 3. Imbalance weight
        min_p            = p_real.min().clamp(min=1e-6)
        imbalance_weight = (1.0 / min_p).clamp(max=20.0)

        losses.append(imbalance_weight * (kl + 0.5 * ce_mean))

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


# ======================================================================
# FOLLOWUP NORM LOSS
# ======================================================================

def followup_norm_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
) -> torch.Tensor:
    pred = pred_followup.float()
    real = real_followup.float().to(pred.device)

    l_mean = F.mse_loss(pred.mean(), real.mean())

    var_real = real.var().clamp(min=1e-6)
    var_pred = pred.var().clamp(min=1e-6)
    l_var    = F.relu(var_real - var_pred)

    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    q_real  = torch.quantile(real, qs)
    q_pred  = torch.quantile(pred, qs)
    l_quant = F.mse_loss(q_pred, q_real)

    return l_mean + 0.5 * l_var + 2.0 * l_quant


# ======================================================================
# FEATURE MATCHING LOSS
# ======================================================================

def feature_matching_loss(
    features_real: torch.Tensor,
    features_fake: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(features_fake.mean(dim=0), features_real.mean(dim=0))


# ======================================================================
# N_VISITS SUPERVISION LOSS
# ======================================================================

def n_visits_supervision_loss(
    n_v_pred: torch.Tensor,
    n_v_real: torch.Tensor,
) -> torch.Tensor:
    pred    = n_v_pred.float()
    real    = n_v_real.float().to(pred.device)
    l_mean  = F.smooth_l1_loss(pred, real)
    var_real = real.var().clamp(min=0.0)
    var_pred = pred.var().clamp(min=0.0)
    l_var    = F.relu(var_real - var_pred)
    qs       = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    q_real   = torch.quantile(real, qs)
    q_pred   = torch.quantile(pred, qs)
    l_quant  = F.mse_loss(q_pred, q_real)
    return l_mean + 0.1 * l_var + 1.0 * l_quant


# ======================================================================
# STATIC CONTINUOUS DISTRIBUTION LOSS
# ======================================================================

def static_cont_dist_loss(
    fake_sc: torch.Tensor,
    real_sc: torch.Tensor,
) -> torch.Tensor:
    fake    = fake_sc.float()
    real    = real_sc.float().to(fake.device)
    l_mean  = F.mse_loss(fake.mean(dim=0), real.mean(dim=0))
    var_real = real.var(dim=0).clamp(min=1e-6)
    var_pred = fake.var(dim=0).clamp(min=1e-6)
    l_var    = F.relu(var_real - var_pred).mean()
    qs       = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=fake.device)
    q_real   = torch.quantile(real, qs, dim=0)
    q_pred   = torch.quantile(fake, qs, dim=0)
    l_quant  = F.mse_loss(q_pred, q_real)
    return 1.0 * l_mean + 0.5 * l_var + 1.0 * l_quant


# ======================================================================
# INTER-VISIT INTERVAL LOSS  [v10 — semplificata]
# ======================================================================

def inter_visit_interval_loss(
    visit_times:    torch.Tensor,   # [B, T] delta_months dal Generator
    visit_mask:     torch.Tensor,   # [B, T] o [B, T, 1]
    real_intervals: torch.Tensor,   # [N_real] in mesi, precalcolati
) -> torch.Tensor:
    """
    [v10] Semplificata: mean error + quantile matching.
    Rimosso l_mean_sq (errore relativo): instabile quando
    dt_synth.mean() → 0 nelle prime epoche.
    """
    vm = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask

    dt    = visit_times[:, 1:] * vm[:, 1:]
    valid = dt > 1e-8
    if valid.sum() < 10:
        return torch.tensor(0.0, device=visit_times.device)

    dt_synth  = dt[valid]
    real_iv   = real_intervals.to(dt_synth.device)
    mean_real = real_iv.mean().clamp(min=1e-6)

    l_mean    = F.mse_loss(dt_synth.mean(), mean_real)

    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dt_synth.device)
    q_synth = torch.quantile(dt_synth, qs)
    q_real  = torch.quantile(real_iv,  qs)
    l_quant = F.mse_loss(q_synth, q_real)

    return l_mean + 2.0 * l_quant


# ======================================================================
# UNIFORMITY LOSS
# ======================================================================

def inter_visit_uniformity_loss(
    delta_months: torch.Tensor,   # [B, T]
    visit_mask:   torch.Tensor,   # [B, T] o [B, T, 1]
    cv2_target:   float = 1.0,
) -> torch.Tensor:
    """
    Penalizza CV² > target per-paziente (visite "bunched" tutte vicine).
    CV² = Var(delta) / Mean(delta)²
    """
    vm   = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask
    dm   = delta_months[:, 1:].float()
    vm_d = vm[:, 1:].float()

    n_valid  = vm_d.sum(dim=1).clamp(min=2.0)
    mean_dm  = (dm * vm_d).sum(dim=1) / n_valid
    diff_sq  = ((dm - mean_dm.unsqueeze(1)) ** 2) * vm_d
    var_dm   = diff_sq.sum(dim=1) / n_valid
    cv2      = var_dm / (mean_dm ** 2 + 1e-6)

    has_multi = (vm_d.sum(dim=1) > 2).float()
    penalty   = F.relu(cv2 - cv2_target) * has_multi
    return penalty.mean()


# ======================================================================
# CONT RANGE LOSS  [v10 — NUOVA, DoppelGANger-inspired]
# ======================================================================

def cont_range_loss(
    temporal_cont:    torch.Tensor,   # [B, T, n_cont]  z-score
    cont_ranges_pred: torch.Tensor,   # [B, 2*n_cont]   min+max predetti (Sigmoid → [0,1])
    visit_mask:       torch.Tensor,   # [B, T] o [B, T, 1]
    real_cont:        torch.Tensor,   # [B, T, n_cont]  reale per calibrazione
) -> torch.Tensor:
    """
    [v10] Supervisiona il range dei valori continui per-paziente.

    Il cont_range_head predice [min_norm, max_norm] ∈ [0,1] per ogni variabile,
    dove la normalizzazione è: norm = (z / Z_RANGE + 1.0) / 2.0, Z_RANGE=4.0.

    Due componenti:
      1. range_penalty:      penalizza valori sintetici fuori dal range predetto.
      2. range_supervision:  MSE tra range predetti e range reali (supervisione).
    """
    Z_RANGE = 4.0
    vm      = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask  # [B, T]
    B, T, n_cont = temporal_cont.shape
    vm_exp  = vm.unsqueeze(-1).float()   # [B, T, 1]

    # Separa min e max predetti, rescala in z-score
    min_pred = cont_ranges_pred[:, :n_cont]          # [B, n_cont] ∈ [0,1]
    max_pred = cont_ranges_pred[:, n_cont:]           # [B, n_cont] ∈ [0,1]
    min_z    = (min_pred * 2.0 - 1.0) * Z_RANGE      # [B, n_cont] ∈ [-4,+4]
    max_z    = (max_pred * 2.0 - 1.0) * Z_RANGE
    # Garantisce min < max
    min_z, max_z = torch.min(min_z, max_z), torch.max(min_z, max_z)

    # Penalizza valori sintetici fuori range
    below_min  = F.relu(min_z.unsqueeze(1) - temporal_cont) * vm_exp   # [B, T, n_cont]
    above_max  = F.relu(temporal_cont - max_z.unsqueeze(1)) * vm_exp
    n_valid    = vm.sum().clamp(min=1.0) * n_cont
    range_penalty = (below_min + above_max).sum() / n_valid

    # Supervisione range con i range reali del batch
    with torch.no_grad():
        tc_for_min = real_cont.clone().masked_fill(vm_exp == 0,  1e9)
        tc_for_max = real_cont.clone().masked_fill(vm_exp == 0, -1e9)
        real_min   = tc_for_min.min(dim=1).values    # [B, n_cont]
        real_max   = tc_for_max.max(dim=1).values    # [B, n_cont]
        real_min_norm = ((real_min.clamp(-Z_RANGE, Z_RANGE) / Z_RANGE + 1.0) / 2.0)
        real_max_norm = ((real_max.clamp(-Z_RANGE, Z_RANGE) / Z_RANGE + 1.0) / 2.0)

    range_supervision = (
        F.mse_loss(min_pred, real_min_norm) +
        F.mse_loss(max_pred, real_max_norm)
    )

    return range_penalty + 0.5 * range_supervision