"""
utils/losses.py  [single-gru-fix]
================================================================================
CAMBIAMENTI RISPETTO ALLA VERSIONE PRECEDENTE:

  [RIMOSSA] fup_coverage_loss
    Era la causa degli intervalli enormi (84 mesi vs 10 reali).
    Spingeva i delta verso valori grandi per coprire il follow-up,
    indipendentemente dal numero di visite. Sostituita da:

  [NUOVO] time_budget_loss
    Penalizza DUE cose contemporaneamente:
      a) ultima visita < target (coverage undershoot) — come prima
      b) intervalli > max_interval_norm — penalizza delta troppo grandi
    Così il modello impara a coprire il follow-up aumentando le visite,
    non aumentando gli intervalli.
    max_interval_norm = 1.0 / min_visits (es. 0.5 con min_visits=2)

  [MODIFICATO] static_cat_marginal_loss
    Aggiunto gradient clipping interno alla loss (clamp del KL a 10.0)
    per prevenire l'esplosione che causava Scat crescente.
    La causa del Scat crescente era: collasso GRU → distribuzioni degeneri
    → KL divergenza → inf. Con single GRU + clamp questo è risolto.

  [INVARIATO]
    intra_patient_variance_loss, variance_loss, autocorrelation_loss,
    delta_distribution_loss, n_visits_loss, followup_norm_loss,
    time_consistency_loss, gradient_penalty, tutte le altre.
================================================================================
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List


# ======================================================================
# WGAN
# ======================================================================

def wgan_discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    return -(d_real.mean() - d_fake.mean())


def wgan_generator_loss(d_fake_static: torch.Tensor, d_fake_temporal: torch.Tensor) -> torch.Tensor:
    return -(d_fake_static.mean() + d_fake_temporal.mean())


def gradient_penalty(critic_fn, real, fake, device, lambda_gp=10.0):
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
    return lambda_gp * ((grads_flat.norm(2, dim=1) - 1) ** 2).mean()


# ======================================================================
# [NUOVO] TIME BUDGET LOSS — sostituisce fup_coverage_loss
# ======================================================================

def time_budget_loss(
    visit_times_raw:    torch.Tensor,   # [B, T] cumsum/fup, scala [0,∞)
    deltas_raw:         torch.Tensor,   # [B, T] Δt grezzi
    valid_flag:         torch.Tensor,   # [B, T] bool
    n_visits:           torch.Tensor,   # [B] numero visite predetto
    followup_norm:      torch.Tensor,   # [B] ∈ [0,1]
    target_coverage:    float = 0.85,
    max_interval_norm:  float = 0.35,   # Δt massimo accettabile in scala [0,1]
) -> torch.Tensor:
    """
    Penalizza simultaneamente:
      1. Coverage undershoot: ultima visita < target_coverage * 1.0
      2. Intervalli troppo grandi: Δt > max_interval_norm

    max_interval_norm = 0.35 significa che un singolo delta non può
    essere più di 35% del follow-up totale. Con n_visits=5, questo
    impone almeno ~3 intervalli realistici prima di coprire il follow-up.

    Questo è il fix critico: prima, fup_coverage_loss spingeva i delta
    verso valori grandi (es. 0.5 * followup_norm per visita) per coprire
    il follow-up con 2 visite. Ora, la penalità su max_interval_norm forza
    il modello a usare più visite con delta più piccoli.

    Args:
        max_interval_norm: soglia per gli intervalli in scala [0,1].
            Con n_visits medio = 5 e follow-up uniformemente distribuito,
            ogni Δt dovrebbe essere ~0.20. Usiamo 0.35 come soglia morbida
            (1.75× il valore ideale) per non essere troppo restrittivi.
    """
    if valid_flag.dtype != torch.bool:
        valid_flag = valid_flag.bool()

    # 1. Coverage undershoot
    t_pos  = torch.arange(visit_times_raw.shape[1],
                          dtype=torch.float32, device=visit_times_raw.device)
    idx    = (valid_flag.float() * (t_pos.unsqueeze(0) + 1)).argmax(dim=1)
    t_last = visit_times_raw.gather(1, idx.unsqueeze(1)).squeeze(1)    # [B]
    loss_under = F.relu(target_coverage - t_last).pow(2).mean()

    # 2. Penalità su intervalli troppo grandi
    # Considera solo i delta delle visite valide (esclude delta[0]=0 e padding)
    # e penalizza quelli > max_interval_norm
    big_delta_penalty = torch.tensor(0.0, device=deltas_raw.device)
    B = deltas_raw.shape[0]
    penalties = []
    for b in range(B):
        f_idx = valid_flag[b].nonzero(as_tuple=True)[0]
        if len(f_idx) < 2:
            continue
        dt = deltas_raw[b, f_idx[1:]]   # skip delta[0]=0
        big_deltas = F.relu(dt - max_interval_norm)
        if big_deltas.numel() > 0:
            penalties.append(big_deltas.pow(2).mean())
    if penalties:
        big_delta_penalty = torch.stack(penalties).mean()

    return 2.0 * loss_under + 3.0 * big_delta_penalty


# ======================================================================
# TIME CONSISTENCY LOSS
# ======================================================================

def time_consistency_loss(
    visit_times_raw:   torch.Tensor,
    valid_flag:        torch.Tensor,
    lambda_undershoot: float = 2.0,
    lambda_overshoot:  float = 0.5,
) -> torch.Tensor:
    if valid_flag.dtype != torch.bool:
        valid_flag = valid_flag.bool()
    t_pos  = torch.arange(visit_times_raw.shape[1],
                          dtype=torch.float32, device=visit_times_raw.device)
    idx    = (valid_flag.float() * (t_pos.unsqueeze(0) + 1)).argmax(dim=1)
    t_last = visit_times_raw.gather(1, idx.unsqueeze(1)).squeeze(1)
    # Penalità quadratica per undershoot
    loss_under = F.relu(1.0 - t_last).pow(2)
    loss_over  = F.relu(t_last - 1.0)
    return lambda_undershoot * loss_under.mean() + lambda_overshoot * loss_over.mean()


# ======================================================================
# INTRA-PATIENT VARIANCE LOSS
# ======================================================================

def intra_patient_variance_loss(
    fake_cont:    torch.Tensor,
    real_cont:    torch.Tensor,
    fake_valid:   torch.Tensor,
    real_valid:   torch.Tensor,
    min_std_frac: float = 0.3,
) -> torch.Tensor:
    """
    Penalizza ogni paziente sintetico con std intra-paziente < min_std_frac * std_real.
    Fix principale per le traiettorie piatte.
    """
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)

    flat_penalties, dist_penalties = [], []

    for j in range(n_cont):
        f_stds, r_stds = [], []
        for b in range(fake_cont.shape[0]):
            fv = fake_cont[b, :, j][fake_valid[b]]
            rv = real_cont[b, :, j][real_valid[b]]
            if fv.numel() >= 2:
                f_stds.append(fv.std())
            if rv.numel() >= 2:
                r_stds.append(rv.std().detach())

        if not f_stds:
            continue

        f_stds_t = torch.stack(f_stds)
        threshold = (torch.stack(r_stds).mean().clamp(min=1e-6) * min_std_frac
                     if r_stds else
                     torch.tensor(min_std_frac * 0.1, device=fake_cont.device))

        flat_penalties.append(F.relu(threshold - f_stds_t).pow(2).mean())

        if r_stds:
            r_stds_t = torch.stack(r_stds)
            dist_penalties.append(
                (f_stds_t.mean() - r_stds_t.mean()).pow(2) +
                (f_stds_t.std().clamp(1e-6) - r_stds_t.std().clamp(1e-6)).pow(2)
            )

    loss = torch.tensor(0.0, device=fake_cont.device)
    if flat_penalties:
        loss = loss + torch.stack(flat_penalties).mean()
    if dist_penalties:
        loss = loss + 0.5 * torch.stack(dist_penalties).mean()
    return loss


# ======================================================================
# DELTA DISTRIBUTION LOSS
# ======================================================================

def delta_distribution_loss(
    deltas_raw:  torch.Tensor,
    real_times:  torch.Tensor,
    fake_valid:  torch.Tensor,
    real_valid:  torch.Tensor,
    quantiles:   Optional[List[float]] = None,
) -> torch.Tensor:
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    fake_iv_list, real_iv_list = [], []
    B = deltas_raw.shape[0]
    for b in range(B):
        f_idx = fake_valid[b].nonzero(as_tuple=True)[0]
        if len(f_idx) >= 2:
            fd = deltas_raw[b, f_idx[1:]].clamp(min=0.0)
            if fd.numel() > 0:
                fake_iv_list.append(fd)
        r_idx = real_valid[b].nonzero(as_tuple=True)[0]
        if len(r_idx) >= 2:
            rt = real_times[b, r_idx]
            rd = (rt[1:] - rt[:-1]).clamp(min=0.0)
            if rd.numel() > 0:
                real_iv_list.append(rd)

    if not fake_iv_list or not real_iv_list:
        return torch.tensor(0.0, device=deltas_raw.device)

    f_iv = torch.cat(fake_iv_list)
    r_iv = torch.cat(real_iv_list)
    if len(f_iv) < 2 or len(r_iv) < 2:
        return torch.tensor(0.0, device=deltas_raw.device)

    loss = (f_iv.mean() - r_iv.mean().detach()).pow(2)
    loss = loss + (f_iv.std().clamp(1e-6) - r_iv.std().clamp(1e-6).detach()).pow(2)
    qs   = torch.tensor(quantiles, device=f_iv.device)
    loss = loss + F.mse_loss(torch.quantile(f_iv, qs), torch.quantile(r_iv.detach(), qs))
    return loss


# ======================================================================
# VARIANCE LOSS (aggregata inter-paziente)
# ======================================================================

def variance_loss(fake_cont, real_cont, fake_valid, real_valid):
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)
    losses = []
    for j in range(n_cont):
        f_vals = fake_cont[:, :, j][fake_valid]
        r_vals = real_cont[:, :, j][real_valid]
        if len(f_vals) < 2 or len(r_vals) < 2:
            continue
        loss_j = (f_vals.std() - r_vals.std().detach()).pow(2)
        f_stds, r_stds = [], []
        for b in range(fake_cont.shape[0]):
            fv = fake_cont[b, :, j][fake_valid[b]]
            rv = real_cont[b, :, j][real_valid[b]]
            if fv.numel() > 1:
                f_stds.append(fv.std())
            if rv.numel() > 1:
                r_stds.append(rv.std().detach())
        if f_stds and r_stds:
            loss_j = loss_j + (torch.stack(f_stds).mean() -
                                torch.stack(r_stds).mean()).pow(2)
        losses.append(loss_j)
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=fake_cont.device)


# ======================================================================
# AUTOCORRELATION LOSS
# ======================================================================

def autocorrelation_loss(fake_cont, real_cont, fake_valid, real_valid, max_lag=2):
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)
    losses = []
    for lag in range(1, max_lag + 1):
        for j in range(n_cont):
            f_ac_list, r_ac_list = [], []
            for b in range(fake_cont.shape[0]):
                f_idx = fake_valid[b].nonzero(as_tuple=True)[0]
                if len(f_idx) > lag:
                    fx, fy = fake_cont[b, f_idx[:-lag], j], fake_cont[b, f_idx[lag:], j]
                    if fx.std() > 1e-6 and fy.std() > 1e-6:
                        f_ac_list.append(((fx - fx.mean()) * (fy - fy.mean())).mean() /
                                         (fx.std() * fy.std() + 1e-8))
                r_idx = real_valid[b].nonzero(as_tuple=True)[0]
                if len(r_idx) > lag:
                    rx, ry = real_cont[b, r_idx[:-lag], j], real_cont[b, r_idx[lag:], j]
                    if rx.std() > 1e-6 and ry.std() > 1e-6:
                        r_ac_list.append((((rx - rx.mean()) * (ry - ry.mean())).mean() /
                                          (rx.std() * ry.std() + 1e-8)).detach())
            if f_ac_list and r_ac_list:
                losses.append((torch.stack(f_ac_list).mean() -
                                torch.stack(r_ac_list).mean()).pow(2))
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=fake_cont.device)


# ======================================================================
# IRREVERSIBILITY LOSS
# ======================================================================

def irreversibility_loss(irr_states, valid_flag):
    if valid_flag.dim() == 3:
        valid_flag = valid_flag.squeeze(-1)
    vf       = valid_flag[:, 1:].float()
    diff     = irr_states[:, 1:] - irr_states[:, :-1]
    loss_mono = (torch.clamp(-diff, min=0) * vf).mean()
    vm       = valid_flag.unsqueeze(-1).float()
    h        = irr_states.clamp(1e-6, 1 - 1e-6)
    ent      = -(h * h.log() + (1 - h) * (1 - h).log())
    loss_ent = (ent * vm.squeeze(-1)).sum() / vm.sum().clamp(min=1.0)
    return loss_mono + 0.1 * loss_ent


# ======================================================================
# N_VISITS LOSS
# ======================================================================

def n_visits_loss(n_v_pred, n_v_real):
    pred    = n_v_pred.float()
    real    = n_v_real.float().to(pred.device)
    l_mean  = F.smooth_l1_loss(pred, real)
    l_var   = F.relu(real.var().clamp(min=0.0) - pred.var().clamp(min=0.0))
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    l_quant = F.mse_loss(torch.quantile(pred, qs), torch.quantile(real, qs))
    return l_mean + 0.2 * l_var + 1.5 * l_quant


# ======================================================================
# FOLLOWUP NORM LOSS
# ======================================================================

def followup_norm_loss(pred_followup, real_followup):
    pred    = pred_followup.float()
    real    = real_followup.float().to(pred.device)
    l_mean  = F.mse_loss(pred.mean(), real.mean())
    l_var   = F.relu(real.var().clamp(1e-6) - pred.var().clamp(1e-6))
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    l_quant = F.mse_loss(torch.quantile(pred, qs), torch.quantile(real, qs))
    return l_mean + 0.5 * l_var + 2.0 * l_quant


# ======================================================================
# STATIC CAT MARGINAL LOSS — con clamp KL per prevenire esplosione
# ======================================================================

def static_cat_marginal_loss(
    fake_static_cat_soft: Dict[str, torch.Tensor],
    target_probs:         Dict[str, torch.Tensor],
    eps:                  float = 1e-6,
    max_kl:               float = 10.0,   # clamp per stabilità
) -> torch.Tensor:
    """
    KL(target || fake_marginal) + CE.
    [FIX] Aggiunto clamp del KL a max_kl per prevenire esplosione quando
    il generatore collassa — era la causa del Scat crescente.
    """
    losses = []
    for name, p_soft in fake_static_cat_soft.items():
        if name not in target_probs:
            continue
        p_real = target_probs[name].to(p_soft.device).float()
        p_s    = p_soft.float()

        p_m  = p_s.mean(dim=0).clamp(min=eps)
        p_m  = p_m / p_m.sum()
        pr   = p_real.clamp(min=eps)
        kl   = (pr * (pr / p_m).log()).sum().clamp(max=max_kl)   # [FIX] clamp

        ce   = -(pr.unsqueeze(0) * p_s.clamp(min=eps).log()).sum(dim=-1).mean()
        min_p = p_real.min().clamp(min=1e-6)
        w     = (1.0 / min_p).clamp(max=20.0)
        losses.append(w * (kl + 0.5 * ce))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


# ======================================================================
# FEATURE MATCHING
# ======================================================================

def feature_matching_loss(features_real, features_fake):
    return F.mse_loss(features_fake.mean(dim=0), features_real.mean(dim=0))


# ======================================================================
# STATIC CONTINUOUS DISTRIBUTION LOSS
# ======================================================================

def static_cont_dist_loss(fake_sc, real_sc):
    fake    = fake_sc.float()
    real    = real_sc.float().to(fake.device)
    l_mean  = F.mse_loss(fake.mean(dim=0), real.mean(dim=0))
    l_var   = F.relu(real.var(dim=0).clamp(1e-6) - fake.var(dim=0).clamp(1e-6)).mean()
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=fake.device)
    l_quant = F.mse_loss(torch.quantile(fake, qs, dim=0), torch.quantile(real, qs, dim=0))
    return l_mean + 0.5 * l_var + l_quant


# ======================================================================
# CATEGORICAL FREQUENCY
# ======================================================================

def compute_category_weights(real_cat_dict, visit_mask, smoothing=1e-3, power=1.0):
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
        raw_w  = 1.0 / (counts / total + smoothing).pow(power)
        weights[var_name] = raw_w / raw_w.sum()
    return weights


def categorical_frequency_loss_generator(fake_cat_dict, cat_weights, visit_mask, eps=1e-4):
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    losses = []
    for var_name, fake_tensor in fake_cat_dict.items():
        if var_name not in cat_weights:
            continue
        w = cat_weights[var_name].to(fake_tensor.device)
        if fake_tensor.dim() == 3:
            vm        = visit_mask.unsqueeze(-1).float().to(fake_tensor.device)
            fake_dist = (fake_tensor.float() * vm).sum(dim=(0, 1)) / vm.sum(dim=(0, 1)).clamp(1)
        else:
            fake_dist = fake_tensor.float().mean(dim=0)
        fake_dist = fake_dist.clamp(min=0.0) / fake_dist.sum().clamp(1e-8)
        losses.append(-(w * (fake_dist + eps).log()).sum())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def categorical_frequency_loss_discriminator(real_cat_dict, fake_cat_dict, cat_weights, visit_mask):
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)
    losses = []
    for var_name in real_cat_dict:
        if var_name not in fake_cat_dict or var_name not in cat_weights:
            continue
        w = cat_weights[var_name].to(real_cat_dict[var_name].device)
        if real_cat_dict[var_name].dim() == 3:
            vm = visit_mask.unsqueeze(-1).float().to(real_cat_dict[var_name].device)
            n  = vm.sum(dim=(0, 1)).clamp(1)
            rd = (real_cat_dict[var_name].float() * vm).sum(dim=(0, 1)) / n
            fd = (fake_cat_dict[var_name].float() * vm).sum(dim=(0, 1)) / n
        else:
            rd = real_cat_dict[var_name].float().mean(dim=0)
            fd = fake_cat_dict[var_name].float().mean(dim=0)
        rd = rd.clamp(1e-8) / rd.sum()
        fd = fd.clamp(1e-8) / fd.sum()
        m  = (0.5 * (rd + fd)).clamp(1e-8)
        losses.append((w * (0.5 * F.kl_div(m.log(), rd, reduction="none") +
                            0.5 * F.kl_div(m.log(), fd, reduction="none"))).sum())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)


def inter_visit_interval_loss(visit_times, visit_mask, real_intervals):
    vm    = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask
    dt    = visit_times[:, 1:] * vm[:, 1:].float()
    valid = dt > 1e-8
    if valid.sum() < 10:
        return torch.tensor(0.0, device=visit_times.device)
    dt_synth = dt[valid]
    real_iv  = real_intervals.to(dt_synth.device)
    qs       = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dt_synth.device)
    return (F.mse_loss(dt_synth.mean(), real_iv.mean().clamp(1e-6)) +
            2.0 * F.mse_loss(torch.quantile(dt_synth, qs), torch.quantile(real_iv, qs)))


def inter_visit_uniformity_loss(delta_months, visit_mask, cv2_target=1.0):
    vm      = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask
    dm      = delta_months[:, 1:].float()
    vm_d    = vm[:, 1:].float()
    n_valid = vm_d.sum(dim=1).clamp(min=2.0)
    mean_dm = (dm * vm_d).sum(dim=1) / n_valid
    var_dm  = ((dm - mean_dm.unsqueeze(1)) ** 2 * vm_d).sum(dim=1) / n_valid
    cv2     = var_dm / (mean_dm ** 2 + 1e-6)
    return (F.relu(cv2 - cv2_target) * (vm_d.sum(dim=1) > 2).float()).mean()


# ======================================================================
# UTILITY
# ======================================================================

def check_finite(loss: torch.Tensor, name: str) -> torch.Tensor:
    import warnings
    if not torch.isfinite(loss):
        warnings.warn(f"Loss '{name}' NaN/Inf. Sostituita con 0.", UserWarning, stacklevel=2)
        return torch.zeros_like(loss)
    return loss