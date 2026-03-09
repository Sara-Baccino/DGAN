"""
utils/losses.py
================================================================================
Modifiche rispetto a v7-fix2:

  [v8] coverage_loss: normalizzazione relativa per-paziente.
       PROBLEMA v7: divisione per gtm² = 160000 rendeva il gradiente quasi nullo.
       SOLUZIONE: divide per d3_fup_m² individuale → loss ∈ [0,1] per costruzione.

  [v8] inter_visit_interval_loss: aggiunto termine l_mean_sq (errore relativo
       sulla media) per rafforzare il segnale quando la media sintetica è molto
       lontana da quella reale (3.8 vs 10.7 mesi).

  [v8] static_cat_marginal_loss: CE sostituita con Focal CE (γ=2) per amplificare
       il gradiente sulle classi rare (SEX, ETHNICC, ecc.) che il generatore ignora.

  [v8] uniformity_loss: nuova loss che penalizza la varianza normalizzata
       (coefficient of variation²) degli inter-visit intervals per-paziente.
       Impedisce che il TimeEncoder generi visite "bunched" tutte vicine a t=0
       con un salto finale verso D3_fup.
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
# IRREVERSIBILITY — HAZARD ENTROPY LOSS
# ======================================================================

def irreversibility_loss(
    irr_states: torch.Tensor,
    visit_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Penalizza hazard intermedi (~0.5) per le variabili irreversibili.
    """
    if visit_mask.dim() == 3:
        visit_mask = visit_mask.squeeze(-1)

    vm  = visit_mask.unsqueeze(-1).float()
    h   = irr_states.clamp(1e-6, 1 - 1e-6)
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

        real_dist = real_dist.clamp(min=1e-8); real_dist = real_dist / real_dist.sum()
        fake_dist = fake_dist.clamp(min=1e-8); fake_dist = fake_dist / fake_dist.sum()
        m         = (0.5 * (real_dist + fake_dist)).clamp(min=1e-8)
        kl_real   = F.kl_div(m.log(), real_dist, reduction="none")
        kl_fake   = F.kl_div(m.log(), fake_dist, reduction="none")
        js        = (w * (0.5 * kl_real + 0.5 * kl_fake)).sum()
        losses.append(js)

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


# ==================================================================
# STATIC CATEGORICAL MARGINAL LOSS  [v8]
# ==================================================================

def static_cat_marginal_loss(
    fake_static_cat:      Dict[str, torch.Tensor],
    fake_static_cat_soft: Dict[str, torch.Tensor],
    target_probs:         Dict[str, torch.Tensor],
    eps:                  float = 1e-6,
) -> torch.Tensor:
    """
    [v8] KL(real || fake) + Focal CE (γ=2) per le variabili categoriche statiche.

    NOVITÀ v8 rispetto a v6.5:
      La plain CE campione-per-campione non genera gradiente sufficiente per le
      classi rare quando il generatore le ignora quasi completamente (p_fake≈0.001).
      La Focal CE con γ=2 moltiplica la loss per (1-p_t)² dove p_t è la probabilità
      assegnata alla classe target: quando il modello sbaglia molto (p_t→0),
      (1-p_t)²→1 e la loss è piena; quando sbaglia poco (p_t→1), la loss è soppressa.
      Questo forza il gradiente a concentrarsi sui campioni dove il generatore
      collassa sulla classe maggioritaria (p_t≈0 per la classe minoritaria).

    ARGOMENTI:
      fake_static_cat:      {nome: one-hot hard [B, n_cat]}
      fake_static_cat_soft: {nome: soft prob    [B, n_cat]}
      target_probs:         {nome: [n_cat]} distribuzione marginale reale
      eps:                  smoothing (1e-6)
    """
    losses = []
    for name, fake_ohe in fake_static_cat.items():
        if name not in target_probs:
            continue
        p_real = target_probs[name].to(fake_ohe.device)   # [n_cat]

        # ── Termine 1: KL(real || fake_hard) ────────────────────────────
        p_fake_hard = fake_ohe.float().mean(dim=0).clamp(min=0.0)
        p_fake_hard = p_fake_hard / p_fake_hard.sum().clamp(min=1e-8)
        p_real_safe = p_real.clamp(min=eps)
        kl = (p_real_safe * (p_real_safe / (p_fake_hard + eps)).log()).sum()

        # ── Termine 2: Focal CE campione-per-campione (γ=2) [v8] ────────
        focal_loss = torch.tensor(0.0, device=fake_ohe.device)
        if name in fake_static_cat_soft:
            p_soft   = fake_static_cat_soft[name].float()         # [B, n_cat]
            log_soft = torch.log(p_soft.clamp(min=eps))           # [B, n_cat]

            # p_t: probabilità media pesata per target_probs (quanto il generatore
            # assegna correttamente alle classi nella giusta proporzione)
            # shape: [B] — probabilità "corretta" per ogni campione
            p_t = (p_real.unsqueeze(0) * p_soft).sum(dim=-1).clamp(min=eps)  # [B]

            # Focal weight: amplifica i campioni dove il generatore è più sbagliato
            focal_weight = (1.0 - p_t) ** 2   # [B]

            # CE pesata per target_probs (come KL ma campione per campione)
            ce_per_sample = -(p_real.unsqueeze(0) * log_soft).sum(dim=-1)  # [B]

            focal_loss = (focal_weight * ce_per_sample).mean()

        # ── Peso imbalance: variabili binarie sbilanciate ────────────────
        min_p            = p_real.min().clamp(min=1e-6)
        imbalance_weight = (1.0 / min_p).clamp(max=20.0)

        losses.append(imbalance_weight * (kl + 1.5 * focal_loss))

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def followup_norm_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
) -> torch.Tensor:
    """
    [v6.4] Loss di distribuzione per followup_norm. Invariata.
    """
    pred = pred_followup.float()
    real = real_followup.float().to(pred.device)

    l_mean = F.mse_loss(pred.mean(), real.mean())

    var_real = real.var().clamp(min=1e-6)
    var_pred = pred.var().clamp(min=1e-6)
    l_var    = F.relu(var_real - var_pred)

    qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    q_real = torch.quantile(real, qs)
    q_pred = torch.quantile(pred, qs)
    l_quant = F.mse_loss(q_pred, q_real)

    return l_mean + 0.5 * l_var + 2.0 * l_quant


# ==================================================================
# FEATURE MATCHING LOSS
# ==================================================================

def feature_matching_loss(
    features_real: torch.Tensor,
    features_fake: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(features_fake.mean(dim=0), features_real.mean(dim=0))


# ==================================================================
# N_VISITS SUPERVISION LOSS
# ==================================================================

def n_visits_supervision_loss(
    n_v_pred: torch.Tensor,
    n_v_real: torch.Tensor,
) -> torch.Tensor:
    pred  = n_v_pred.float()
    real  = n_v_real.float().to(pred.device)

    l_mean = F.smooth_l1_loss(pred, real)

    var_real = real.var().clamp(min=0.0)
    var_pred = pred.var().clamp(min=0.0)
    l_var    = F.relu(var_real - var_pred)

    qs     = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    q_real = torch.quantile(real, qs)
    q_pred = torch.quantile(pred, qs)
    l_quant = F.mse_loss(q_pred, q_real)

    return l_mean + 0.1 * l_var + 1.0 * l_quant


# ==================================================================
# FOLLOWUP CONSTRAINT LOSS
# ==================================================================

def followup_constraint_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
    visit_times:   torch.Tensor,
    visit_mask:    torch.Tensor,
) -> torch.Tensor:
    """
    [v5.1] Vincolo composito sulla lunghezza della sequenza. Invariato.
    """
    lambda_sl = 0.2

    L_fn = F.mse_loss(
        pred_followup.float(),
        real_followup.float().to(pred_followup.device),
    )

    vm = visit_mask.squeeze(-1)
    B  = vm.shape[0]

    last_valid_idx = (vm * torch.arange(
        vm.shape[1], dtype=torch.float32, device=vm.device
    ).unsqueeze(0)).argmax(dim=1)

    t_last = visit_times[
        torch.arange(B, device=vm.device), last_valid_idx
    ]

    has_multiple = (vm.sum(dim=1) > 1).float()
    L_sl = ((t_last - 1.0) ** 2 * has_multiple).mean()

    return L_fn + lambda_sl * L_sl


# ==================================================================
# STATIC CONTINUOUS DISTRIBUTION LOSS
# ==================================================================

def static_cont_dist_loss(
    fake_sc: torch.Tensor,
    real_sc: torch.Tensor,
) -> torch.Tensor:
    fake = fake_sc.float()
    real = real_sc.float().to(fake.device)

    l_mean = F.mse_loss(fake.mean(dim=0), real.mean(dim=0))

    var_real = real.var(dim=0).clamp(min=1e-6)
    var_pred = fake.var(dim=0).clamp(min=1e-6)
    l_var    = F.relu(var_real - var_pred).mean()

    qs     = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=fake.device)
    q_real = torch.quantile(real, qs, dim=0)
    q_pred = torch.quantile(fake, qs, dim=0)
    l_quant = F.mse_loss(q_pred, q_real)

    return 1.0 * l_mean + 0.5 * l_var + 1.0 * l_quant


# ==================================================================
# INTER-VISIT INTERVAL DISTRIBUTION LOSS  [v8]
# ==================================================================

def inter_visit_interval_loss(
    visit_times:    torch.Tensor,   # [B, T] delta_months dal Generator
    visit_mask:     torch.Tensor,   # [B, T] o [B, T, 1]
    real_intervals: torch.Tensor,   # [N_real_intervals] in mesi, precalcolati
) -> torch.Tensor:
    """
    [v8] Supervisiona la distribuzione degli inter-visit intervals sintetici.

    NOVITÀ v8:
      Aggiunto termine l_mean_sq = ((mean_synth - mean_real) / mean_real)²
      che misura l'errore RELATIVO sulla media. Con mean_synth=3.8 e mean_real=10.7:
        l_mean_sq = ((3.8-10.7)/10.7)² ≈ 0.416
      Molto più forte di l_mean assoluta = (3.8-10.7)² = 47.6 diviso per nulla.
      Il termine relativo è naturalmente scalato in [0, ∞) con 0 = match perfetto
      e non dipende dalla scala assoluta degli intervalli.
    """
    vm = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask  # [B, T]

    dt = (visit_times[:, 1:] - visit_times[:, :-1])
    dt = dt * vm[:, 1:]

    valid = dt > 1e-8
    if valid.sum() < 10:
        return torch.tensor(0.0, device=visit_times.device)

    dt_synth = dt[valid]
    real_iv  = real_intervals.to(dt_synth.device)

    # 1. Errore relativo sulla media [v8] — invariante alla scala
    mean_real = real_iv.mean().clamp(min=1e-6)
    l_mean_sq = ((dt_synth.mean() - mean_real) / mean_real) ** 2

    # 2. Errore assoluto sulla media (segnale supplementare)
    l_mean = F.mse_loss(dt_synth.mean(), mean_real)

    # 3. Quantile matching [0.1, 0.25, 0.5, 0.75, 0.9]
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=dt_synth.device)
    q_synth = torch.quantile(dt_synth, qs)
    q_real  = torch.quantile(real_iv,  qs)
    # Quantile matching relativo: normalizza per i quantili reali
    q_real_safe = q_real.clamp(min=1e-6)
    l_quant = F.mse_loss(q_synth / q_real_safe, torch.ones_like(q_synth))

    return 3.0 * l_mean_sq + l_mean + 1.5 * l_quant


# ==================================================================
# UNIFORMITY LOSS  [v8 — NUOVA]
# ==================================================================

def inter_visit_uniformity_loss(
    delta_months: torch.Tensor,   # [B, T] delta in mesi dal TimeEncoder
    visit_mask:   torch.Tensor,   # [B, T] o [B, T, 1]
) -> torch.Tensor:
    """
    [v8] Penalizza la non-uniformità degli inter-visit intervals per-paziente.

    PROBLEMA:
      Il TimeEncoder può generare distribuzioni di delta molto sbilanciate:
      es. [0, 0.5, 0.3, 0.2, 0.1, 48.0] mesi — tutte le visite ravvicinate
      a t=0 e poi un salto finale verso D3_fup. Questo soddisfa coverage_loss
      (l'ultima visita ≈ D3_fup) ma produce inter-visit intervals irrealistici.

    SOLUZIONE:
      Penalizza il Coefficient of Variation² (CV²) degli intervalli per-paziente.
      CV² = Var(delta) / Mean(delta)² — misura la dispersione relativa.
      CV²=0: tutti gli intervalli uguali (uniforme, desiderabile per PBC ~12 mesi).
      CV²>>0: un intervallo domina tutti gli altri (bunching, indesiderabile).

      Per PBC con follow-up annuale, ci aspettiamo CV ≈ 0.5-1.0 (qualche
      variabilità ma non estrema). Penalizziamo CV² > cv2_target.

    NOTA: viene penalizzato solo CV² > target, non CV² < target, per non
    forzare uniformità perfetta (che non esiste nei dati reali).

    lambda suggerito: 3.0–5.0
    cv2_target: 1.0 (CV≈1, distribuzione esponenziale — ragionevole per PBC)
    """
    vm = visit_mask.squeeze(-1) if visit_mask.dim() == 3 else visit_mask  # [B, T]

    # Prendi i delta dall'indice 1 in poi (indice 0 = 0 per costruzione)
    dm   = delta_months[:, 1:].float()    # [B, T-1]
    vm_d = vm[:, 1:].float()             # [B, T-1]

    n_valid = vm_d.sum(dim=1).clamp(min=2.0)   # almeno 2 visite per calcolare var

    # Media per-paziente degli intervalli validi
    mean_dm = (dm * vm_d).sum(dim=1) / n_valid                   # [B]

    # Varianza per-paziente
    diff_sq = ((dm - mean_dm.unsqueeze(1)) ** 2) * vm_d          # [B, T-1]
    var_dm  = diff_sq.sum(dim=1) / n_valid                       # [B]

    # CV² = var / mean² — penalizza solo se > target
    cv2_target = 1.0
    cv2 = var_dm / (mean_dm ** 2 + 1e-6)                         # [B]

    # Penalizza solo i pazienti con più di 2 visite (altrimenti CV² non ha senso)
    has_multi = (vm_d.sum(dim=1) > 2).float()
    penalty   = F.relu(cv2 - cv2_target) * has_multi             # [B]

    return penalty.mean()