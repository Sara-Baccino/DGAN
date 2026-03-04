"""
utils/losses.py
================================================================================
Modifiche rispetto alla versione precedente:

  [v4] visit_length_loss RIMOSSA.
       Il generatore ora predice n_visits direttamente (n_visits_head).
       La loss era un workaround per il meccanismo cumprod indiretto.
       Con n_visits_head, la lunghezza è controllata esplicitamente
       e nessuna loss aggiuntiva è necessaria.

  [v3] categorical_frequency_loss_generator: CE pesata con eps=1e-4.
       (invariata rispetto a v2)

  [v3] irreversibility_loss: entropia hazard.
       (invariata rispetto a v2)

  [v3] categorical_frequency_loss_discriminator: JS pesata.
       (invariata rispetto a v2)
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

    Con cummax nel generatore, la loss ReLU(s[t-1]-s[t]) era strutturalmente
    0 (monotonicità garantita per costruzione). Questa loss agisce a monte:
    forza l'hazard a essere netto (0 o 1) invece di 0.5 diffuso ovunque.

    loss = mean[-h·log(h) - (1-h)·log(1-h)]  sugli step visitati

    Minimo (0): hazard ∈ {0,1} — segnali netti ✓
    Massimo (0.693): hazard = 0.5 — massima incertezza ✗
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
    """
    Calcola pesi inverso-frequenza per classe su tutto il training set.
    Chiamata UNA VOLTA prima del loop epoche per stime stabili.

    weight(c) = [1 / (freq(c) + smoothing)]^power  normalizzato.

    power=1.0 : aggressivo (frequenze > 1%)
    power=0.5 : conservativo (frequenze < 1%)
    """
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
    """
    Cross-entropy pesata sulla distribuzione aggregata del generatore.

    loss = -Σ_c [ w(c) · log(fake_dist(c) + eps) ]

    Monotona decrescente: quando il generatore migliora verso la
    distribuzione target, la loss scende. eps=1e-4 garantisce gradiente
    finito anche quando fake_dist→0 (evita il problema Fgen=18 costante
    che si aveva con KL + clamp(1e-8)).
    """
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
    """
    Jensen-Shannon pesata via F.kl_div (numericamente stabile).
    Amplifica le classi rare tramite cat_weights.
    """
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
# STATIC CATEGORICAL MARGINAL LOSS  [v6.1]
# ==================================================================

def static_cat_marginal_loss(
    fake_static_cat: Dict[str, torch.Tensor],
    target_probs:    Dict[str, torch.Tensor],
    eps:             float = 1e-4,
) -> torch.Tensor:
    """
    [v6.1] Cross-entropy pesata tra distribuzione marginale generata
    e distribuzione marginale target (calcolata sul training set reale).

    PROBLEMA CHE RISOLVE:
      Le variabili statiche categoriche (SEX, ETHNICC, ALCOHOL, SMOKING,
      INRPT, WDRAWYN, DEATHRYN, ecc.) collassano al 100% sulla classe
      maggioritaria (Cramér's V = 1.0 su 11 variabili).

      La categorical_frequency_loss_generator agisce solo sulle temporali.
      Le static_cat_heads producono logit → gumbel_softmax → one-hot.
      Il discriminatore statico vede solo uno scalare → gradiente troppo
      debole per imporre distribuzioni marginali corrette su 20+ variabili.

    SOLUZIONE:
      Supervisione diretta: -Σ_c p_real(c) * log(p_fake(c) + eps)
      p_fake(c) = media delle one-hot generate nel batch (approssima
      la distribuzione marginale con varianza 1/sqrt(B))

      La target_probs viene calcolata UNA VOLTA dal training set intero
      (distribuzione marginale esatta) e salvata come buffer in DGAN.

    EFFETTO ATTESO:
      Con lambda_static_cat = 2.0-4.0, la loss spinge i logit di ogni
      head verso log(p_real), che equivale a inizializzarli correttamente
      E mantenerli ancorati durante il training.

    ARGOMENTI:
      fake_static_cat: {nome: one-hot [B, n_cat]} dal forward() del generatore
      target_probs:    {nome: [n_cat]} distribuzione marginale reale, normalizzata
      eps:             smoothing per log-stabilità

    NOTA: non è compatibile con le variabili in embedding_configs (CENTRE),
      che vengono gestite da auxiliary_loss nel discriminatore.
    """
    losses = []
    for name, fake_ohe in fake_static_cat.items():
        if name not in target_probs:
            continue
        p_real = target_probs[name].to(fake_ohe.device)    # [n_cat]
        p_fake = fake_ohe.float().mean(dim=0)               # [n_cat], media batch
        p_fake = p_fake.clamp(min=0.0)
        p_fake = p_fake / p_fake.sum().clamp(min=1e-8)
        ce     = -(p_real * (p_fake + eps).log()).sum()
        losses.append(ce)

    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()


def followup_norm_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
) -> torch.Tensor:
    """
    [v5] MSE tra followup_norm predetto dal generatore e quello reale.

    SCOPO:
      Vincola la durata assoluta del follow-up sintetico a replicare
      la distribuzione empirica del batch reale.
      real_followup è campionato con shuffle (nessuna corrispondenza
      paziente-per-paziente), quindi supervisiona la distribuzione
      marginale, non il singolo paziente.

    SCHEMA IN _train_generator:
      real_fup = batch["followup_norm"].to(device)          # [B] reale
      fake_fup = fake_batch["followup_norm"]                # [B] predetto
      L_fup = followup_norm_loss(fake_fup, real_fup)

    NOTA: con real_followup_norm passato al forward() come conditioning
    il generatore usa il valore reale (non predetto). In quel caso
    followup_head è supervisionato da questa loss in modo che in
    inference (senza conditioning) produca valori corretti.

    lambda suggerito: 0.5–1.0
    """
    return F.mse_loss(pred_followup, real_followup.to(pred_followup.device))


# ==================================================================
# FEATURE MATCHING LOSS
# ==================================================================

def feature_matching_loss(
    features_real: torch.Tensor,
    features_fake: torch.Tensor,
) -> torch.Tensor:
    """
    [v5] MSE tra le feature interne medie del discriminatore statico.

    SCOPO:
      Il WGAN score è uno scalare → gradiente collo di bottiglia per
      30+ variabili categoriche. Questa loss dà segnale denso:
      ogni dimensione dello spazio interno del discriminatore che
      differisce tra reale e sintetico contribuisce direttamente
      al gradiente del generatore.

    EFFETTO ATTESO:
      - SEX, ETHNICC, ALCOHOL, SMOKING convergono più velocemente
      - Il generatore non può ignorare le distribuzioni categoriche
        puntando tutto sulla loss WGAN continua

    USO IN dgan._train_generator:
      feat_real = disc_s.get_features(static_real).detach()  # [B, H]
      feat_fake = disc_s.get_features(static_fake)           # [B, H]
      L_fm = feature_matching_loss(feat_real, feat_fake)
      loss_g += lambda_fm * L_fm

    CRITICO: .detach() sui reali è OBBLIGATORIO.
      Senza .detach(): il gradiente fluisce anche nel discriminatore
      durante l'aggiornamento del generatore → training instabile.
      Con .detach(): solo il generatore si aggiorna.

    Usa medie di batch (non sample-per-sample) per matchare la
    distribuzione, non copiare i singoli campioni reali.

    lambda suggerito: 2.0–5.0
    """
    return F.mse_loss(features_fake.mean(dim=0), features_real.mean(dim=0))


# ==================================================================
# N_VISITS SUPERVISION LOSS  [v5.1]
# ==================================================================

def n_visits_supervision_loss(
    n_v_pred: torch.Tensor,
    n_v_real: torch.Tensor,
) -> torch.Tensor:
    """
    [v6.2] Loss composta per n_visits_head: SmoothL1 sulla media + penalità varianza.

    PROBLEMA DEL COLLASSO SULLA MEDIA:
      SmoothL1(pred, real) con real campionato shuffled converge verso E[real].
      Il generatore impara a predire sempre la media (~6-7) indipendentemente da z_static.
      Risultato: tutte le sequenze sintetiche hanno la stessa lunghezza.

    SOLUZIONE:
      L_total = SmoothL1(pred, real) + lambda_var * max(0, var_real - var_pred)

      Il termine var penalizza se la varianza predetta è INFERIORE a quella reale.
      Forza n_visits_head a produrre output dispersi su tutto il range [2, T_max].

    lambda_var = 0.1: leggero, evita instabilità. La SmoothL1 rimane dominante.
    """
    l_mean = F.smooth_l1_loss(
        n_v_pred.float(),
        n_v_real.float().to(n_v_pred.device),
    )
    var_real = n_v_real.float().var().clamp(min=0.0)
    var_pred = n_v_pred.float().var().clamp(min=0.0)
    l_var    = F.relu(var_real - var_pred)   # penalizza solo se var_pred < var_real

    return l_mean + 0.1 * l_var


# ==================================================================
# FOLLOWUP SEQUENCE LENGTH LOSS  [v5.1]
# ==================================================================

def followup_constraint_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
    visit_times:   torch.Tensor,
    visit_mask:    torch.Tensor,
) -> torch.Tensor:
    """
    [v5.1] Vincolo composito sulla lunghezza della sequenza.

    Combina due contributi:

    1. followup_norm MSE: il generatore deve predire la durata corretta
       del follow-up (già presente come followup_loss in losses.py).
       L_fn = MSE(pred_followup, real_followup)

    2. Sequence length consistency: l'ultimo step valido deve essere
       a t_norm ≈ 1.0 (sfrutta l'intera durata prevista).
       L_sl = MSE(t_norm_last_valid, 1.0)

       MOTIVAZIONE: anche se n_visits è fissato in training, il TimeEncoder
       potrebbe generare visite molto ravvicinate tutte all'inizio
       (t_norm_last ≈ 0.1), lasciando il 90% del follow-up inutilizzato.
       Questo vincolo forza la sequenza ad espandersi sull'intero range.

    L_total = L_fn + lambda_sl * L_sl

    lambda_sl suggerito: 0.1–0.3 (leggero, non deve dominare)
    lambda (esterno, per la loss totale): 0.5

    USO IN dgan._train_generator:
      L_fc = followup_constraint_loss(
          gen_out["followup_norm"], real_fn_batch,
          gen_out["visit_times"], gen_out["visit_mask"]
      )
      loss_g += lambda_fc * L_fc
    """
    lambda_sl = 0.2

    # 1. followup_norm MSE
    L_fn = F.mse_loss(
        pred_followup.float(),
        real_followup.float().to(pred_followup.device),
    )

    # 2. t_norm dell'ultimo step valido deve essere ≈ 1.0
    vm = visit_mask.squeeze(-1)    # [B, T]
    B  = vm.shape[0]

    # Indice dell'ultimo step valido per ogni paziente
    last_valid_idx = (vm * torch.arange(
        vm.shape[1], dtype=torch.float32, device=vm.device
    ).unsqueeze(0)).argmax(dim=1)  # [B]

    t_last = visit_times[
        torch.arange(B, device=vm.device), last_valid_idx
    ]  # [B] t_norm all'ultimo step valido

    # Pazienti con 1 sola visita: t_last = 0 (forzato da TimeEncoder)
    # Non penalizziamo se n_v = 1 (mask_valid = 0 se single visit)
    has_multiple = (vm.sum(dim=1) > 1).float()  # [B]
    L_sl = ((t_last - 1.0) ** 2 * has_multiple).mean()

    return L_fn + lambda_sl * L_sl