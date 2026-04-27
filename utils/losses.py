"""
model/losses.py  [gretel-style v3]
================================================================================
Solo le loss necessarie per DGAN:

  WGAN
    wgan_d_loss          — discriminatore: max E[real] - E[fake]
    wgan_g_loss          — generatore: min -E[fake_s] - E[fake_t]
    gradient_penalty     — penalità Lipschitz su interpolazione real/fake

  DISTRIBUZIONE TEMPORALE
    delta_distribution_loss     — media/std/quantili degli intervalli inter-visita
    intra_patient_variance_loss — std intra-paziente per feature continue
    autocorrelation_loss        — autocorrelazione lag-1..k per feature continue

  SUPERVISIONE SCALARI GLOBALI
    followup_norm_loss   — media/std/quantili del follow-up normalizzato
    n_visits_loss        — media/std/quantili del numero di visite

  STRUTTURA CATEGORICA
    static_cat_marginal_loss    — KL(target || fake_marginal) per categoriche statiche
    irreversibility_loss        — penalizza transizioni 1→0 per stati irreversibili

  FEATURE MATCHING
    feature_matching_loss       — MSE tra feature medie del disc_static

  UTILITÀ
    check_finite         — sostituisce NaN/Inf con 0 + warning

================================================================================
Rimosse rispetto alla versione precedente:
  - time_budget_loss          (duplicata con delta_distribution_loss)
  - time_consistency_loss     (gestita internamente dal generatore via ratio)
  - cumsum_constraint_loss    (idem)
  - gate_regularization_loss  (non usata)
  - delta_min_spacing_loss    (coperta da delta_distribution_loss quantili)
  - categorical_frequency_loss_* (rimpiazzate da static_cat_marginal_loss)
  - inter_visit_interval_loss / inter_visit_uniformity_loss (unificate)
  - static_cont_dist_loss     (non usata nel training loop)
================================================================================
"""

import warnings
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


# ======================================================================
# WGAN
# ======================================================================

def wgan_d_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Wasserstein discriminator loss: massimizza E[real] - E[fake]."""
    return -(d_real.mean() - d_fake.mean())


def wgan_g_loss(
    d_fake_static:   torch.Tensor,
    d_fake_temporal: torch.Tensor,
) -> torch.Tensor:
    """Wasserstein generator loss: minimizza -E[fake_s] - E[fake_t]."""
    return -(d_fake_static.mean() + d_fake_temporal.mean())


def gradient_penalty(
    critic_fn,
    real:      torch.Tensor,
    fake:      torch.Tensor,
    device,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """
    WGAN-GP: penalità sul gradiente dell'interpolazione real/fake.
    Mantiene il critico 1-Lipschitz senza weight clipping.
    """
    B         = real.size(0)
    eps_shape = (B,) + (1,) * (real.dim() - 1)
    eps       = torch.rand(eps_shape, device=device)
    interp    = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp  = critic_fn(interp)
    grads     = torch.autograd.grad(
        outputs        = d_interp,
        inputs         = interp,
        grad_outputs   = torch.ones_like(d_interp),
        create_graph   = True,
        retain_graph   = True,
        only_inputs    = True,
    )[0]
    grads_flat = grads.reshape(B, -1)
    return lambda_gp * ((grads_flat.norm(2, dim=1) - 1) ** 2).mean()


# ======================================================================
# DISTRIBUZIONE INTERVALLI INTER-VISITA
# ======================================================================

def delta_distribution_loss(
    deltas_raw:  torch.Tensor,               # [B, T]  Δt generati
    real_times:  torch.Tensor,               # [B, T]  visit_times reali (norm.)
    fake_valid:  torch.Tensor,               # [B, T]  bool
    real_valid:  torch.Tensor,               # [B, T]  bool
    quantiles:   Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Penalizza la differenza di distribuzione degli intervalli inter-visita.

    Per i sintetici usa i deltas espliciti del generatore (già normalizzati).
    Per i reali calcola Δt = visit_time[t] - visit_time[t-1].

    Confronta: media, std, e quantili [0.1, 0.25, 0.5, 0.75, 0.9].
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    device = deltas_raw.device

    # --- FAKE DELTAS ---
    # Vogliamo tutti i delta dove fake_valid è True, MA saltando il primo True di ogni riga
    # (perché il primo delta è solitamente 0 o l'origine).
    cum_valid_f = torch.cumsum(fake_valid.long(), dim=1)
    f_mask = fake_valid & (cum_valid_f > 1)
    f_iv = deltas_raw[f_mask].clamp(min=0.0)

    # --- REAL DELTAS ---
    # Δt = t[i] - t[i-1]
    r_diffs = real_times[:, 1:] - real_times[:, :-1]
    # Valido solo se sia lo step attuale che il precedente erano validi
    r_mask = real_valid[:, 1:] & real_valid[:, :-1]
    r_iv = r_diffs[r_mask].clamp(min=0.0)

    if f_iv.numel() < 2 or r_iv.numel() < 2:
        return torch.tensor(0.0, device=device)

    # Statistiche (r_iv.detach() perché i dati reali sono costanti per il generatore)
    r_mean = r_iv.mean().detach()
    r_std  = r_iv.std().clamp(1e-6).detach()
    r_qs   = torch.quantile(r_iv.detach(), torch.tensor(quantiles, device=device))

    loss = (f_iv.mean() - r_mean).pow(2)
    loss += (f_iv.std().clamp(1e-6) - r_std).pow(2)

    # Penalità asimmetrica per il collasso temporale
    if f_iv.mean() < r_mean * 0.5:
        loss += ((r_mean - f_iv.mean()) / (r_mean + 1e-8)).pow(2)

    # MSE sui quantili
    f_qs = torch.quantile(f_iv, torch.tensor(quantiles, device=device))
    loss += F.mse_loss(f_qs, r_qs)

    return loss


# ======================================================================
# VARIANZA INTRA-PAZIENTE
# ======================================================================

def intra_patient_variance_loss(
    fake_cont:    torch.Tensor,   # [B, T, n_cont]
    real_cont:    torch.Tensor,   # [B, T, n_cont]
    fake_valid:   torch.Tensor,   # [B, T] bool
    real_valid:   torch.Tensor,   # [B, T] bool
    min_std_frac: float = 0.3,
) -> torch.Tensor:
    """
    Penalizza ogni paziente sintetico con std intra-paziente
    < min_std_frac * std_reale medio.

    Forza il generatore a produrre traiettorie con variabilità realistica
    invece di valori piatti (problema comune nei GAN su serie temporali cliniche).

    Componenti:
      1. flat_penalty:  relu(threshold - f_std)^2  per ogni paziente/feature
      2. dist_penalty:  (mean_f_std - mean_r_std)^2  distribuzione delle std
    """
    n_cont = fake_cont.shape[-1]
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)

    # Trasformiamo le maschere in float [B, T, 1]
    f_mask = fake_valid.unsqueeze(-1).float()
    r_mask = real_valid.unsqueeze(-1).float()

    def get_patient_stds(data, mask):
        # Conta quanti step validi ha ogni paziente [B, 1, n_cont]
        counts = mask.sum(dim=1, keepdim=True).clamp(min=2) # serve almeno 2 per la std
        
        # Media per paziente
        means = (data * mask).sum(dim=1, keepdim=True) / counts
        
        # Varianza per paziente: sum((x - mu)^2) / (n - 1)
        # Usiamo .clamp(min=0) per evitare errori numerici dovuti a precisione float
        vars = ((data - means) * mask).pow(2).sum(dim=1, keepdim=True) / (counts - 1)
        
        # Deviazione standard [B, n_cont]
        # Epsilon dentro la radice per stabilità gradienti
        return torch.sqrt(vars.squeeze(1) + 1e-8)

    # Calcoliamo le std intra-paziente per tutto il batch
    f_stds = get_patient_stds(fake_cont, f_mask)     # [B, n_cont]
    r_stds = get_patient_stds(real_cont, r_mask).detach() # [B, n_cont]

    # 1. FLAT PENALTY
    # Soglia basata sulla media delle std reali per ogni feature
    thresholds = r_stds.mean(dim=0, keepdim=True) * min_std_frac # [1, n_cont]
    # Penalizza solo dove f_stds < threshold
    flat_penalty = F.relu(thresholds - f_stds).pow(2).mean()

    # 2. DIST PENALTY (Matching della distribuzione delle deviazioni standard)
    # Media delle std
    mean_f_std = f_stds.mean(dim=0)
    mean_r_std = r_stds.mean(dim=0)
    
    # Variabilità delle std tra pazienti (std della std)
    # Serve a far sì che non tutti i pazienti abbiano la stessa varianza
    std_f_std = f_stds.std(dim=0).clamp(min=1e-6)
    std_r_std = r_stds.std(dim=0).clamp(min=1e-6)

    dist_penalty = (mean_f_std - mean_r_std).pow(2).mean() + \
                   (std_f_std - std_r_std).pow(2).mean()

    return flat_penalty + 0.5 * dist_penalty


# ======================================================================
# AUTOCORRELAZIONE
# ======================================================================

def autocorrelation_loss(
    fake_cont:   torch.Tensor,   # [B, T, n_cont]
    real_cont:   torch.Tensor,   # [B, T, n_cont]
    fake_valid:  torch.Tensor,   # [B, T] bool
    real_valid:  torch.Tensor,   # [B, T] bool
    max_lag:     int = 2,
) -> torch.Tensor:
    """
    Penalizza la differenza di struttura di autocorrelazione lag-1..max_lag.

    Per ogni feature continua e ogni lag, calcola la correlazione di Pearson
    tra x[t] e x[t+lag] solo sugli step validi. Spinge il generatore a
    riprodurre la smoothness e la persistenza temporale dei dati reali.
    """
    batch_size, T, n_cont = fake_cont.shape
    if n_cont == 0:
        return torch.tensor(0.0, device=fake_cont.device)

    # Trasformiamo le maschere in float e aggiungiamo dimensione feature [B, T, 1]
    f_mask = fake_valid.unsqueeze(-1).float()
    r_mask = real_valid.unsqueeze(-1).float()

    # Applichiamo la maschera subito per sicurezza
    fake_cont = fake_cont * f_mask
    real_cont = real_cont * r_mask

    total_loss = []

    for lag in range(1, max_lag + 1):
        # Definiamo i segmenti temporali: x è [t : end-lag], y è [lag : end]
        # fake
        f_x = fake_cont[:, :-lag, :]
        f_y = fake_cont[:, lag:, :]
        f_m = f_mask[:, :-lag, :] * f_mask[:, lag:, :] # valida solo se entrambi gli step sono validi

        # real
        r_x = real_cont[:, :-lag, :]
        r_y = real_cont[:, lag:, :]
        r_m = r_mask[:, :-lag, :] * r_mask[:, lag:, :]

        # Calcolo medie pesate dalla maschera (mean = sum / count)
        def get_moments(x, y, mask):
            # count: numero di coppie (t, t+lag) valide nel batch per ogni feature
            count = mask.sum(dim=(0, 1), keepdim=True).clamp(min=1)
            
            mu_x = (x * mask).sum(dim=(0, 1), keepdim=True) / count
            mu_y = (y * mask).sum(dim=(0, 1), keepdim=True) / count
            
            # Centratura
            x_c = (x - mu_x) * mask
            y_c = (y - mu_y) * mask
            
            # Covarianza e Varianza
            cov = (x_c * y_c).sum(dim=(0, 1)) / count.squeeze()
            var_x = (x_c**2).sum(dim=(0, 1)) / count.squeeze()
            var_y = (y_c**2).sum(dim=(0, 1)) / count.squeeze()
            
            # --- STABILITÀ NUMERICA ---
            # Invece di sqrt(var_x * var_y) + eps, facciamo sqrt(var_x * var_y + eps)
            # L'epsilon dentro la radice evita che la derivata esploda a zero.
            denom = torch.sqrt(var_x * var_y + 1e-8)
            
            corr = cov / denom
            return corr

        f_corr = get_moments(f_x, f_y, f_m)
        r_corr = get_moments(r_x, r_y, r_m).detach() # I dati reali non generano gradienti

        # MSE tra le correlazioni medie del batch per ogni feature
        total_loss.append(torch.mean((f_corr - r_corr)**2))

    return torch.stack(total_loss).mean()


# ======================================================================
# SUPERVISIONE SCALARI GLOBALI
# ======================================================================

def followup_norm_loss(
    pred_followup: torch.Tensor,
    real_followup: torch.Tensor,
) -> torch.Tensor:
    """
    Allinea la distribuzione del follow-up normalizzato fake vs reale.
    Confronta media, varianza (con penalità asimmetrica se fake < reale)
    e quantili [0.1, 0.25, 0.5, 0.75, 0.9].
    """
    pred = pred_followup.float()
    real = real_followup.float().to(pred.device)
    l_mean  = F.mse_loss(pred.mean(), real.mean())
    l_var   = F.relu(real.var().clamp(1e-6) - pred.var().clamp(1e-6))
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    l_quant = F.mse_loss(torch.quantile(pred, qs), torch.quantile(real, qs))
    return l_mean + 0.5 * l_var + 2.0 * l_quant


def n_visits_loss(
    n_v_pred: torch.Tensor,
    n_v_real: torch.Tensor,
) -> torch.Tensor:
    """
    Allinea la distribuzione del numero di visite fake vs reale.
    Usa Smooth L1 (robusto agli outlier) + penalità su varianza + quantili.
    """
    pred    = n_v_pred.float()
    real    = n_v_real.float().to(pred.device)
    l_mean  = F.smooth_l1_loss(pred, real)
    l_var   = F.relu(real.var().clamp(min=0.0) - pred.var().clamp(min=0.0))
    qs      = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=pred.device)
    l_quant = F.mse_loss(torch.quantile(pred, qs), torch.quantile(real, qs))
    return l_mean + 0.2 * l_var + 1.5 * l_quant


# ======================================================================
# STRUTTURA CATEGORICA STATICA
# ======================================================================

def static_cat_marginal_loss(
    fake_static_cat_soft: Dict[str, torch.Tensor],
    target_probs:         Dict[str, torch.Tensor],
    eps:                  float = 1e-6,
    max_kl:               float = 10.0,
) -> torch.Tensor:
    """
    KL(target || fake_marginal) + CE pesata per categoria rara.

    [Stabilità] Il KL è clampato a max_kl per prevenire esplosione quando
    il generatore collassa su distribuzioni degeneri nelle prime epoche.
    Il peso w = 1/min_p enfatizza le categorie rare (anti-mode-drop).
    """
    losses = []
    # Pre-calcola il device per evitare chiamate ripetute
    device = next(iter(fake_static_cat_soft.values())).device 

    for name, p_soft in fake_static_cat_soft.items():
        if name not in target_probs: continue
        
        # Carica target_probs solo se necessario e assicurati sia float
        p_real = target_probs[name].to(device, non_blocking=True).float()
        p_s = p_soft.float()

        # Marginalizzazione (media lungo il batch)
        p_m = p_s.mean(dim=0).clamp(min=eps)
        p_m = p_m / p_m.sum()
        pr = p_real.clamp(min=eps)

        # KL + CE
        kl = (pr * (pr / p_m).log()).sum().clamp(max=max_kl)
        # La CE può essere semplificata
        ce = -(pr.unsqueeze(0) * p_s.clamp(min=eps).log()).sum(dim=-1).mean()

        w = (1.0 / pr.min().clamp(min=1e-6)).clamp(max=20.0)
        losses.append(w * (kl + 0.5 * ce))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


# ======================================================================
# IRREVERSIBILITÀ
# ======================================================================

def irreversibility_loss(
    irr_states: torch.Tensor,   # [B, T] o [B, T, 1]  ∈ [0,1]
    valid_flag: torch.Tensor,   # [B, T] bool
) -> torch.Tensor:
    """
    Penalizza transizioni 1→0 negli stati irreversibili (es. morte, trapianto).

    Componenti:
      1. Monotonia: relu(-Δstate) sui passi validi
      2. Entropia: penalizza stati incerti (p ~0.5) che si cristallizzano
    """
    if irr_states.dim() == 3:
        irr_states = irr_states.squeeze(-1)
    if valid_flag.dtype != torch.bool:
        valid_flag = valid_flag.bool()

    vf        = valid_flag[:, 1:].float()
    diff      = irr_states[:, 1:] - irr_states[:, :-1]
    loss_mono = (torch.clamp(-diff, min=0) * vf).mean()

    vm  = valid_flag.unsqueeze(-1).float()
    h   = irr_states.clamp(1e-6, 1 - 1e-6)
    ent = -(h * h.log() + (1 - h) * (1 - h).log())
    loss_ent = (ent * vm.squeeze(-1)).sum() / vm.sum().clamp(min=1.0)

    return loss_mono + 0.1 * loss_ent


# ======================================================================
# FEATURE MATCHING
# ======================================================================

def feature_matching_loss(
    features_real: torch.Tensor,
    features_fake: torch.Tensor,
) -> torch.Tensor:
    """
    MSE tra feature medie reali e fake del discriminatore statico.
    Stabilizza il training del generatore spingendo verso rappresentazioni
    simili a quelle dei dati reali nello spazio latente del discriminatore.
    """
    return F.mse_loss(features_fake.mean(dim=0), features_real.mean(dim=0))


# ======================================================================
# UTILITÀ
# ======================================================================

def check_finite(loss: torch.Tensor, name: str) -> torch.Tensor:
    """Sostituisce NaN/Inf con 0 e emette un warning."""
    if not torch.isfinite(loss):
        warnings.warn(
            f"Loss '{name}' non finita ({loss.item():.4f}). "
            f"Sostituita con 0. Controlla lr, lambda o stabilità del training.",
            UserWarning,
            stacklevel=2,
        )
        return torch.zeros_like(loss)
    return loss


# ======================================================================
# PREVALENZA CATEGORICHE TEMPORALI IRREVERSIBILI
# ======================================================================

def temporal_irr_prevalence_loss(
    fake_temporal_cat:    Dict[str, torch.Tensor],   # {name: [B, T, n_cat]}
    target_prevalence:    Dict[str, float],           # {name: float ∈ [0,1]}
    valid_flag:           torch.Tensor,               # [B, T] bool
    eps:                  float = 1e-6,
) -> torch.Tensor:
    """
    Allinea la prevalenza finale delle variabili categoriche binarie
    irreversibili (stato=1 nell'ultima visita valida) tra dati fake e reali.

    Il problema: per variabili rare (HEPC=0.4%, VARB=1.2%), il generatore
    tende a produrre stato=1 per quasi tutti i pazienti perché:
      1. La irreversibility_loss premia la monotonia crescente senza target
      2. Nessuna loss penalizzava la frequenza assoluta dell'evento

    Componenti:
      1. MSE tra prevalenza finale fake e target reale
      2. Penalità asimmetrica forte per sovrastima (più comune del caso inverso)

    Parametri
    ----------
    target_prevalence : {nome_var: frazione_reale} calcolata dal dataset.
                        Es. {"HEPC": 0.004, "VARB": 0.012, "ESOVAR": 0.074}
    """
    if not fake_temporal_cat or not target_prevalence:
        return torch.tensor(0.0)

    losses = []
    B = valid_flag.shape[0]

    for name, cat_ohe in fake_temporal_cat.items():
        if name not in target_prevalence:
            continue

        target = float(target_prevalence[name])

        # Stato=1: ultima colonna dell'OHE per binarie irreversibili (0=no, 1=sì)
        # cat_ohe: [B, T, 2] per variabili binarie
        if cat_ohe.shape[-1] != 2:
            continue  # skip non-binarie

        p_state1 = cat_ohe[:, :, 1]  # [B, T] probabilità stato=1

        # Prendi il valore all'ultima visita valida per paziente
        final_probs = []
        for b in range(B):
            valid_idx = valid_flag[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                final_probs.append(p_state1[b, 0])
            else:
                final_probs.append(p_state1[b, valid_idx[-1]])

        final_p  = torch.stack(final_probs)          # [B]
        prev_est = final_p.mean()                     # prevalenza media finale

        t_tensor  = torch.tensor(target, device=cat_ohe.device)
        mse       = (prev_est - t_tensor).pow(2)

        # Penalità asimmetrica: sovrastima (fake > target) è molto più comune
        # e dannosa (genera eventi falsi). Peso 5× in direzione sovrastima.
        overshoot = F.relu(prev_est - t_tensor * 3.0)   # tollera fino a 3× il target
        asymm     = 5.0 * overshoot.pow(2)

        losses.append(mse + asymm)

    if not losses:
        return torch.tensor(0.0)

    return torch.stack(losses).mean()