# ======================================================
# eval/metrics_scores.py  [v4]
# Composite [0-1] summary scores.
#
# Cambiamenti rispetto a v3:
#   - SFS ridotto a 6 componenti (rimosso JS e Wasserstein
#     ridondanti; rimasto KS come metrica primaria)
#   - LFS ridotto a 5 componenti core
#   - iv_mean_raw usa 1/(1+diff) invece di 1-clip(diff,0,1)
#     -> nessun clipping, distingue diff=1.3 da diff=2.8
#   - Aggiunto truncate_to_max_visits() per confronto equo
# ======================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, kendalltau
from dataclasses import dataclass, field
from typing import Dict, Optional


# ======================================================
# Utility: tronca il dataset reale al max_visits del config
# ======================================================

def truncate_to_max_visits(
    df: pd.DataFrame,
    patient_col: str,
    time_col: str,
    max_visits: int,
) -> pd.DataFrame:
    """
    Per ogni paziente, mantieni solo le prime `max_visits` visite
    (ordinate per time_col). Questo rende il dataset reale comparabile
    con il sintetico, che non puo' avere piu' di max_visits step.

    Senza questo troncamento, pazienti reali con 30-40 visite hanno
    slope, varianze e intervalli sistematicamente diversi da quelli
    sintetici (che ne hanno al massimo max_visits), gonfiando
    artificialmente le metriche longitudinali.

    Uso in main_eval.py:
        real_trunc = truncate_to_max_visits(
            real, patient_col, time_col, data_cfg.max_visits)
        long_metrics = calculate_longitudinal_metrics(
            real_trunc, synth, ...)
    """
    # Rank rows within each patient by time, keep first max_visits.
    # This approach is compatible with pandas >= 2.x where groupby apply
    # drops the groupby column from the result.
    df2 = df.copy()
    df2["_visit_rank"] = df2.groupby(patient_col)[time_col].rank(method="first")
    return df2[df2["_visit_rank"] <= max_visits].drop(columns="_visit_rank").reset_index(drop=True)


# ======================================================
# Converters
# ======================================================

def _score_lower(raw: float, max_val: float = 1.0) -> float:
    """Lower-is-better -> score [0,1]. score = 1 - clip(raw/max_val, 0, 1)."""
    if np.isnan(raw) or raw < 0:
        return float("nan")
    return float(np.clip(1.0 - raw / max_val, 0.0, 1.0))


def _score_higher(raw: float) -> float:
    """Already [0,1] higher-is-better. Just clip."""
    if np.isnan(raw):
        return float("nan")
    return float(np.clip(raw, 0.0, 1.0))


def _score_reldiff(diff: float) -> float:
    """
    Relative difference [0, inf) -> score [0,1].
    Usa 1/(1+diff) invece di 1-clip(diff,0,1).
    Vantaggi:
      - Nessun clipping: diff=1.3 -> 0.43, diff=2.8 -> 0.26
      - Distingue diff=0.1 (0.91) da diff=0.5 (0.67) da diff=2.0 (0.33)
      - Sempre in (0,1], mai satura a 0
    """
    if np.isnan(diff) or diff < 0:
        return float("nan")
    return float(1.0 / (1.0 + diff))


def _grade(overall: float) -> str:
    if np.isnan(overall):
        return "N/A"
    if overall >= 0.85:   return "Eccellente  [****]"
    elif overall >= 0.70: return "Buona       [***]"
    elif overall >= 0.55: return "Accettabile [**]"
    else:                 return "Scadente    [*]"


def _fmt(v) -> str | float:
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return round(float(v), 4)


# ======================================================
# TEMPORAL COHERENCE SCORE (TCS) — invariato
# ======================================================

def compute_temporal_coherence_score(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    temporal_vars: list,
    time_col: str,
    patient_col: str,
) -> Dict:
    """
    Due componenti (peso 1:1.5):
    1. Transition consistency: 1 - |p_up_real - p_up_synth| per variabile
    2. Cross-variable Kendall tau: 1 - KS(tau_real, tau_synth) per coppia
    """
    tc_scores, ktc_scores = [], []

    for v in temporal_vars:
        r_ups, s_ups = [], []
        for _, g in real.groupby(patient_col):
            y = pd.to_numeric(g.sort_values(time_col)[v], errors="coerce").dropna().values
            if len(y) >= 2:
                r_ups.extend((np.diff(y) > 0).tolist())
        for _, g in synth.groupby(patient_col):
            y = pd.to_numeric(g.sort_values(time_col)[v], errors="coerce").dropna().values
            if len(y) >= 2:
                s_ups.extend((np.diff(y) > 0).tolist())
        if r_ups and s_ups:
            tc_scores.append(1.0 - abs(float(np.mean(r_ups)) - float(np.mean(s_ups))))

    for i in range(len(temporal_vars)):
        for j in range(i + 1, len(temporal_vars)):
            vi, vj = temporal_vars[i], temporal_vars[j]
            if vi not in real.columns or vj not in real.columns:
                continue
            taus_r, taus_s = [], []
            for _, g in real.groupby(patient_col):
                g = g.sort_values(time_col)
                xi = pd.to_numeric(g[vi], errors="coerce").dropna()
                xj = pd.to_numeric(g[vj], errors="coerce").dropna()
                n = min(len(xi), len(xj))
                if n >= 3:
                    tau, _ = kendalltau(xi.values[:n], xj.values[:n])
                    if not np.isnan(tau):
                        taus_r.append(tau)
            for _, g in synth.groupby(patient_col):
                g = g.sort_values(time_col)
                xi = pd.to_numeric(g[vi], errors="coerce").dropna()
                xj = pd.to_numeric(g[vj], errors="coerce").dropna()
                n = min(len(xi), len(xj))
                if n >= 3:
                    tau, _ = kendalltau(xi.values[:n], xj.values[:n])
                    if not np.isnan(tau):
                        taus_s.append(tau)
            if len(taus_r) > 5 and len(taus_s) > 5:
                ks, _ = ks_2samp(taus_r, taus_s)
                ktc_scores.append(1.0 - ks)

    tc  = float(np.mean(tc_scores))  if tc_scores  else float("nan")
    ktc = float(np.mean(ktc_scores)) if ktc_scores else float("nan")

    ws, tw = 0.0, 0.0
    for val, w in [(tc, 1.0), (ktc, 1.5)]:
        if not np.isnan(val):
            ws += val * w; tw += w
    overall = float(ws / tw) if tw > 0 else float("nan")

    return {
        "TCS - Transition consistency [0-1]":      _fmt(tc),
        "TCS - Cross-var Kendall tau [0-1]":        _fmt(ktc),
        "*** Temporal Coherence Score (TCS) [0-1]": _fmt(overall),
    }


# ======================================================
# STATISTICAL FIDELITY SCORE (SFS) — 6 componenti
# ======================================================
# Ridotto da 8 a 6: rimosso JS (ridondante con KS) e Wasserstein
# exp sim (ridondante con KS). Mantenute le metriche piu' informative.

_SFS_COMPONENTS = [
    # (attr,          label,                               dir,      w,   max_val)
    ("ks_raw",   "KS distance (avg)",                    "lower",  2.5,  1.0),
    ("cat_raw",  "Categorical overlap",                  "higher", 1.5,  1.0),
    ("corr_c_raw","Corr. dist. cont. (MAE Pearson)",     "lower",  2.0,  1.0),
    ("corr_k_raw","Corr. dist. cat. (MAE Cramer's V)",   "lower",  1.5,  1.0),
    ("pca_c_raw","PCA centroid sim",                     "higher", 1.5,  1.0),
    ("pca_o_raw","PCA distribution overlap",             "higher", 1.0,  1.0),
]


@dataclass
class StatisticalFidelityScore:
    ks_raw:    float = float("nan")
    cat_raw:   float = float("nan")
    corr_c_raw:float = float("nan")
    corr_k_raw:float = float("nan")
    pca_c_raw: float = float("nan")
    pca_o_raw: float = float("nan")
    overall:   float = float("nan")
    grade:     str   = "N/A"

    def _score(self, attr: str, direction: str, max_val: float) -> float:
        raw = getattr(self, attr)
        if np.isnan(raw): return float("nan")
        return _score_lower(raw, max_val) if direction == "lower" else _score_higher(raw)

    def to_dict(self) -> Dict:
        rows = {}
        for attr, label, direction, weight, max_val in _SFS_COMPONENTS:
            raw   = getattr(self, attr)
            score = self._score(attr, direction, max_val)
            arrow = "(lower=better -> 1-score)" if direction == "lower" else "(higher=better)"
            rows[f"  {label}"] = (
                f"raw={_fmt(raw)}  score={_fmt(score)}  peso={weight}  {arrow}"
            )
        rows["NOTE"] = "SFS ridotto a 6 metriche core (KS, Cat.overlap, Corr x2, PCA x2)"
        rows["*** SFS overall [0-1]"] = _fmt(self.overall)
        rows["*** SFS Grade"]         = self.grade
        return rows

    def radar_values(self) -> Dict[str, float]:
        out = {}
        for attr, label, direction, weight, max_val in _SFS_COMPONENTS:
            s = self._score(attr, direction, max_val)
            short = label.split("(")[0].strip()[:14]
            out[short] = s if not np.isnan(s) else 0.0
        return out


def compute_statistical_fidelity_score(dist_metrics: dict) -> StatisticalFidelityScore:
    sfs = StatisticalFidelityScore()

    def _get(key):
        v = dist_metrics.get(key)
        if v is None or v == "N/A": return float("nan")
        try: return float(v)
        except: return float("nan")

    sfs.ks_raw     = _get("Avg Kolmogorov-Smirnov Distance ((lower) better)")
    sfs.cat_raw    = _get("Avg Categorical Overlap [0-1] ((higher) better)")
    sfs.corr_c_raw = _get("Correlation Distance - Continuous (MAE Pearson)")
    sfs.corr_k_raw = _get("Correlation Distance - Categorical (MAE Cramer's V)")
    sfs.pca_c_raw  = _get("PCA Centroid Similarity [0-1] ((higher) better)")
    sfs.pca_o_raw  = _get("PCA Distribution Overlap [0-1] ((higher) better)")

    ws, tw = 0.0, 0.0
    for attr, _, direction, weight, max_val in _SFS_COMPONENTS:
        s = sfs._score(attr, direction, max_val)
        if not np.isnan(s):
            ws += s * weight; tw += weight
    if tw > 0: sfs.overall = float(ws / tw)
    sfs.grade = _grade(sfs.overall)
    return sfs


# ======================================================
# LONGITUDINAL FIDELITY SCORE (LFS) — 5 componenti core
# ======================================================
# Ridotto da 8 a 5:
#   - KS slopes + KS variance -> media (slope/variance sono correlati)
#   - KS autocorrelation rimane (struttura temporale)
#   - KS timing posizionale rimane (critico per PBC)
#   - Interval mean rel.diff rimane (usa 1/(1+diff))
#   - TCS (coerenza) rimane con peso maggiore
#
# KS visit count rimosso: già incorporato nel timing posizionale
# Interval std rimosso: molto correlato con iv_mean e instabile
# (std=33 vs 8 è già catturato da iv_mean=24 vs 10)

_LFS_COMPONENTS = [
    # (attr,           label,                         dir,      w,    max_val/note)
    ("dyn_raw",  "KS dynamics (slopes+variance)",    "lower",  2.0,  1.0),
    ("ac_raw",   "KS autocorrelation",               "lower",  1.0,  1.0),
    ("timing_raw","KS timing (positional)",          "lower",  2.5,  1.0),
    ("iv_raw",   "Interval mean rel.diff",           "reldiff",2.0,  None),
    ("coh_raw",  "Temporal coherence (TCS)",         "higher", 2.5,  1.0),
]


@dataclass
class LongitudinalFidelityScore:
    dyn_raw:    float = float("nan")   # media KS slopes + KS variance
    ac_raw:     float = float("nan")
    timing_raw: float = float("nan")
    iv_raw:     float = float("nan")   # rel.diff media intervalli (non clippato)
    coh_raw:    float = float("nan")
    # Raw auxiliaries (per tabella dettaglio, non nello score)
    slope_raw:  float = float("nan")
    var_raw:    float = float("nan")
    vc_raw:     float = float("nan")
    iv_mean_raw:float = float("nan")
    iv_std_raw: float = float("nan")
    overall:    float = float("nan")
    grade:      str   = "N/A"

    def _score(self, attr: str, direction: str, max_val) -> float:
        raw = getattr(self, attr)
        if np.isnan(raw): return float("nan")
        if direction == "lower":  return _score_lower(raw, max_val)
        if direction == "higher": return _score_higher(raw)
        if direction == "reldiff":return _score_reldiff(raw)
        return float("nan")

    def to_dict(self) -> Dict:
        rows = {}
        for attr, label, direction, weight, max_val in _LFS_COMPONENTS:
            raw   = getattr(self, attr)
            score = self._score(attr, direction, max_val)
            if direction == "lower":
                arrow = "(lower=better -> 1-score)"
            elif direction == "reldiff":
                arrow = "(rel.diff -> 1/(1+diff), no clipping)"
            else:
                arrow = "(higher=better)"
            rows[f"  {label}"] = (
                f"raw={_fmt(raw)}  score={_fmt(score)}  peso={weight}  {arrow}"
            )
        # Auxiliary details (non nello score, solo informativo)
        rows["  -- Dettaglio: KS slopes (raw)"]     = _fmt(self.slope_raw)
        rows["  -- Dettaglio: KS variance (raw)"]   = _fmt(self.var_raw)
        rows["  -- Dettaglio: KS visit count (raw)"]= _fmt(self.vc_raw)
        rows["  -- Dettaglio: iv_mean real vs synth"]= (
            f"real={_fmt(self.iv_mean_raw)}  synth=? (raw diff={_fmt(self.iv_raw)})"
        )
        rows["NOTE"] = (
            "LFS calcolato su reali TRONCATI a max_visits (confronto equo con sintetici). "
            "iv_raw usa 1/(1+diff): diff=1.3->score=0.43, diff=0->score=1.0"
        )
        rows["*** LFS overall [0-1]"] = _fmt(self.overall)
        rows["*** LFS Grade"]         = self.grade
        return rows

    def radar_values(self) -> Dict[str, float]:
        out = {}
        for attr, label, direction, weight, max_val in _LFS_COMPONENTS:
            s = self._score(attr, direction, max_val)
            short = label.split("(")[0].strip()[:14]
            out[short] = s if not np.isnan(s) else 0.0
        return out


def compute_longitudinal_fidelity_score(
    long_metrics: dict,
    pos_metrics:  dict,
    tcs_dict:     dict,
    interval_stats_real:  Optional[Dict] = None,
    interval_stats_synth: Optional[Dict] = None,
) -> LongitudinalFidelityScore:
    """
    Calcola LFS dal dict di metriche longitudinali.

    IMPORTANTE: long_metrics deve essere calcolato su real TRONCATO
    a max_visits. Usa truncate_to_max_visits() prima di chiamare
    calculate_longitudinal_metrics().
    """
    lfs = LongitudinalFidelityScore()

    def _get(d, key):
        v = d.get(key)
        if v is None or v == "N/A": return float("nan")
        try: return float(v)
        except: return float("nan")

    # Auxiliaries (solo per dettaglio)
    lfs.slope_raw = _get(long_metrics, "Avg KS - Patient Slopes ((lower) better)")
    lfs.var_raw   = _get(long_metrics, "Avg KS - Within-patient Variance ((lower) better)")
    lfs.ac_raw    = _get(long_metrics, "Avg KS - Autocorrelation ((lower) better)")
    lfs.vc_raw    = _get(long_metrics, "KS - Visit Count Distribution ((lower) better)")

    # dyn_raw = media di slopes e variance (due aspetti dello stesso fenomeno)
    s_slope = lfs.slope_raw
    s_var   = lfs.var_raw
    valid   = [x for x in [s_slope, s_var] if not np.isnan(x)]
    if valid:
        lfs.dyn_raw = float(np.mean(valid))

    lfs.timing_raw = _get(pos_metrics, "Avg KS across visit positions (timing)")

    # Interval: usa rel.diff SENZA clipping, poi _score_reldiff = 1/(1+diff)
    if interval_stats_real and interval_stats_synth:
        try:
            mr = float(interval_stats_real.get("Mean inter-visit interval",  float("nan")))
            ms = float(interval_stats_synth.get("Mean inter-visit interval", float("nan")))
            sr = float(interval_stats_real.get("Std inter-visit interval",   float("nan")))
            ss = float(interval_stats_synth.get("Std inter-visit interval",  float("nan")))
            lfs.iv_mean_raw = mr   # store raw mean real for display
            # iv_raw = rel.diff delle medie (senza clip)
            if not np.isnan(mr) and not np.isnan(ms) and mr > 0:
                lfs.iv_raw = float(abs(mr - ms) / mr)
            # iv_std_raw solo per display
            if not np.isnan(sr) and not np.isnan(ss) and sr > 0:
                lfs.iv_std_raw = float(abs(sr - ss) / sr)
        except Exception:
            pass

    # TCS: higher=better
    tcs_val = _get(tcs_dict, "*** Temporal Coherence Score (TCS) [0-1]")
    if not np.isnan(tcs_val):
        lfs.coh_raw = float(np.clip(tcs_val, 0.0, 1.0))

    # Weighted overall
    ws, tw = 0.0, 0.0
    for attr, _, direction, weight, max_val in _LFS_COMPONENTS:
        s = lfs._score(attr, direction, max_val)
        if not np.isnan(s):
            ws += s * weight; tw += weight
    if tw > 0:
        lfs.overall = float(ws / tw)
    lfs.grade = _grade(lfs.overall)
    return lfs