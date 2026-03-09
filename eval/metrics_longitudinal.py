# ======================================================
# eval/metrics_longitudinal.py
# Longitudinal & temporal metrics:
#   - per-patient slopes, within-patient variance,
#     autocorrelation, visit counts, temporal correlation
#   - [NEW] per-visit-position timing metrics
# ======================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from sklearn.linear_model import LinearRegression


# -- Per-patient trajectory statistics ---------------

def compute_patient_slopes(
    df: pd.DataFrame, var: str, time_col: str, patient_col: str
) -> np.ndarray:
    slopes = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[time_col], errors="coerce").values.reshape(-1, 1)
        y = pd.to_numeric(g[var],      errors="coerce").values
        mask = ~(np.isnan(x.flatten()) | np.isnan(y))
        if mask.sum() >= 2:
            model = LinearRegression().fit(x[mask], y[mask])
            slopes.append(model.coef_[0])
    return np.array(slopes)


def compute_within_variance(
    df: pd.DataFrame, var: str, patient_col: str
) -> np.ndarray:
    variances = []
    for _, g in df.groupby(patient_col):
        y = pd.to_numeric(g[var], errors="coerce").dropna()
        if len(y) >= 2:
            variances.append(float(np.var(y)))
    return np.array(variances)


def compute_autocorrelation(
    df: pd.DataFrame, var: str, time_col: str, patient_col: str
) -> np.ndarray:
    ac_vals = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        y = pd.to_numeric(g[var], errors="coerce").dropna().values
        if len(y) >= 3:
            if np.std(y[:-1]) == 0 or np.std(y[1:]) == 0:
                continue
            corr = np.corrcoef(y[:-1], y[1:])[0, 1]
            if not np.isnan(corr):
                ac_vals.append(corr)
    return np.array(ac_vals)


# -- Visit-structure metrics --------------------------

def compute_visit_counts(
    df: pd.DataFrame, patient_col: str
) -> pd.Series:
    """Number of visits per patient."""
    return df.groupby(patient_col).size()


def compute_visit_interval_stats(
    df: pd.DataFrame, time_col: str, patient_col: str
) -> dict:
    """Mean and std of inter-visit intervals across all patients."""
    intervals = []
    for _, g in df.groupby(patient_col):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
        if len(t) >= 2:
            intervals.extend(np.diff(t).tolist())
    if not intervals:
        return {"Mean inter-visit interval": "N/A", "Std inter-visit interval": "N/A"}
    return {
        "Mean inter-visit interval": float(np.mean(intervals)),
        "Std inter-visit interval":  float(np.std(intervals)),
    }


# -- [NEW] Per-visit-position timing metrics ----------

def compute_visit_position_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    n_positions: int | None = None,
) -> dict:
    """
    [NEW] For each visit position 1..N, compute:
      - KS distance between real and synthetic time distributions
      - Median absolute error between real and synthetic medians
      - Median visit time for real and synthetic

    N defaults to int(median synthetic visit count).

    Returns a flat dict: position-level metrics + summary averages.
    The per-position KS scores are the most actionable metric for diagnosing
    timing problems -- a model that compresses visits near t=0 will show high
    KS for positions 2 onwards even when position 1 (baseline, always 0) is perfect.

    Parameters
    ----------
    n_positions : maximum number of sequential positions to evaluate.
                  If None, uses int(median of synthetic visit counts).
    """
    synth_counts = synth.groupby(patient_col).size()
    real_counts  = real.groupby(patient_col).size()

    if n_positions is None:
        n_positions = int(np.median(synth_counts.values))

    n_positions = max(2, min(n_positions, int(synth_counts.max())))

    def extract_by_position(df, max_pos):
        pos_times: dict[int, list] = {p: [] for p in range(1, max_pos + 1)}
        for _, g in df.groupby(patient_col):
            times = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            for idx, t in enumerate(times):
                pos = idx + 1
                if pos > max_pos:
                    break
                pos_times[pos].append(float(t))
        return {k: np.array(v) for k, v in pos_times.items() if len(v) >= 3}

    real_pos  = extract_by_position(real,  n_positions)
    synth_pos = extract_by_position(synth, n_positions)

    metrics: dict = {}
    ks_vals, mae_medians = [], []
    positions_evaluated = sorted(set(real_pos) & set(synth_pos))

    for pos in positions_evaluated:
        r_arr = real_pos[pos]
        s_arr = synth_pos[pos]
        if len(r_arr) < 2 or len(s_arr) < 2:
            continue

        ks, _ = ks_2samp(r_arr, s_arr)
        mae   = abs(float(np.median(r_arr)) - float(np.median(s_arr)))
        ks_vals.append(ks)
        mae_medians.append(mae)

        # Only log positions 1, 2, 3 and every 5th after that to keep output compact
        if pos <= 3 or pos % 5 == 0:
            metrics[f"Visit pos {pos:02d}: KS (real vs synth timing)"] = float(ks)
            metrics[f"Visit pos {pos:02d}: median real (mo)"]           = float(np.median(r_arr))
            metrics[f"Visit pos {pos:02d}: median synth (mo)"]          = float(np.median(s_arr))
            metrics[f"Visit pos {pos:02d}: MAE medians (mo)"]           = float(mae)

    if ks_vals:
        metrics["Avg KS across visit positions (timing)"]          = float(np.mean(ks_vals))
        metrics["Max KS across visit positions (timing)"]          = float(np.max(ks_vals))
        metrics["Avg MAE of median visit time per position (mo)"]  = float(np.mean(mae_medians))
        metrics["N positions evaluated"]                            = len(ks_vals)

    return metrics


# -- Temporal correlation (cross-variable at same visit) --

def compute_temporal_cross_correlation(
    df: pd.DataFrame, var1: str, var2: str, time_col: str, patient_col: str,
    n_bins: int = 10
) -> pd.DataFrame | None:
    """
    Mean correlation between var1 and var2 computed in time bins.
    Returns a DataFrame with columns [time_bin, correlation].
    """
    df = df.copy()
    df["_t"] = pd.to_numeric(df[time_col], errors="coerce")
    df["_v1"] = pd.to_numeric(df[var1], errors="coerce")
    df["_v2"] = pd.to_numeric(df[var2], errors="coerce")
    df = df.dropna(subset=["_t", "_v1", "_v2"])
    if len(df) < 10:
        return None
    bins = pd.cut(df["_t"], bins=n_bins)
    rows = []
    for b, grp in df.groupby(bins, observed=True):
        if len(grp) >= 5:
            r, _ = pearsonr(grp["_v1"], grp["_v2"])
            rows.append({"time_bin": b.mid, "correlation": r})
    return pd.DataFrame(rows) if rows else None


# -- Aggregate longitudinal scores --------------------

def calculate_longitudinal_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    temporal_vars: list[str],
    time_col: str,
    patient_col: str,
) -> dict:
    """
    Compute summary longitudinal metrics for all temporal variables.
    Returns a flat dict suitable for PDF reporting.
    """
    metrics = {}

    all_ks_slopes, all_ks_var, all_ks_ac = [], [], []

    for v in temporal_vars:
        slopes_r = compute_patient_slopes(real,  v, time_col, patient_col)
        slopes_s = compute_patient_slopes(synth, v, time_col, patient_col)
        if len(slopes_r) > 5 and len(slopes_s) > 5:
            ks, _ = ks_2samp(slopes_r, slopes_s)
            all_ks_slopes.append(ks)

        var_r = compute_within_variance(real,  v, patient_col)
        var_s = compute_within_variance(synth, v, patient_col)
        if len(var_r) > 5 and len(var_s) > 5:
            ks, _ = ks_2samp(var_r, var_s)
            all_ks_var.append(ks)

        ac_r = compute_autocorrelation(real,  v, time_col, patient_col)
        ac_s = compute_autocorrelation(synth, v, time_col, patient_col)
        if len(ac_r) > 5 and len(ac_s) > 5:
            ks, _ = ks_2samp(ac_r, ac_s)
            all_ks_ac.append(ks)

    metrics["Avg KS - Patient Slopes ((lower) better)"] = (
        float(np.mean(all_ks_slopes)) if all_ks_slopes else "N/A"
    )
    metrics["Avg KS - Within-patient Variance ((lower) better)"] = (
        float(np.mean(all_ks_var)) if all_ks_var else "N/A"
    )
    metrics["Avg KS - Autocorrelation ((lower) better)"] = (
        float(np.mean(all_ks_ac)) if all_ks_ac else "N/A"
    )

    # Visit-count distribution KS
    vc_r = compute_visit_counts(real,  patient_col).values
    vc_s = compute_visit_counts(synth, patient_col).values
    if len(vc_r) > 1 and len(vc_s) > 1:
        ks, _ = ks_2samp(vc_r, vc_s)
        metrics["KS - Visit Count Distribution ((lower) better)"] = float(ks)

    # Interval stats comparison
    iv_r = compute_visit_interval_stats(real,  time_col, patient_col)
    iv_s = compute_visit_interval_stats(synth, time_col, patient_col)
    for k in iv_r:
        vr = iv_r[k]
        vs = iv_s[k]
        if isinstance(vr, float) and isinstance(vs, float):
            metrics[f"{k} - Real"] = vr
            metrics[f"{k} - Synthetic"] = vs

    return metrics