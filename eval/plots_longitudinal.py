# ======================================================
# eval/plots_longitudinal.py
# Longitudinal & temporal plots:
#   - temporal trajectories (mean +/- 95% CI)
#   - slope / within-variance / autocorrelation KDE grids
#   - visit count distribution
#   - visit timing distribution (over time)
#   - temporal cross-correlation heatmap
#   - Kaplan-Meier (POISE responder)
# ======================================================

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

from eval.config import COLOR_REAL, COLOR_SYNTH
from eval.metrics_longitudinal import (
    compute_patient_slopes,
    compute_within_variance,
    compute_autocorrelation,
    compute_visit_counts,
)


# -- Interpolation helpers ----------------------------

def _build_interpolated_curves(df, var, time_col, patient_col, grid):
    curves = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[time_col], errors="coerce").values
        y = pd.to_numeric(g[var],      errors="coerce").values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            continue
        xv, yv = x[mask], y[mask]
        interp_y = np.where(
            (grid >= xv.min()) & (grid <= xv.max()),
            np.interp(grid, xv, yv),
            np.nan,
        )
        curves.append(interp_y)
    return np.array(curves) if curves else np.empty((0, len(grid)))


def _ci95(curves):
    n_cols = curves.shape[1] if curves.ndim > 1 else 0
    if curves.shape[0] == 0:
        nan_arr = np.full(n_cols, np.nan)
        return nan_arr, nan_arr, nan_arr, np.zeros(n_cols)
    n_valid = np.sum(~np.isnan(curves), axis=0)
    mean = np.where(n_valid > 0, np.nanmean(curves, axis=0), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.where(n_valid >= 2, np.nanstd(curves, axis=0, ddof=0), np.nan)
    stderr = np.where(n_valid >= 3, std / np.sqrt(np.maximum(n_valid, 1)), np.nan)
    lo = np.where(n_valid >= 3, mean - 1.96 * stderr, np.nan)
    hi = np.where(n_valid >= 3, mean + 1.96 * stderr, np.nan)
    return mean, lo, hi, n_valid


# -- Temporal trajectory ------------------------------

def plot_temporal_trajectory(
    real, synth, var,
    time_col="MONTHS_FROM_BASELINE",
    patient_col="RECORD_ID",
    max_time=None,
    n_grid=60,
    min_patients=5,
    outdir="plots",
) -> str | None:
    t_max = pd.to_numeric(synth[time_col], errors="coerce").max()
    if max_time is None:
        max_time = t_max
    if max_time is None or np.isnan(max_time) or max_time <= 0:
        return None

    grid     = np.linspace(0.0, float(max_time), int(n_grid))
    r_curves = _build_interpolated_curves(real,  var, time_col, patient_col, grid)
    s_curves = _build_interpolated_curves(synth, var, time_col, patient_col, grid)
    if len(r_curves) == 0 or len(s_curves) == 0:
        return None

    r_mean, r_lo, r_hi, r_n = _ci95(r_curves)
    s_mean, s_lo, s_hi, s_n = _ci95(s_curves)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(grid, r_mean, color=COLOR_REAL,  lw=2, label=f"Real (n={len(r_curves)})")
    ax.fill_between(grid, r_lo, r_hi, where=r_n >= min_patients,
                    color=COLOR_REAL, alpha=0.2, label="CI 95% Real")
    ax.plot(grid, s_mean, color=COLOR_SYNTH, lw=2, label=f"Synth (n={len(s_curves)})")
    ax.fill_between(grid, s_lo, s_hi, where=s_n >= min_patients,
                    color=COLOR_SYNTH, alpha=0.2, label="CI 95% Synth")
    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel(var)
    ax.set_title(f"Trajectory - {var}", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(0, max_time)
    plt.tight_layout()

    path = os.path.join(outdir, f"trajectory_{var}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -- Longitudinal KDE grids (3 per row, marker-by-marker) --

def _kde_panel(ax, real_vals, synth_vals, title):
    sns.kdeplot(real_vals,  ax=ax, label=f"Real (n={len(real_vals)})",  fill=True, alpha=0.3,
                color=COLOR_REAL)
    sns.kdeplot(synth_vals, ax=ax, label=f"Synth (n={len(synth_vals)})", fill=True, alpha=0.3,
                color=COLOR_SYNTH)
    if len(real_vals) > 5 and len(synth_vals) > 5:
        ks, p = ks_2samp(real_vals, synth_vals)
        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.text(0.02, 0.97, f"KS={ks:.3f}, {p_str}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlabel("")


def _save_grid(rows_data: list[tuple], outdir: str, prefix: str, ncols: int = 3) -> list[str]:
    """
    rows_data: list of (real_vals, synth_vals, title)
    Saves grids of ncols panels per row, max 3 rows per page -> 9 per page.
    """
    paths = []
    per_page = ncols * 3
    for i in range(0, len(rows_data), per_page):
        chunk = rows_data[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = axes.flatten()
        for ax, (r_vals, s_vals, title) in zip(axes, chunk):
            _kde_panel(ax, r_vals, s_vals, title)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.subplots_adjust(hspace=0.55, wspace=0.35)
        path = os.path.join(outdir, f"{prefix}_{i}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def plot_slope_grid(real, synth, temporal_vars, time_col, patient_col, outdir) -> list[str]:
    data = []
    for v in temporal_vars:
        r = compute_patient_slopes(real,  v, time_col, patient_col)
        s = compute_patient_slopes(synth, v, time_col, patient_col)
        if len(r) > 5 and len(s) > 5:
            data.append((r, s, f"Slope - {v}"))
    return _save_grid(data, outdir, "slope") if data else []


def plot_variance_grid(real, synth, temporal_vars, patient_col, outdir) -> list[str]:
    data = []
    for v in temporal_vars:
        r = compute_within_variance(real,  v, patient_col)
        s = compute_within_variance(synth, v, patient_col)
        if len(r) > 5 and len(s) > 5:
            data.append((r, s, f"Intra-patient Variance - {v}"))
    return _save_grid(data, outdir, "variance") if data else []


def plot_autocorrelation_grid(real, synth, temporal_vars, time_col, patient_col, outdir) -> list[str]:
    data = []
    for v in temporal_vars:
        r = compute_autocorrelation(real,  v, time_col, patient_col)
        s = compute_autocorrelation(synth, v, time_col, patient_col)
        if len(r) > 5 and len(s) > 5:
            data.append((r, s, f"Autocorrelation - {v}"))
    return _save_grid(data, outdir, "autocorr") if data else []


# -- Visit count distribution -------------------------

def plot_visit_distribution(real, synth, patient_col, outdir) -> str:
    real_counts  = compute_visit_counts(real,  patient_col)
    synth_counts = compute_visit_counts(synth, patient_col)
    real_dist  = real_counts.value_counts(normalize=True).sort_index()
    synth_dist = synth_counts.value_counts(normalize=True).sort_index()
    all_visits = sorted(set(real_dist.index) | set(synth_dist.index))
    real_dist  = real_dist.reindex(all_visits, fill_value=0)
    synth_dist = synth_dist.reindex(all_visits, fill_value=0)

    df_plot = pd.DataFrame({
        "Visits": all_visits,
        "Real": real_dist.values,
        "Synthetic": synth_dist.values,
    }).melt(id_vars="Visits", var_name="Dataset", value_name="Percentage")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_plot, x="Visits", y="Percentage", hue="Dataset",
                palette=[COLOR_REAL, COLOR_SYNTH], ax=ax)
    ks, p = ks_2samp(real_counts.values, synth_counts.values)
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.text(0.02, 0.97, f"KS={ks:.3f}, {p_str}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax.set_title("Distribution of Number of Visits per Patient", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "visit_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -- Visit timing distribution -------------------------

def plot_visit_timing(real, synth, time_col, outdir) -> str:
    """
    KDE of all visit time-points (i.e. when visits occur along the timeline).
    """
    r_times = pd.to_numeric(real[time_col],  errors="coerce").dropna()
    s_times = pd.to_numeric(synth[time_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(r_times, ax=ax, color=COLOR_REAL,  label=f"Real (n={len(r_times)})",  fill=True, alpha=0.3)
    sns.kdeplot(s_times, ax=ax, color=COLOR_SYNTH, label=f"Synth (n={len(s_times)})", fill=True, alpha=0.3)

    ks, p = ks_2samp(r_times, s_times)
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.text(0.02, 0.97, f"KS={ks:.3f}, {p_str}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_title("Distribution of Visit Timing", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "visit_timing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -- Temporal cross-correlation heatmap ---------------

def plot_temporal_cross_correlation(
    real, synth, temporal_vars, time_col, outdir, n_bins=8
) -> str | None:
    """
    Heatmap of pairwise Pearson correlations at each time bin,
    showing Real vs Synthetic side by side.
    """
    def corr_matrix_at_bins(df):
        df = df.copy()
        df["_t"] = pd.to_numeric(df[time_col], errors="coerce")
        valid = [v for v in temporal_vars if v in df.columns]
        if len(valid) < 2:
            return None
        mat = np.zeros((len(valid), len(valid)))
        for i, vi in enumerate(valid):
            for j, vj in enumerate(valid):
                if i == j:
                    mat[i, j] = 1.0
                elif j > i:
                    xi = pd.to_numeric(df[vi], errors="coerce")
                    xj = pd.to_numeric(df[vj], errors="coerce")
                    mask = xi.notna() & xj.notna()
                    if mask.sum() >= 10:
                        r = np.corrcoef(xi[mask], xj[mask])[0, 1]
                        mat[i, j] = mat[j, i] = r if not np.isnan(r) else 0.0
        return pd.DataFrame(mat, index=valid, columns=valid)

    r_mat = corr_matrix_at_bins(real)
    s_mat = corr_matrix_at_bins(synth)
    if r_mat is None or s_mat is None:
        return None

    diff = (r_mat - s_mat).abs()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    kw = dict(cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
              annot=(len(temporal_vars) <= 12), fmt=".2f", annot_kws={"size": 7})
    sns.heatmap(r_mat,   ax=axes[0], **kw)
    axes[0].set_title("Real - Temporal Correlations", color=COLOR_REAL, fontweight="bold")
    sns.heatmap(s_mat,   ax=axes[1], **kw)
    axes[1].set_title("Synthetic - Temporal Correlations", color=COLOR_SYNTH, fontweight="bold")
    sns.heatmap(diff, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1, square=True,
                annot=(len(temporal_vars) <= 12), fmt=".2f", annot_kws={"size": 7})
    axes[2].set_title("|Real - Synthetic|", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "temporal_cross_corr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -- Kaplan-Meier (POISE responder) -------------------

def _classify_poise_responder(
    df,
    alp_col="ALP",
    bili_col="BIL",
    time_col="MONTHS_FROM_BASELINE",
    patient_col="RECORD_ID",
):
    df = df.copy()

    # Converte tempo in numerico
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    # Rimuove righe senza tempo
    df = df.dropna(subset=[time_col])

    # Calcola distanza assoluta da 12 mesi
    df["_TIME_DIFF"] = (df[time_col] - 12).abs()

    # Trova per ogni paziente la riga con visita più vicina a 12
    idx = df.groupby(patient_col)["_TIME_DIFF"].idxmin()

    subset = df.loc[idx].copy()

    # Converte biomarcatori
    subset[alp_col] = pd.to_numeric(subset[alp_col], errors="coerce")
    subset[bili_col] = pd.to_numeric(subset[bili_col], errors="coerce")

    # Definizione POISE
    subset["RESPONDER"] = (
        (subset[alp_col] <= 1.67) &
        #(subset[bili_col] > 1) &
        (subset[bili_col] < 1)
    ).astype(int)

    # Restituisce solo info per paziente
    return subset[[patient_col, "RESPONDER"]]


def plot_km_responder(real, synth, time_col, patient_col, outdir) -> str | None:
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("[WARNING] lifelines not installed - KM plot skipped.")
        return None

    real  = _classify_poise_responder(real,  time_col=time_col, patient_col=patient_col)
    synth = _classify_poise_responder(synth, time_col=time_col, patient_col=patient_col)

    def build_km_df(df):
        return df.groupby(patient_col).agg(
            time=(time_col, "max"),
            RESPONDER=("RESPONDER", "first"),
        ).reset_index().assign(event=1)

    real_km  = build_km_df(real)
    synth_km = build_km_df(synth)

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    colors = [COLOR_REAL, COLOR_REAL, COLOR_SYNTH, COLOR_SYNTH]
    styles = ["-", "--", "-", "--"]
    for (label, ds), c, ls in zip(
        [("Real Responder",      real_km[real_km["RESPONDER"] == 1]),
         ("Real Non-Responder",  real_km[real_km["RESPONDER"] == 0]),
         ("Synth Responder",     synth_km[synth_km["RESPONDER"] == 1]),
         ("Synth Non-Responder", synth_km[synth_km["RESPONDER"] == 0])],
        colors, styles,
    ):
        if len(ds) > 0:
            kmf.fit(ds["time"], ds["event"], label=f"{label} (n={len(ds)})")
            kmf.plot(ax=ax, color=c, linestyle=ls)

    ax.set_title("KM: Responder vs Non-Responder (POISE Criteria)", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(outdir, "km_responder.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path