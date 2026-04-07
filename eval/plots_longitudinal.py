# ======================================================
# eval/plots_longitudinal.py
# Longitudinal & temporal plots:
#   - temporal trajectories (mean +/- 95% CI)
#   - slope / within-variance / autocorrelation KDE grids
#   - visit count distribution
#   - visit timing distribution (over time)
#   - temporal cross-correlation heatmap
#   - Kaplan-Meier (POISE responder)
#   - [NEW] per-visit-position timing distribution
#   - [NEW] last-visit vs D3_fup discrepancy analysis
# ======================================================

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ks_2samp
import umap as umap_lib
from sklearn.preprocessing import StandardScaler
from lifelines.statistics import logrank_test
from lifelines import KaplanMeierFitter

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



def plot_umap(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_vars: list,
    outdir: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    max_samples: int = 1500,
) -> str | None:
    """
    UMAP projection of real and synthetic patients.
    PCA fittato su reali, UMAP fittato su reali, sintetici proiettati.
    Richiede:  pip install umap-learn
    """

    r = real[num_vars].dropna(how="all").fillna(0)
    s = synth[num_vars].dropna(how="all").fillna(0)
    if len(r) < 10 or len(s) < 10:
        return None

    # Subsample for speed
    rr = r.sample(min(max_samples, len(r)), random_state=random_state)
    ss = s.sample(min(max_samples, len(s)), random_state=random_state)

    scaler = StandardScaler()
    Xr = scaler.fit_transform(rr)
    Xs = scaler.transform(ss)

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
    )
    Zr = reducer.fit_transform(Xr)
    Zs = reducer.transform(Xs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.35, s=10,
               color=COLOR_REAL,  label=f"Real (n={len(rr)})")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.35, s=10,
               color=COLOR_SYNTH, label=f"Synthetic (n={len(ss)})")
    ax.scatter(*Zr.mean(0), marker="X", s=130, color=COLOR_REAL,
               zorder=5, edgecolors="white", lw=0.8, label="Real centroid")
    ax.scatter(*Zs.mean(0), marker="X", s=130, color=COLOR_SYNTH,
               zorder=5, edgecolors="white", lw=0.8, label="Synth centroid")
    ax.set_title(
        f"UMAP Projection  (fit on Real, n_neighbors={n_neighbors})\n"
        "Crosses = centroids. Good model: overlapping clouds.",
        fontsize=10, fontweight="bold",
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(outdir, "umap_projection.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path



# -- Longitudinal KDE grids ---------------------------

def _kde_panel(ax, real_vals, synth_vals, title):
    sns.kdeplot(real_vals,  ax=ax, label=f"Real (n={len(real_vals)})",  fill=True, alpha=0.3, warn_singular=False,
                color=COLOR_REAL)
    sns.kdeplot(synth_vals, ax=ax, label=f"Synth (n={len(synth_vals)})", fill=True, alpha=0.3, warn_singular=False,
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
    r_times = pd.to_numeric(real[time_col],  errors="coerce").dropna()
    s_times = pd.to_numeric(synth[time_col], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(r_times, ax=ax, color=COLOR_REAL,  label=f"Real (n={len(r_times)})", warn_singular=False, fill=True, alpha=0.3)
    sns.kdeplot(s_times, ax=ax, color=COLOR_SYNTH, label=f"Synth (n={len(s_times)})", warn_singular=False, fill=True, alpha=0.3)

    ks, p = ks_2samp(r_times, s_times)
    p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
    ax.text(0.02, 0.97, f"KS={ks:.3f}, {p_str}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_title("Distribution of Visit Timing (all visits pooled)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(outdir, "visit_timing.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================
# [NEW] Per-visit-position timing analysis
# ======================================================

def _extract_visit_times_by_position(
    df: pd.DataFrame,
    time_col: str,
    patient_col: str,
    max_position: int,
) -> dict[int, np.ndarray]:
    """
    For each visit position (1-indexed), collect the absolute time values
    across all patients that have at least that many visits.

    Returns: {position: array_of_times}
    """
    position_times: dict[int, list] = {pos: [] for pos in range(1, max_position + 1)}

    for _, g in df.groupby(patient_col):
        times = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
        for pos_idx, t in enumerate(times):
            pos = pos_idx + 1          # 1-indexed
            if pos > max_position:
                break
            position_times[pos].append(float(t))

    return {k: np.array(v) for k, v in position_times.items() if len(v) >= 3}


def plot_visit_position_timing(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    outdir: str,
    n_positions: int | None = None,
    ncols: int = 4,
) -> list[str]:
    """
    [NEW] For each visit position 1..N (where N = mean of synthetic visit counts),
    plot a KDE comparing the distribution of absolute visit times in real vs synthetic.

    This reveals whether the model learns the correct visit spacing at each
    sequential position -- e.g. visit 1 should be at ~0 months, visit 2 at ~6,
    visit 3 at ~12, visit 4 at ~24, etc.

    Also produces a companion boxplot grid showing median ± IQR per position.

    Parameters
    ----------
    n_positions : if None, defaults to int(median of synthetic visit counts).
    ncols       : panels per row in the KDE grid.

    Returns list of saved image paths.
    """
    # Determine N from synthetic median visit count
    synth_counts = synth.groupby(patient_col).size()
    real_counts  = real.groupby(patient_col).size()

    if n_positions is None:
        n_positions = int(np.mean(np.concatenate([real_counts, synth_counts])))

    n_positions = max(2, min(n_positions, int(synth_counts.max())))

    print(f"  [visit-position] N = {n_positions} "
          f"(synth mean={np.mean(synth_counts):.1f}, "
          f"real mean={np.mean(real_counts):.1f})")

    real_pos  = _extract_visit_times_by_position(real,  time_col, patient_col, n_positions)
    synth_pos = _extract_visit_times_by_position(synth, time_col, patient_col, n_positions)

    all_positions = sorted(set(real_pos) & set(synth_pos))
    if not all_positions:
        return []

    paths = []

    # -- 1. KDE grid --------------------------------------------------
    per_page = ncols * 3
    for page_start in range(0, len(all_positions), per_page):
        chunk = all_positions[page_start : page_start + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()

        for ax, pos in zip(axes, chunk):
            r_vals = real_pos.get(pos, np.array([]))
            s_vals = synth_pos.get(pos, np.array([]))

            if len(r_vals) >= 3:
                if np.std(r_vals) < 1e-9:
                    # All values identical (e.g. position 1 always = 0): draw a vertical line
                    ax.axvline(r_vals[0], color=COLOR_REAL, lw=2,
                               label=f"Real  n={len(r_vals)} (all={r_vals[0]:.1f})")
                else:
                    sns.kdeplot(r_vals, ax=ax, color=COLOR_REAL, fill=True, alpha=0.35, 
                                label=f"Real  n={len(r_vals)}", warn_singular=False)
            if len(s_vals) >= 3:
                if np.std(s_vals) < 1e-9:
                    ax.axvline(s_vals[0], color=COLOR_SYNTH, lw=2, linestyle="--",
                               label=f"Synth n={len(s_vals)} (all={s_vals[0]:.1f})")
                else:
                    sns.kdeplot(s_vals, ax=ax, color=COLOR_SYNTH, fill=True, alpha=0.35,
                                label=f"Synth n={len(s_vals)}", warn_singular=False)

            # KS + median annotations
            annotations = []
            if len(r_vals) >= 2 and len(s_vals) >= 2:
                ks_stat, p_val = ks_2samp(r_vals, s_vals)
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                annotations.append(f"KS={ks_stat:.3f}, {p_str}")
            if len(r_vals) >= 1:
                annotations.append(f"mean_real={np.median(r_vals):.1f} mo")
            if len(s_vals) >= 1:
                annotations.append(f"mean_synth={np.median(s_vals):.1f} mo")

            if annotations:
                ax.text(0.02, 0.97, "\n".join(annotations),
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

            ax.set_title(f"Visit position {pos}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Months from baseline")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        for ax in axes[len(chunk):]:
            ax.axis("off")

        fig.suptitle(
            f"Visit-position timing distributions (positions {chunk[0]}-{chunk[-1]} of {n_positions})\n"
            "Each panel shows the distribution of absolute visit time at that sequential position.",
            fontsize=10, y=1.01
        )
        plt.tight_layout()
        path = os.path.join(outdir, f"visit_position_kde_{page_start}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        paths.append(path)

    # -- 2. Median-and-IQR summary plot -------------------------------
    # One figure: median line + IQR band for real and synth across positions
    positions_arr  = np.array(all_positions)
    r_medians = np.array([np.mean(real_pos[p])  for p in all_positions])
    s_medians = np.array([np.mean(synth_pos[p]) for p in all_positions])
    r_q25 = np.array([np.percentile(real_pos[p],  25) for p in all_positions])
    r_q75 = np.array([np.percentile(real_pos[p],  75) for p in all_positions])
    s_q25 = np.array([np.percentile(synth_pos[p], 25) for p in all_positions])
    s_q75 = np.array([np.percentile(synth_pos[p], 75) for p in all_positions])

    # KS per position for the bar subplot
    ks_per_pos = []
    for p in all_positions:
        r_v = real_pos.get(p, np.array([]))
        s_v = synth_pos.get(p, np.array([]))
        if len(r_v) >= 2 and len(s_v) >= 2:
            ks_per_pos.append(ks_2samp(r_v, s_v)[0])
        else:
            ks_per_pos.append(np.nan)

    fig, (ax_main, ax_ks) = plt.subplots(
        2, 1, figsize=(max(8, len(all_positions) * 0.7), 9),
        gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Main: median lines + IQR bands
    ax_main.plot(positions_arr, r_medians, "o-", color=COLOR_REAL,
                 lw=2, ms=5, label="Real -- mean")
    ax_main.fill_between(positions_arr, r_q25, r_q75,
                         color=COLOR_REAL, alpha=0.20, label="Real IQR")
    ax_main.plot(positions_arr, s_medians, "s--", color=COLOR_SYNTH,
                 lw=2, ms=5, label="Synth -- mean")
    ax_main.fill_between(positions_arr, s_q25, s_q75,
                         color=COLOR_SYNTH, alpha=0.20, label="Synth IQR")

    # Expected approximate visit schedule annotation for PBC
    expected = {1: 0, 2: 6, 3: 12}
    for pos, mo in expected.items():
        if pos in all_positions:
            ax_main.axhline(mo, color="gray", lw=0.8, ls=":", alpha=0.5)

    ax_main.set_ylabel("Months from baseline")
    ax_main.set_title(
        f"Mean visit time by sequential position (first {n_positions} visits)\n"
        "Shaded area = IQR. Dotted lines = expected PBC schedule (0, 6, 12 months).",
        fontsize=10, fontweight="bold"
    )
    ax_main.legend(fontsize=8)
    ax_main.grid(axis="y", alpha=0.3)

    # KS subplot (bar per position)
    ks_arr = np.array(ks_per_pos, dtype=float)
    colors_bar = [
        "#d73027" if v >= 0.3 else "#fc8d59" if v >= 0.15 else "#91cf60"
        for v in np.nan_to_num(ks_arr)
    ]
    ax_ks.bar(positions_arr, np.nan_to_num(ks_arr), color=colors_bar, alpha=0.8)
    ax_ks.axhline(0.15, color="#fc8d59", lw=1.2, ls="--", label="KS=0.15 (acceptable)")
    ax_ks.axhline(0.30, color="#d73027", lw=1.2, ls="--", label="KS=0.30 (poor)")
    ax_ks.set_xlabel("Visit position")
    ax_ks.set_ylabel("KS distance")
    ax_ks.set_title("KS distance per visit position (green<0.15, orange<0.30, red>=0.30)")
    ax_ks.set_ylim(0, 1.0)
    ax_ks.set_xticks(positions_arr)
    ax_ks.legend(fontsize=7)
    ax_ks.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    summary_path = os.path.join(outdir, "visit_position_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths.insert(0, summary_path)  # summary first

    return paths


# ======================================================
# [NEW] Last-visit vs D3_fup discrepancy analysis
# ======================================================

def plot_last_visit_vs_d3fup(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    d3fup_col: str,
    outdir: str,
) -> list[str]:
    """
    [NEW] Diagnose the discrepancy between the time of the last recorded visit
    (last value of `time_col` per patient) and the declared follow-up duration
    (`fup_col`, in the same unit as `time_col`).

    For PBC/UDCA data, MONTHS_FROM_BASELINE of the last visit should equal
    t_FUP (total follow-up from randomisation).  In exp 5 this was not the
    case because the synthetic inter-visit intervals were too compressed
    (mean 3 months vs real 10.7 months), so the last visit fell well short
    of D3_fup.

    Produces three panels:
      1. Scatter: last_visit_time vs t_FUP  (real & synth side by side)
      2. KDE:     distribution of (t_FUP - last_visit_time), the residual gap
      3. KDE:     distribution of last_visit_time itself

    Also returns a metrics dict for the caller to embed in the PDF.
    """
    paths = []

    def _per_patient(df: pd.DataFrame, label: str):
        """Return per-patient DataFrame with last_visit and t_FUP columns."""
        last_t = (
            df.groupby(patient_col)[time_col]
            .apply(lambda s: pd.to_numeric(s, errors="coerce").max())
            .rename("last_visit")
            .reset_index()
        )
        # D3_fup: take the value from any row (it's a static attribute)
        fup = (
            df.groupby(patient_col)[d3fup_col]
            .first()
            .reset_index()
            .rename(columns={d3fup_col: "d3fup"})
        )
        merged = last_t.merge(fup, on=patient_col, how="inner")
        merged["d3fup"]      = pd.to_numeric(merged["d3fup"],      errors="coerce")
        merged["last_visit"] = pd.to_numeric(merged["last_visit"],  errors="coerce")
        merged = merged.dropna(subset=["d3fup", "last_visit"])
        merged["gap"]    = merged["d3fup"] - merged["last_visit"]
        merged["label"]  = label
        merged["pct_covered"] = (merged["last_visit"] / merged["d3fup"].replace(0, np.nan)).clip(0, 1)
        return merged

    r_df = _per_patient(real,  "Real")
    s_df = _per_patient(synth, "Synthetic")

    if r_df.empty or s_df.empty:
        print(f"  [WARN] plot_last_visit_vs_d3fup: d3fup_col='{d3fup_col}' not found "
              f"or no valid data. Skipping.")
        return []

    # -- Panel layout (1 figure, 3 rows) ------------------------------
    fig = plt.figure(figsize=(16, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_scatter_r = fig.add_subplot(gs[0, 0])
    ax_scatter_s = fig.add_subplot(gs[0, 1])
    ax_gap       = fig.add_subplot(gs[1, :])
    ax_last      = fig.add_subplot(gs[2, :])

    # -- Scatter: last_visit vs d3fup ----------------------------------
    for ax, df_pt, color, title in [
        (ax_scatter_r, r_df, COLOR_REAL,  "Real: last visit vs t_FUP"),
        (ax_scatter_s, s_df, COLOR_SYNTH, "Synthetic: last visit vs t_FUP"),
    ]:
        lim = max(df_pt["d3fup"].max(), df_pt["last_visit"].max()) * 1.05
        ax.scatter(df_pt["d3fup"], df_pt["last_visit"],
                   alpha=0.45, s=14, color=color)
        ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="y = x  (perfect match)")

        # Annotate coverage statistics
        cov = df_pt["pct_covered"]
        ax.text(0.03, 0.96,
                f"Med coverage: {cov.median():.1%}\n"
                f"% fully covered (>=95%): {(cov >= 0.95).mean():.1%}\n"
                f"Median gap: {df_pt['gap'].median():.1f} mo",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        ax.set_xlabel("t_FUP (months)", fontsize=9)
        ax.set_ylabel("Last visit time (months)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # -- KDE: gap = D3_fup - last_visit -------------------------------
    '''
    gap_r = r_df["gap"].dropna().values
    gap_s = s_df["gap"].dropna().values

    sns.kdeplot(gap_r, ax=ax_gap, color=COLOR_REAL,  fill=True, alpha=0.35, warn_singular=False,
                label=f"Real  -- median gap = {np.median(gap_r):.1f} mo  "
                      f"(mean {np.mean(gap_r):.1f} mo)")
    sns.kdeplot(gap_s, ax=ax_gap, color=COLOR_SYNTH, fill=True, alpha=0.35, warn_singular=False,
                label=f"Synth -- median gap = {np.median(gap_s):.1f} mo  "
                      f"(mean {np.mean(gap_s):.1f} mo)")
    ax_gap.axvline(0, color="black", lw=1.2, ls="--", label="Gap = 0 (perfect)")

    if len(gap_r) >= 2 and len(gap_s) >= 2:
        ks_stat, p_val = ks_2samp(gap_r, gap_s)
        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
        ax_gap.text(0.02, 0.96, f"KS={ks_stat:.3f}, {p_str}",
                    transform=ax_gap.transAxes, fontsize=8, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax_gap.set_xlabel("Gap = D3_fup - last_visit_time  (months)", fontsize=9)
    ax_gap.set_ylabel("Density")
    ax_gap.set_title(
        "Distribution of gap between declared follow-up (D3_fup) and last recorded visit\n"
        "Gap ~= 0 means the last visit coincides with end-of-follow-up (correct).\n"
        "Large positive gap means the last visit is earlier than D3_fup (timing compressed).",
        fontsize=9, fontweight="bold"
    )
    ax_gap.legend(fontsize=8)
    ax_gap.grid(alpha=0.3)
    '''
    # -- KDE: last_visit distribution ---------------------------------
    lv_r = r_df["last_visit"].dropna().values
    lv_s = s_df["last_visit"].dropna().values

    sns.kdeplot(lv_r, ax=ax_last, color=COLOR_REAL,  fill=True, alpha=0.35, warn_singular=False,
                label=f"Real  -- median {np.median(lv_r):.1f} mo")
    sns.kdeplot(lv_s, ax=ax_last, color=COLOR_SYNTH, fill=True, alpha=0.35, warn_singular=False,
                label=f"Synth -- median {np.median(lv_s):.1f} mo")

    if len(lv_r) >= 2 and len(lv_s) >= 2:
        ks_stat, p_val = ks_2samp(lv_r, lv_s)
        p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
        ax_last.text(0.02, 0.96, f"KS={ks_stat:.3f}, {p_str}",
                     transform=ax_last.transAxes, fontsize=8, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax_last.set_xlabel("Last visit time (months from baseline)", fontsize=9)
    ax_last.set_ylabel("Density")
    ax_last.set_title("Distribution of last-visit time per patient", fontsize=9, fontweight="bold")
    ax_last.legend(fontsize=8)
    ax_last.grid(alpha=0.3)

    fig.suptitle(
        "Last visit vs t_FUP: temporal alignment diagnostic\n"
        "In a well-calibrated model the last visit time ~= t_FUP for every patient.",
        fontsize=12, fontweight="bold", y=1.01
    )

    path_main = os.path.join(outdir, "last_visit_vs_d3fup.png")
    plt.savefig(path_main, dpi=150, bbox_inches="tight")
    plt.close()
    paths.append(path_main)

    # -- Per-decile diagnostic -----------------------------------------
    # Splits patients into D3_fup deciles and shows coverage per decile.
    # Reveals whether the gap is uniform or concentrated on long-fup patients.
    for df_pt, color, label_str, fname in [
        (r_df, COLOR_REAL,  "Real",      "decile_real"),
        (s_df, COLOR_SYNTH, "Synthetic", "decile_synth"),
    ]:
        if len(df_pt) < 20:
            continue
        df_pt = df_pt.copy()
        df_pt["fup_decile"] = pd.qcut(df_pt["d3fup"], q=10, labels=False, duplicates="drop")
        summary = df_pt.groupby("fup_decile").agg(
            d3fup_median=("d3fup", "median"),
            coverage_median=("pct_covered", "median"),
            coverage_q25=("pct_covered", lambda x: x.quantile(0.25)),
            coverage_q75=("pct_covered", lambda x: x.quantile(0.75)),
        ).reset_index()

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        ax2.plot(summary["d3fup_median"], summary["coverage_median"],
                 "o-", color=color, lw=2, ms=6, label="Median coverage")
        ax2.fill_between(summary["d3fup_median"],
                         summary["coverage_q25"], summary["coverage_q75"],
                         color=color, alpha=0.2, label="IQR coverage")
        ax2.axhline(0.95, color="gray", lw=1, ls="--", label="95% coverage threshold")
        ax2.axhline(1.00, color="black", lw=0.8, ls=":")
        ax2.set_xlabel("t_FUP median per decile (months)", fontsize=9)
        ax2.set_ylabel("Coverage = last_visit / t_FUP", fontsize=9)
        ax2.set_ylim(0, 1.1)
        ax2.set_title(
            f"{label_str}: last-visit coverage by t_FUP decile\n"
            "Coverage = 1 means last visit aligns perfectly with t_FUP.",
            fontsize=9, fontweight="bold"
        )
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        path_dec = os.path.join(outdir, f"last_visit_{fname}.png")
        plt.savefig(path_dec, dpi=150, bbox_inches="tight")
        plt.close()
        paths.append(path_dec)

    return paths


def compute_last_visit_vs_d3fup_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    d3fup_col: str,
) -> dict:
    """
    [NEW] Returns a flat metrics dict for embedding in the PDF summary table.
    Safe to call even if d3fup_col is absent (returns "N/A" values).
    """
    metrics: dict = {}

    def _stats(df, label):
        if d3fup_col not in df.columns:
            return
        last_t = df.groupby(patient_col)[time_col].apply(
            lambda s: pd.to_numeric(s, errors="coerce").max()
        ).rename("last_visit").reset_index()
        fup = df.groupby(patient_col)[d3fup_col].first().reset_index().rename(
            columns={d3fup_col: "fup"}
        )
        m = last_t.merge(fup, on=patient_col).dropna(subset=["fup", "last_visit"])
        if m.empty:
            return
        m["fup"]      = pd.to_numeric(m["fup"],      errors="coerce")
        m["last_visit"] = pd.to_numeric(m["last_visit"],  errors="coerce")
        m["gap"]        = m["fup"] - m["last_visit"]
        m["pct_cov"]    = (m["last_visit"] / m["fup"].replace(0, np.nan)).clip(0, 1)
        pfx = f"Last-visit vs D3_fup [{label}]"
        metrics[f"{pfx} - Median gap (months)"]           = float(m["gap"].median())
        metrics[f"{pfx} - Mean gap (months)"]             = float(m["gap"].mean())
        metrics[f"{pfx} - Median coverage (last/D3_fup)"] = float(m["pct_cov"].median())
        metrics[f"{pfx} - % patients >=95% coverage"]      = float((m["pct_cov"] >= 0.95).mean())

    _stats(real,  "Real")
    _stats(synth, "Synthetic")
    return metrics


# -- Temporal cross-correlation heatmap ---------------

def plot_temporal_cross_correlation(
    real, synth, temporal_vars, time_col, outdir, n_bins=8
) -> str | None:
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

def _classify_poise_responder(df, alp_col="ALP", bili_col="BIL",
                               time_col="MONTHS_FROM_BASELINE",
                               patient_col="RECORD_ID"):
    df = df.copy()
    df["RESPONDER"] = 0
    df["_TIME_DIFF"] = np.abs(pd.to_numeric(df[time_col], errors="coerce") - 12)
    idx = df.sort_values("_TIME_DIFF").groupby(patient_col).head(1) #.index
    subset = idx    #df.loc[idx]
    cond = (
        (pd.to_numeric(subset[alp_col], errors="coerce") <= 2) &
        (pd.to_numeric(subset[bili_col], errors="coerce") <= 1)
    )
    #df.loc[idx, "RESPONDER"] = cond.astype(int).values
    #return df
    out = idx[[patient_col]].copy()
    out["RESPONDER"] = cond.astype(int).values
    
    return out


def plot_km_overall(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    event_col: str | None = None,
    outdir: str = "plots",
) -> str | None:
    """
    Kaplan-Meier overall (non-stratificato) per real vs synthetic.
    Aggiunge log-rank test con p-value.

    Parameters
    ----------
    time_col     : colonna del tempo di follow-up (es. t_FUP o MONTHS_FROM_BASELINE)
    patient_col  : colonna ID paziente
    event_col    : colonna evento binario (1=evento, 0=censurato).
                   Se None, usa last-visit come tempo e event=1 per tutti.
    """
    def _build_km_data(df: pd.DataFrame) -> pd.DataFrame:
        times = df.groupby(patient_col)[time_col].apply(
            lambda s: pd.to_numeric(s, errors="coerce").max()
        ).rename("time").reset_index()
        if event_col and event_col in df.columns:
            events = df.groupby(patient_col)[event_col].max().rename("event").reset_index()
            km_df  = times.merge(events, on=patient_col, how="left")
            km_df["event"] = km_df["event"].fillna(0).astype(int)
        else:
            km_df = times.copy()
            km_df["event"] = 1
        return km_df.dropna(subset=["time"])

    r_km = _build_km_data(real)
    s_km = _build_km_data(synth)

    if len(r_km) < 5 or len(s_km) < 5:
        return None

    # Log-rank test
    lr = logrank_test(
        r_km["time"].values, s_km["time"].values,
        r_km["event"].values, s_km["event"].values,
    )
    p_val  = lr.p_value
    test_s = lr.test_statistic
    p_str  = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"

    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(r_km["time"], r_km["event"],
            label=f"Real (n={len(r_km)})")
    kmf.plot_survival_function(ax=ax, color=COLOR_REAL, ci_show=True)

    kmf2 = KaplanMeierFitter()
    kmf2.fit(s_km["time"], s_km["event"],
             label=f"Synthetic (n={len(s_km)})")
    kmf2.plot_survival_function(ax=ax, color=COLOR_SYNTH, ci_show=True, linestyle="--")

    ax.text(0.98, 0.97,
            f"Log-rank test\nchi2 = {test_s:.2f}\n{p_str}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))

    ax.set_title(
        f"Kaplan-Meier Survival (Overall — non-stratified)\n"
        f"Column: {time_col}   Event: {'last visit' if not event_col else event_col}",
        fontweight="bold", fontsize=10,
    )
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Survival probability")
    plt.tight_layout()
    path = os.path.join(outdir, "km_overall.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_km_responder(real, synth, time_col, patient_col, outdir) -> str | None:
    try:
        from lifelines import KaplanMeierFitter
    except ImportError:
        print("[WARNING] lifelines not installed - KM plot skipped.")
        return None

    real_resp  = _classify_poise_responder(real,  time_col=time_col, patient_col=patient_col)
    synth_resp = _classify_poise_responder(synth, time_col=time_col, patient_col=patient_col)

    def build_km_df(df_original, df_resp):
        max_time = (
            df_original.groupby(patient_col)[time_col]
            .max()
            .reset_index()
            .rename(columns={time_col: "time"})
        )
        merged = max_time.merge(df_resp[[patient_col, "RESPONDER"]], on=patient_col, how="inner")
        merged["event"] = 1
        return merged

    real_km  = build_km_df(real,  real_resp)
    synth_km = build_km_df(synth, synth_resp)

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

# per-visit variable comparison
def plot_variable_by_visit(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    temporal_vars: list,
    time_col: str,
    patient_col: str,
    outdir: str,
    n_visit_bins: int = 6,
    max_vars_per_page: int = 4,
) -> list[str]:
    """
    Per ogni variabile temporale, confronta la distribuzione dei valori
    raggruppando le visite in N bin equidistanti (es. 0-2, 2-4, 4-6 mesi se
    n_visit_bins=6 sulla durata massima).

    Per ogni bin: boxplot reali vs sintetici affiancati.
    Annota la KS distance per bin.

    Restituisce lista di path PNG (una per gruppo di variabili).
    """
    # Compute the time grid
    #max_time = pd.to_numeric(pd.concat([real[time_col], synth[time_col]]), errors="coerce").quantile(0.95)
    max_time = 72
    if max_time <= 0 or np.isnan(max_time):
        return []

    edges = np.linspace(0, float(max_time), n_visit_bins + 1)
    labels = [f"{edges[i]:.0f}-{edges[i+1]:.0f}" for i in range(n_visit_bins)]

    def _assign_bins(df):
        out = df.copy()
        t_num = pd.to_numeric(df[time_col], errors="coerce")
        out["_visit_bin"] = pd.cut(t_num, bins=edges, labels=labels, include_lowest=True)
        return out

    r_binned = _assign_bins(real)
    s_binned = _assign_bins(synth)

    paths = []
    for page_start in range(0, len(temporal_vars), max_vars_per_page):
        chunk = temporal_vars[page_start : page_start + max_vars_per_page]
        ncols = 2
        nrows = len(chunk)
        # 3 vars per page: 3 * 3.5" = 10.5" height fits A4 with margins
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows),
                                  squeeze=False)

        for row_idx, v in enumerate(chunk):
            # LEFT: boxplot per bin
            ax_box = axes[row_idx][0]
            ax_ks  = axes[row_idx][1]

            box_data_r, box_data_s = [], []
            ks_vals = []
            valid_labels = []

            for lbl in labels:
                r_vals = pd.to_numeric(
                    r_binned[r_binned["_visit_bin"] == lbl][v], errors="coerce"
                ).dropna().values
                s_vals = pd.to_numeric(
                    s_binned[s_binned["_visit_bin"] == lbl][v], errors="coerce"
                ).dropna().values
                if len(r_vals) < 3 or len(s_vals) < 3:
                    continue
                box_data_r.append(r_vals)
                box_data_s.append(s_vals)
                ks, _ = ks_2samp(r_vals, s_vals)
                ks_vals.append(ks)
                valid_labels.append(lbl)

            if not valid_labels:
                ax_box.axis("off"); ax_ks.axis("off")
                continue

            # Interleaved boxplot positions
            positions_r = np.arange(1, len(valid_labels) * 3, 3)
            positions_s = positions_r + 1

            bp_r = ax_box.boxplot(box_data_r, positions=positions_r, widths=0.7,
                                   patch_artist=True, notch=False,
                                   medianprops=dict(color="white", lw=1.5))
            bp_s = ax_box.boxplot(box_data_s, positions=positions_s, widths=0.7,
                                   patch_artist=True, notch=False,
                                   medianprops=dict(color="white", lw=1.5))
            for patch in bp_r["boxes"]:
                patch.set_facecolor(COLOR_REAL);  patch.set_alpha(0.7)
            for patch in bp_s["boxes"]:
                patch.set_facecolor(COLOR_SYNTH); patch.set_alpha(0.7)

            mid_positions = (positions_r + positions_s) / 2
            ax_box.set_xticks(mid_positions)
            ax_box.set_xticklabels(valid_labels, rotation=30, ha="right", fontsize=8)
            ax_box.set_ylabel(v, fontsize=9)
            ax_box.set_title(f"{v} — distribution per visit time bin",
                              fontsize=9, fontweight="bold")
            legend_patches = [
                mpatches.Patch(color=COLOR_REAL,  alpha=0.7, label="Real"),
                mpatches.Patch(color=COLOR_SYNTH, alpha=0.7, label="Synthetic"),
            ]
            ax_box.legend(handles=legend_patches, fontsize=8)
            ax_box.grid(axis="y", alpha=0.3)

            # RIGHT: KS per bin
            bar_colors = [
                "#d73027" if ks >= 0.30 else "#fc8d59" if ks >= 0.15 else "#91cf60"
                for ks in ks_vals
            ]
            x_pos = np.arange(len(valid_labels))
            ax_ks.bar(x_pos, ks_vals, color=bar_colors, alpha=0.85)
            ax_ks.axhline(0.15, color="#fc8d59", lw=1.2, ls="--", label="KS=0.15 (acceptable)")
            ax_ks.axhline(0.30, color="#d73027", lw=1.2, ls="--", label="KS=0.30 (poor)")
            ax_ks.set_xticks(x_pos)
            ax_ks.set_xticklabels(valid_labels, rotation=30, ha="right", fontsize=8)
            ax_ks.set_ylabel("KS distance", fontsize=9)
            ax_ks.set_ylim(0, 1.0)
            ax_ks.set_title(f"{v} — KS per time bin (green<0.15, orange<0.30, red>=0.30)",
                              fontsize=9, fontweight="bold")
            ax_ks.legend(fontsize=7)
            ax_ks.grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"Variables by visit time bin  (0-{max_time:.0f} months, {n_visit_bins} bins)",
            fontsize=12, fontweight="bold", y=1.01,
        )
        plt.tight_layout()
        path = os.path.join(outdir, f"variable_by_visit_{page_start}.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        paths.append(path)

    return paths

