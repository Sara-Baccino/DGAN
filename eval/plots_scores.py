# ======================================================
# eval/plots_scores.py
# Plots for SFS, LFS, TFS scores and new analyses:
#   - plot_score_radar()    : radar for SFS, LFS, TFS
#   - plot_scores_dashboard(): 3-radar comparison dashboard
#   - plot_umap()           : UMAP projection real vs synth
#   - plot_km_overall()     : KM overall (no stratification) + log-rank
#   - plot_variable_by_visit(): per-visit boxplots for temporal vars
# ======================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

try:
    import umap as umap_lib
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    _LIFELINES_AVAILABLE = True
except ImportError:
    _LIFELINES_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

try:
    from eval.config import COLOR_REAL, COLOR_SYNTH
except ImportError:
    COLOR_REAL  = "#457B9D"
    COLOR_SYNTH = "#E63946"


# ======================================================
# RADAR CHART (reusable) — standard, periferia=buono
# ======================================================

def _radar_chart(
    ax,
    values_dict: dict,
    title:   str,
    overall: float,
    grade:   str,
    color:   str = '#1a6faf',
):
    """
    Standard radar chart.
    values_dict: {label: score} all in [0,1] higher=better.
    (lower-is-better raw metrics must be converted before passing here.)
    """
    labels = list(values_dict.keys())
    values = [float(v) if not np.isnan(float(v)) else 0.0 for v in values_dict.values()]
    N      = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    plot_values = values + [values[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.55, 0.70, 0.85, 1.0])
    ax.set_yticklabels(["0.25", "0.55", "0.70", "0.85", "1.0"], size=6.5, color="gray")

    full_angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    ax.fill(full_angles, [1.00] * (N + 1), color="#d4f1d4", alpha=0.22)
    ax.fill(full_angles, [0.70] * (N + 1), color="#fff3cd", alpha=0.33)
    ax.fill(full_angles, [0.55] * (N + 1), color="#f8d7da", alpha=0.38)

    ax.plot(angles, plot_values, "o-", linewidth=2, color=color)
    ax.fill(angles, plot_values, alpha=0.28, color=color)

    overall_str = f"{overall:.3f}" if not np.isnan(overall) else "N/A"
    ax.set_title(
        f"{title}\nOverall = {overall_str}   {grade}\n"
        "(1=perfetto, 0=pessimo; lower-better conv. 1-score)",
        size=9, fontweight="bold", pad=22,
    )


def plot_score_radar(score_obj, title: str, color: str, outdir: str, fname: str) -> str:
    """
    Generates a single radar chart PNG.
    Uses score_obj.radar_values() which returns converted scores [0,1] higher=better.
    Lower-is-better raw metrics are already converted (1-score) before reaching here.
    """
    values_dict = score_obj.radar_values()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    _radar_chart(ax, values_dict, title, score_obj.overall, score_obj.grade, color=color)
    patches = [
        mpatches.Patch(color="#d4f1d4", alpha=0.7, label="Eccellente (>=0.85)"),
        mpatches.Patch(color="#fff3cd", alpha=0.7, label="Buona (>=0.70)"),
        mpatches.Patch(color="#f8d7da", alpha=0.7, label="Scadente (<0.55)"),
    ]
    ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(-0.30, -0.12), fontsize=8)
    plt.tight_layout()
    path = os.path.join(outdir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_scores_dashboard(sfs, lfs, tfs, outdir: str) -> str:
    """
    3-panel side-by-side radar dashboard — tutti standard (periferia=buono).
    Tutti i valori sono gia' convertiti in [0,1] higher=better:
    lower-is-better raw metrics vengono convertite con 1-score prima del radar.
    """
    fig = plt.figure(figsize=(20, 7))
    fig.suptitle(
        "Synthetic Data Quality — Three-Score Overview\n"
        "(tutti gli assi: 1=perfetto, 0=pessimo. "
        "Metriche lower-better convertite con 1-score.)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    configs = [
        (sfs, "Statistical\nFidelity (SFS)", "#2A9D8F"),
        (lfs, "Longitudinal\nFidelity (LFS)", "#E9C46A"),
        (tfs, "Temporal\nFidelity (TFS)",     "#1a6faf"),
    ]
    for idx, (score_obj, title, color) in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx + 1, projection="polar")
        _radar_chart(ax, score_obj.radar_values(), title,
                     score_obj.overall, score_obj.grade, color=color)
    plt.tight_layout()
    path = os.path.join(outdir, "scores_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


# ======================================================
# UMAP
# ======================================================

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
    if not _UMAP_AVAILABLE:
        print("[WARN] umap-learn not installed. Skipping UMAP plot. "
              "Install with: pip install umap-learn")
        return None

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


# ======================================================
# KM OVERALL (non-stratified) + LOG-RANK
# ======================================================

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
    if not _LIFELINES_AVAILABLE:
        print("[WARN] lifelines not installed. Skipping KM overall. "
              "Install with: pip install lifelines")
        return None

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


# ======================================================
# PER-VISIT VARIABLE COMPARISON
# ======================================================

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


# ======================================================
# HELPER: import-safe ks_2samp
# ======================================================
try:
    from scipy.stats import ks_2samp
except ImportError:
    pass