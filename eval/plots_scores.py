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

try:
    from eval.config import COLOR_REAL, COLOR_SYNTH
except ImportError:
    COLOR_REAL  = "#457B9D"
    COLOR_SYNTH = "#E63946"


# RADAR CHART ----------------------------------

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

