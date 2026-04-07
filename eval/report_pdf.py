# ======================================================
# eval/report_pdf.py
# PDF report class and image-layout utilities
# ======================================================

import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, CategoricalDtype
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# -- Colors ------------------------------------------
COLOR_REAL  = "#3B5998"   # Blue
COLOR_SYNTH = "#FF7F50"   # Coral

# helpers ------
def get_variable_types(df: pd.DataFrame, exclude: list[str]):
    """Return (numeric_cols, categorical_cols), excluding IDs/time cols."""
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_dtype(df[c]):
            num.append(c)
        elif is_string_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype):
            cat.append(c)
    return num, cat


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_valid_visits(df: pd.DataFrame, visit_mask_col: str = "VISIT_MASK") -> pd.DataFrame:
    if visit_mask_col in df.columns:
        return df[df[visit_mask_col] == 1].copy()
    return df.copy()


def align_real_to_synth_max_visits(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
) -> pd.DataFrame:
    """Trim real dataset to the same max visit count as synthetic."""
    synth_counts = synth.groupby(patient_col)[time_col].count()
    if len(synth_counts) == 0:
        return real.copy()

    max_visits = int(synth_counts.max())
    print(f"[INFO] Max visits in synthetic: {max_visits}")

    real_sorted  = real.sort_values([patient_col, time_col]).copy()
    real_aligned = (
        real_sorted
        .groupby(patient_col, group_keys=False)
        .head(max_visits)
        .copy()
    )
    print(f"[INFO] Real rows: {len(real)} -> {len(real_aligned)} after alignment")
    return real_aligned


def make_plot_dir(path: str = "plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path

class ReportPDF(FPDF):
    # -- Header printed on every page ----------------
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Synthetic Data Validation Report", ln=True, align="C")
        self.ln(3)

    # -- Section title: always starts a NEW page ------
    def section(self, title: str):
        self.add_page()
        self.set_font("Arial", "B", 13)
        self.set_fill_color(220, 230, 245)
        self.cell(0, 11, title, ln=True, fill=True)
        #self.ln(4)

    # -- Metrics table --------------------------------
    def add_metrics_table(self, metrics_dict: dict, title: str):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)
        for k, v in metrics_dict.items():
            text = f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}"
            self.multi_cell(0, 6, text)
        self.ln(4)

    # -- Italic note block ----------------------------
    def add_note(self, text: str):
        self.set_font("Arial", "I", 9)
        self.multi_cell(0, 5, text)
        self.ln(3)


# -- Layout helpers -----------------------------------

def add_images_grid(pdf: ReportPDF, image_paths: list[str],
                    per_row: int = 3, per_page: int = 9,
                    width: float | None = None):
    """
    Place images in a grid layout.
    per_row  : images per row (2 or 3)
    per_page : images per page (e.g. 9 for 3x3, 4 for 2x2, 6 for 3x2)
    width    : image width in mm; auto-computed from per_row if None
    """
    if not image_paths:
        return

    page_w   = 190          # usable page width (mm)
    gap      = 4            # gap between images (mm)
    if width is None:
        width = (page_w - gap * (per_row - 1)) / per_row

    count = 0
    for img in image_paths:
        if count % per_page == 0:
            pdf.add_page()
        # x position based on column index
        col = count % per_row
        x   = pdf.l_margin + col * (width + gap)
        pdf.image(img, x=x, w=width)
        if col == per_row - 1:
            pdf.ln(2)
        count += 1
    pdf.ln(4)


def add_images_full_width(pdf: ReportPDF, image_paths: list[str], width: float = 190):
    """One image per page, full width (for large plots like correlation matrices)."""
    for img in image_paths:
        pdf.add_page()
        pdf.image(img, w=width)
        pdf.ln(4)


def add_images_two_per_page(pdf: ReportPDF, image_paths: list[str], width: float = 190):
    """Two images per page, stacked vertically."""
    for i, img in enumerate(image_paths):
        if i % 2 == 0:
            pdf.add_page()
        pdf.image(img, w=width)
        pdf.ln(5)


def add_images_four_per_page(pdf: ReportPDF, image_paths: list[str], width: float = 93):
    """
    Four images per page in a 2x2 grid (good for temporal trajectories).
    """
    page_w = 190
    gap    = 4
    per_row = 2
    width   = (page_w - gap) / per_row

    count = 0
    for img in image_paths:
        if count % 4 == 0:
            pdf.add_page()
        col = count % per_row
        x   = pdf.l_margin + col * (width + gap)
        pdf.image(img, x=x, w=width)
        if col == 1:
            pdf.ln(2)
        count += 1
    pdf.ln(4)

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

