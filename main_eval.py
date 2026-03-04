#!/usr/bin/env python3
# ======================================================
# main_eval.py
# Entry point for synthetic data validation report.
#
# Usage:
#   python main_eval.py
#
# Produces: <output_path>/Synthetic_Data_Validation_Report.pdf
# ======================================================

import os
import pandas as pd

from config.config_loader import load_config, build_data_config

from eval.config import make_plot_dir
from eval.report_pdf import (
    ReportPDF,
    add_images_grid,
    add_images_full_width,
    add_images_four_per_page,
)
from eval.metrics_distribution import (
    calculate_similarity_metrics,
    calculate_correlation_distance,
    calculate_categorical_correlation_distance,
    calculate_pca_overlap_score,
    privacy_metrics,
)
from eval.metrics_longitudinal import calculate_longitudinal_metrics
from eval.plots_distribution import (
    plot_numeric_grid,
    plot_categorical_grid,
    plot_correlation_comparison,
    plot_pca_shared_space,
)
from eval.plots_longitudinal import (
    plot_temporal_trajectory,
    plot_slope_grid,
    plot_variance_grid,
    plot_autocorrelation_grid,
    plot_visit_distribution,
    plot_visit_timing,
    plot_temporal_cross_correlation,
    plot_km_responder,
)


# ======================================================
# MAIN
# ======================================================

def main(real_path: str, synth_path: str, config_path: str, output_path: str):

    # -- 1. Load data -----------------------------------
    print("[1/9] Loading data...")
    real  = pd.read_excel(real_path)
    synth = pd.read_excel(synth_path)

    time_cfg, variables, _ = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)

    num_vars     = [v.name for v in data_cfg.static_cont]  + [v.name for v in data_cfg.temporal_cont]
    cat_vars     = [v.name for v in data_cfg.static_cat]   + [v.name for v in data_cfg.temporal_cat]
    temporal_vars = [v.name for v in data_cfg.temporal_cont]

    time_col    = time_cfg.visit_column
    patient_col = "RECORD_ID"

    plot_dir = make_plot_dir(os.path.join(output_path, "plots"))
    os.makedirs(output_path, exist_ok=True)

    pdf = ReportPDF()

    # -- 2. Compute metrics ------------------------------
    print("[2/9] Computing metrics...")

    dist_metrics = calculate_similarity_metrics(real, synth, num_vars, cat_vars)
    dist_metrics["Correlation Distance - Continuous (MAE Pearson, (lower) better)"] = \
        calculate_correlation_distance(real, synth, num_vars)
    dist_metrics["Correlation Distance - Categorical (MAE Cramér's V, (lower) better)"] = \
        calculate_categorical_correlation_distance(real, synth, cat_vars)
    dist_metrics.update(calculate_pca_overlap_score(real, synth, num_vars))

    priv_metrics = privacy_metrics(real, synth, num_vars)

    long_metrics = calculate_longitudinal_metrics(
        real, synth, temporal_vars, time_col, patient_col
    )

    # -- 3. Executive Summary ----------------------------
    print("[3/9] Building executive summary...")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "Executive Summary: Synthetic Data Fidelity", ln=True, align="C")
    pdf.ln(5)
    pdf.add_metrics_table(dist_metrics,  "STATISTICAL FIDELITY METRICS")
    pdf.add_metrics_table(priv_metrics,  "PRIVACY & DISTANCE METRICS (DCR)")
    pdf.add_note(
        "KS distance measures distributional divergence (0 = identical). "
        "Wasserstein Distance: minimum transport cost ((lower) better). "
        "Jensen-Shannon Divergence: symmetric, bounded [0-1] (0 = identical). "
        "PCA Centroid Similarity and Distribution Overlap: normalised to [0-1] (1 = perfect overlap). "
        "Correlation Distance: MAE between correlation matrices. "
        "DCR monitors real-data copying risk (higher = more private)."
    )

    # -- 4. Numerical distributions ----------------------
    print("[4/9] Plotting numerical distributions...")
    pdf.section("Numerical Distributions (KDE Analysis)")
    pdf.add_note(
        "KDE plots compare marginal distributions of each continuous variable. "
        "KS statistic and p-value are shown per variable (p<0.05 indicates significant divergence)."
    )
    for img in plot_numeric_grid(real, synth, num_vars, plot_dir):
        pdf.image(img, w=190)
        pdf.ln(4)

    # -- 5. Categorical distributions --------------------
    if cat_vars:
        print("[5/9] Plotting categorical distributions...")
        pdf.section("Categorical Distributions (Frequency Analysis)")
        pdf.add_note(
            "Bar charts compare category proportions. "
            "Cramér's V measures association between real and synthetic frequency distributions."
        )
        for img in plot_categorical_grid(real, synth, cat_vars, plot_dir):
            pdf.image(img, w=190)
            pdf.ln(4)

    # -- 6. Correlation matrices -------------------------
    print("[6/9] Plotting correlation matrices...")
    pdf.section("Correlation Matrices Comparison")
    pdf.add_note(
        "Left: real data. Centre: synthetic data. Right: absolute difference |Real - Synthetic|. "
        "Continuous variables use Pearson r; categorical use factorized Pearson as proxy."
    )

    img_corr_num = plot_correlation_comparison(real, synth, num_vars, "Numerical", plot_dir)
    if img_corr_num:
        pdf.image(img_corr_num, w=190)
        pdf.ln(6)

    if cat_vars:
       # pdf.add_page()
        img_corr_cat = plot_correlation_comparison(real, synth, cat_vars, "Categorical", plot_dir)
        if img_corr_cat:
            pdf.image(img_corr_cat, w=190)
            pdf.ln(6)

    # -- 7. PCA shared space ------------------------------
    print("[7/9] Plotting PCA...")
    pdf.section("Multivariate Analysis (PCA Shared Space)")
    pdf.add_note(
        "PCA fitted exclusively on real data; synthetic data is projected onto the same axes. "
        "Crosses mark the centroid of each cloud. "
        "PCA Centroid Similarity and Distribution Overlap are reported in the Executive Summary."
    )
    img_pca = plot_pca_shared_space(real, synth, num_vars, plot_dir)
    if img_pca:
        pdf.image(img_pca, w=140)

    # -- 8. Temporal trajectories -------------------------
    print("[8/9] Plotting temporal trajectories...")
    pdf.section("Temporal Trajectories (Mean +/- 95% CI)")
    pdf.add_note(
        "Each curve shows the mean value over time, interpolated onto a common time grid. "
        "Shaded bands = 95% CI (mean +/- 1.96 x SEM), shown where >=5 patients contribute. "
        "Irregular visit schedules are handled by per-patient linear interpolation."
    )

    traj_imgs = []
    for v in temporal_vars:
        img = plot_temporal_trajectory(
            real, synth, v,
            time_col=time_col, patient_col=patient_col,
            max_time=None, n_grid=60, min_patients=5, outdir=plot_dir,
        )
        if img:
            traj_imgs.append(img)
    
    for img in traj_imgs:
        pdf.image(img, w=190)
        pdf.ln(4)
    # 4 per page, 2x2 layout
    #add_images_four_per_page(pdf, traj_imgs)

    # -- 9. Longitudinal dynamics -------------------------
    print("[9/9] Plotting longitudinal dynamics...")
    pdf.section("Longitudinal Dynamics Analysis")

    # Summary metrics at top of section
    pdf.add_metrics_table(long_metrics, "LONGITUDINAL SIMILARITY METRICS")
    pdf.add_note(
        "Patient slopes (linear trend per patient), within-patient variance (intra-individual variability), "
        "and autocorrelation (temporal dependency between consecutive visits) are computed per patient "
        "and their distributions compared with KS test. Lower KS = more similar dynamics."
    )

    # Visit structure
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Visit Structure", ln=True)
    pdf.set_font("Arial", "", 10)

    img_vd = plot_visit_distribution(real, synth, patient_col, plot_dir)
    img_vt = plot_visit_timing(real, synth, time_col, plot_dir)

    pdf.add_page()
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Visit Count & Timing Distributions", ln=True)
    pdf.image(img_vd, w=190)
    pdf.ln(4)
    pdf.image(img_vt, w=190)
    pdf.ln(6)
    #pdf.image(img_vd, w=92)
    #pdf.image(img_vt, x=pdf.l_margin + 96, y=pdf.get_y() - 80, w=92)
    #pdf.ln(5)

    # Temporal cross-correlation
    img_tcc = plot_temporal_cross_correlation(real, synth, temporal_vars, time_col, plot_dir)
    if img_tcc:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Temporal Cross-Correlations (pooled across time)", ln=True)
        pdf.add_note(
            "Pairwise Pearson correlations computed across all visits (pooled). "
            "Shows how inter-variable relationships are preserved in the synthetic data."
        )
        pdf.image(img_tcc, w=190)

    # Slope grids
    slope_imgs = plot_slope_grid(real, synth, temporal_vars, time_col, patient_col, plot_dir)
    if slope_imgs:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Patient Slope Distributions", ln=True)
        pdf.add_note("KDE of per-patient linear trend (slope) for each temporal variable.")
        for img in slope_imgs:
            pdf.image(img, w=190)

    # Variance grids
    var_imgs = plot_variance_grid(real, synth, temporal_vars, patient_col, plot_dir)
    if var_imgs:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Within-Patient Variance Distributions", ln=True)
        pdf.add_note("KDE of intra-patient variance for each temporal variable.")
        for img in var_imgs:
            pdf.image(img, w=190)

    # Autocorrelation grids
    ac_imgs = plot_autocorrelation_grid(real, synth, temporal_vars, time_col, patient_col, plot_dir)
    if ac_imgs:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Autocorrelation Distributions", ln=True)
        pdf.add_note("KDE of lag-1 autocorrelation (consecutive-visit correlation) per patient.")
        for img in ac_imgs:
            pdf.image(img, w=190)

    # Kaplan-Meier
    img_km = plot_km_responder(real, synth, time_col, patient_col, plot_dir)
    if img_km:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Kaplan-Meier: POISE Responder Classification", ln=True)
        pdf.add_note(
            "POISE responder: ALP <= 1.67 AND 1 < BIL < 2 at month 12 (+/-1 month). "
            "KM curves show event-free survival stratified by responder status."
        )
        pdf.image(img_km, w=160)

    # -- Save --------------------------------------------
    out_file = os.path.join(output_path, "Synthetic_Data_Validation_Report.pdf")
    pdf.output(out_file)
    print(f"\n[OK]  Report saved: {out_file}")


# ======================================================
if __name__ == "__main__":
    OUTPUT_PATH = "output/exp_2"

    main(
        real_path="PBC_UDCA_long_strat.xlsx",
        synth_path=f"{OUTPUT_PATH}/synthetic_data.xlsx",
        config_path="config/data_config.json",
        output_path=OUTPUT_PATH,
    )