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
from eval.metrics_longitudinal import (
    calculate_longitudinal_metrics,
    compute_visit_position_metrics,
)
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
    plot_visit_position_timing,
    plot_last_visit_vs_d3fup,
    compute_last_visit_vs_d3fup_metrics,
)


# ======================================================
# MAIN
# ======================================================

def main(
    real_path: str,
    synth_path: str,
    config_path: str,
    output_path: str,
    d3fup_col: str = "D3_fup",   # column name for declared follow-up duration
):

    # -- 1. Load data -----------------------------------
    print("[1/11] Loading data...")
    real  = pd.read_excel(real_path)
    synth = pd.read_excel(synth_path)

    time_cfg, variables, _ = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)

    num_vars      = [v.name for v in data_cfg.static_cont]  + [v.name for v in data_cfg.temporal_cont]
    cat_vars      = [v.name for v in data_cfg.static_cat]   + [v.name for v in data_cfg.temporal_cat]
    temporal_vars = [v.name for v in data_cfg.temporal_cont]

    time_col    = time_cfg.visit_column   # "MONTHS_FROM_BASELINE"
    patient_col = "RECORD_ID"

    plot_dir = make_plot_dir(os.path.join(output_path, "plots"))
    os.makedirs(output_path, exist_ok=True)

    pdf = ReportPDF()

    # -- 2. Compute metrics ------------------------------
    print("[2/11] Computing metrics...")

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

    # Per-visit-position timing metrics (new)
    pos_metrics = compute_visit_position_metrics(
        real, synth, time_col, patient_col
    )

    # Last-visit vs D3_fup metrics (new)
    fup_metrics = compute_last_visit_vs_d3fup_metrics(
        real, synth, time_col, patient_col, d3fup_col
    )

    # -- 3. Executive Summary ----------------------------
    print("[3/11] Building executive summary...")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "Executive Summary: Synthetic Data Fidelity", ln=True, align="C")
    pdf.ln(5)
    pdf.add_metrics_table(dist_metrics,  "STATISTICAL FIDELITY METRICS")
    pdf.add_metrics_table(priv_metrics,  "PRIVACY & DISTANCE METRICS (DCR)")
    pdf.add_metrics_table(long_metrics,  "LONGITUDINAL SIMILARITY METRICS")
    if pos_metrics:
        pdf.add_metrics_table(pos_metrics, "VISIT-POSITION TIMING METRICS")
    if fup_metrics:
        pdf.add_metrics_table(fup_metrics, "LAST-VISIT vs D3_FUP ALIGNMENT METRICS")
    pdf.add_note(
        "KS distance measures distributional divergence (0 = identical). "
        "Wasserstein Distance: minimum transport cost (lower = better). "
        "Jensen-Shannon Divergence: symmetric, bounded [0-1] (0 = identical). "
        "PCA Centroid Similarity and Distribution Overlap: normalised to [0-1] (1 = perfect overlap). "
        "Correlation Distance: MAE between correlation matrices. "
        "DCR monitors real-data copying risk (higher = more private). "
        "Visit-position KS: distributional distance of absolute visit time at each sequential position "
        "(position 1 should always be ~0, positions 2+ reveal timing compression). "
        "Last-visit coverage: ratio of last recorded visit to declared D3_fup "
        "(1.0 = last visit perfectly coincides with end of follow-up)."
    )

    # -- 4. Numerical distributions ----------------------
    print("[4/11] Plotting numerical distributions...")
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
        print("[5/11] Plotting categorical distributions...")
        pdf.section("Categorical Distributions (Frequency Analysis)")
        pdf.add_note(
            "Bar charts compare category proportions. "
            "Cramér's V measures association between real and synthetic frequency distributions."
        )
        for img in plot_categorical_grid(real, synth, cat_vars, plot_dir):
            pdf.image(img, w=190)
            pdf.ln(4)

    # -- 6. Correlation matrices -------------------------
    print("[6/11] Plotting correlation matrices...")
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
        img_corr_cat = plot_correlation_comparison(real, synth, cat_vars, "Categorical", plot_dir)
        if img_corr_cat:
            pdf.image(img_corr_cat, w=190)
            pdf.ln(6)

    # -- 7. PCA shared space ------------------------------
    print("[7/11] Plotting PCA...")
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
    print("[8/11] Plotting temporal trajectories...")
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

    # -- 9. Longitudinal dynamics -------------------------
    print("[9/11] Plotting longitudinal dynamics...")
    pdf.section("Longitudinal Dynamics Analysis")

    pdf.add_metrics_table(long_metrics, "LONGITUDINAL SIMILARITY METRICS")
    pdf.add_note(
        "Patient slopes (linear trend per patient), within-patient variance (intra-individual "
        "variability), and autocorrelation (temporal dependency between consecutive visits) "
        "are computed per patient and their distributions compared with KS test. "
        "Lower KS = more similar dynamics."
    )

    # Visit count & timing (existing)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Visit Structure", ln=True)

    img_vd = plot_visit_distribution(real, synth, patient_col, plot_dir)
    img_vt = plot_visit_timing(real, synth, time_col, plot_dir)

    pdf.add_page()
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Visit Count & Timing Distributions", ln=True)
    pdf.image(img_vd, w=190)
    pdf.ln(4)
    pdf.image(img_vt, w=190)
    pdf.ln(6)

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
            "POISE responder: ALP <= 2 AND BIL <= 1 at month 12 (+/-1 month). "
            "KM curves show event-free survival stratified by responder status."
        )
        pdf.image(img_km, w=160)

    # -- 10. Visit-position timing analysis [NEW] --------
    print("[10/11] Plotting visit-position timing analysis...")
    pdf.section("Visit-Position Timing Analysis (NEW)")
    pdf.add_note(
        "For each sequential visit position 1..N (N = median synthetic visit count), "
        "the distribution of absolute visit time is compared between real and synthetic patients. "
        "In PBC/UDCA data the expected schedule is: position 1 = 0 months (baseline), "
        "position 2 ~= 6 months, position 3 ~= 12 months, then approximately yearly. "
        "High KS at early positions (2-3) indicates timing compression: the model is placing "
        "visits too close to baseline. The summary panel shows median ± IQR across positions; "
        "the KS bar chart (bottom) color-codes each position: green < 0.15 (good), "
        "orange < 0.30 (acceptable), red >= 0.30 (poor)."
    )

    if pos_metrics:
        pdf.add_metrics_table(pos_metrics, "VISIT-POSITION TIMING METRICS")

    pos_imgs = plot_visit_position_timing(
        real, synth,
        time_col=time_col,
        patient_col=patient_col,
        outdir=plot_dir,
    )
    if pos_imgs:
        # First image is always the summary (median+IQR + KS bars) -- full width
        pdf.image(pos_imgs[0], w=190)
        pdf.ln(4)
        # Remaining are per-position KDE grids
        if len(pos_imgs) > 1:
            pdf.add_page()
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "Per-position KDE grids", ln=True)
            pdf.add_note(
                "Each panel: KDE of absolute visit time at that sequential position. "
                "Median real and synthetic visit time are annotated. "
                "For PBC: position 1 should peak at 0, position 2 near 6 months, "
                "position 3 near 12 months, subsequent positions near multiples of ~12 months."
            )
            for img in pos_imgs[1:]:
                pdf.image(img, w=190)
                pdf.ln(4)

    # -- 11. Last-visit vs D3_fup analysis [NEW] ---------
    print("[11/11] Plotting last-visit vs D3_fup analysis...")
    pdf.section(f"Last Visit vs {d3fup_col}: Temporal Alignment Diagnostic (NEW)")
    pdf.add_note(
        f"The time of the last recorded visit (last {time_col} per patient) should equal "
        f"{d3fup_col} (declared total follow-up). Discrepancies reveal temporal compression "
        f"in the synthetic data: if the model underestimates inter-visit intervals, "
        f"all visits are shifted toward t=0 and the last visit falls well short of {d3fup_col}. "
        f"'Coverage' = last_visit_time / {d3fup_col}: 1.0 is ideal. "
        f"The scatter plots (top) show the alignment per patient. "
        f"The gap KDE (middle) shows the distribution of ({d3fup_col} - last_visit): "
        f"a peak at 0 is ideal; a positive shift means timing is compressed. "
        f"The decile plots (bottom) reveal whether the gap is uniform across follow-up lengths "
        f"or concentrated in long-follow-up patients."
    )

    if fup_metrics:
        pdf.add_metrics_table(fup_metrics, f"LAST-VISIT vs {d3fup_col.upper()} ALIGNMENT METRICS")

    fup_imgs = plot_last_visit_vs_d3fup(
        real, synth,
        time_col=time_col,
        patient_col=patient_col,
        d3fup_col=d3fup_col,
        outdir=plot_dir,
    )
    if fup_imgs:
        # First image: main 3-panel figure (scatter + gap KDE + last-visit KDE)
        pdf.image(fup_imgs[0], w=190)
        pdf.ln(4)
        # Remaining: per-decile coverage plots for real and synthetic
        if len(fup_imgs) > 1:
            pdf.add_page()
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, f"Coverage by {d3fup_col} Decile", ln=True)
            pdf.add_note(
                f"Patients sorted by {d3fup_col} and split into deciles. "
                f"Coverage (last_visit / {d3fup_col}) is plotted per decile. "
                f"A well-calibrated model should show coverage ~= 1.0 across all deciles. "
                f"Coverage dropping for longer follow-ups indicates that the timing "
                f"compression worsens with follow-up duration."
            )
            for img in fup_imgs[1:]:
                pdf.image(img, w=170)
                pdf.ln(4)

    # -- Save --------------------------------------------
    out_file = os.path.join(output_path, "Synthetic_Data_Validation_Report.pdf")
    pdf.output(out_file)
    print(f"\n[OK]  Report saved: {out_file}")


# ======================================================
if __name__ == "__main__":
    OUTPUT_PATH = "output/exp_5"

    main(
        real_path="PBC_UDCA_long_strat.xlsx",
        synth_path=f"{OUTPUT_PATH}/synthetic_data.xlsx",
        config_path="config/data_config.json",
        output_path=OUTPUT_PATH,
        d3fup_col="D3_fup",
    )