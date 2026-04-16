# main_eval.py  
# Entry point for synthetic data validation report.
#
# Struttura sezioni:
#   1. Load data + config
#   2. Compute metrics
#   3. Executive Summary (radar + score tables + DCR, no raw metrics)
#   4. Distribuzioni Numeriche (KDE)
#   5. Distribuzioni Categoriche (barre affiancate, label string)
#   6. Matrici di Correlazione
#   7. PCA Shared Space
#   8. UMAP Projection
#   9. Traiettorie Temporali
#  10. Variable-by-visit
#  11. Analisi Dinamiche Longitudinali
#  12. Timing per Posizione di Visita
#  13. Last Visit vs t_FUP
#  14. Kaplan-Meier Overall + POISE
#
# Attenzione:
#   - real = imputato (MICE/KNN) dal Preprocessor, troncato a max_len
#   - real_raw = solo per fup/TFS/KM (t_FUP non cambia con imputazione)
#   - Metriche longitudinali su real troncato (confronto equo con synth)
#   - Categoriche: cast int + decode via inverse_maps
# ======================================================

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config.config_loader import load_config, build_data_config
from sklearn.experimental import enable_iterative_imputer  # Necessario per MICE
from sklearn.impute import IterativeImputer, KNNImputer

from processing.processor import Preprocessor

from eval.report_pdf import make_plot_dir, ReportPDF, plot_scores_dashboard
'''
from eval.metrics import (
    calculate_similarity_metrics,
    calculate_correlation_distance,
    calculate_categorical_correlation_distance,
    calculate_pca_overlap_score,
    privacy_metrics,
    calculate_longitudinal_metrics,
    compute_visit_position_metrics,
    compute_temporal_fidelity_score,
    compute_statistical_fidelity_score,
    compute_longitudinal_fidelity_score,
    compute_temporal_coherence_score,
    plot_umap,
    plot_km_overall,
    plot_variable_by_visit,
    plot_numeric_grid,
    plot_categorical_grid,
    plot_correlation_comparison,
    plot_pca_shared_space,
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
'''
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
    compute_temporal_fidelity_score,
)
from eval.metrics_scores import (
    compute_statistical_fidelity_score,
    compute_longitudinal_fidelity_score,
    compute_temporal_coherence_score,
)
from eval.plots_scores import (
    plot_scores_dashboard,
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
    plot_umap,
    plot_km_overall,
    plot_variable_by_visit,
)

_UNICODE_REPLACEMENTS = {
    '\u2014': '--', '\u2013': '-',
    '\u2264': '<=', '\u2265': '>=', '\u2260': '!=',
    '\u00b1': '+/-', '\u00b2': '^2', '\u00b5': 'u',
    '\u03b1': 'alpha', '\u03bb': 'lambda',
    '\u2192': '->', '\u2190': '<-', '\u2022': '-',
    '\u2248': '~=', '\u2019': "'", '\u2018': "'",
    '\u201c': '"', '\u201d': '"',
}

def _safe(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ch, rep in _UNICODE_REPLACEMENTS.items():
        text = text.replace(ch, rep)
    return text.encode('latin-1', errors='replace').decode('latin-1')


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}" if not np.isnan(float(v)) else "N/A"
    except Exception:
        return "N/A"


def apply_custom_imputation(df, num_vars, cat_vars):
    df_imputed = df.copy()
    
    # 1. MICE per variabili continue
    if num_vars:
        print(f"       -> Imputazione MICE su: {num_vars}")
        # Riduciamo max_iter per evitare che resti appeso se non converge perfettamente
        mice_imputer = IterativeImputer(random_state=42, max_iter=5, tol=1e-3)
        df_imputed[num_vars] = mice_imputer.fit_transform(df_imputed[num_vars])
    
    # 2. KNN per variabili categoriche
    if cat_vars:
        print(f"       -> Imputazione KNN su: {cat_vars}")
        temp_knn = pd.DataFrame(index=df_imputed.index)
        storage_mappers = {} 
        
        for col in cat_vars:
            # Proviamo a convertire in numerico. Se fallisce, è una stringa/categorica vera.
            try:
                # Se la colonna è già numerica o convertibile senza errori
                series_float = pd.to_numeric(df_imputed[col], errors='raise')
                temp_knn[col] = series_float.astype(float)
            except (ValueError, TypeError):
                # Se entra qui, significa che contiene stringhe come 'PBC0001'
                print(f"          [Info] Codifica stringhe per colonna: {col}")
                codes, uniques = pd.factorize(df_imputed[col])
                # Convertiamo i -1 (NaN di factorize) in np.nan reali per il KNN
                temp_knn[col] = np.where(codes == -1, np.nan, codes.astype(float))
                storage_mappers[col] = uniques 
        
        # Eseguiamo KNN
        knn_imputer = KNNImputer(n_neighbors=5)
        imputed_array = knn_imputer.fit_transform(temp_knn)
        imputed_df = pd.DataFrame(imputed_array, columns=cat_vars, index=df_imputed.index)
        
        for col in cat_vars:
            # Arrotondiamo sempre per le categoriche
            vals = imputed_df[col]
            vals_rounded = np.floor(vals + 0.5).astype(int)     #imputed_df[col].round().astype(int)
            
            if col in storage_mappers:
                # Ripristiniamo le stringhe originali ('PBC0001', ecc.)
                uniques = storage_mappers[col]
                # Clip per sicurezza se l'arrotondamento finisce fuori indice
                safe_indices = np.clip(vals_rounded, 0, len(uniques) - 1)
                df_imputed[col] = uniques[safe_indices]
            else:
                # Era numerica, salviamo come int pulito (niente 0.0)
                df_imputed[col] = vals_rounded
                
    return df_imputed

# SafePDF

class _SafePDF(ReportPDF):
    def cell(self, w=0, h=0, txt="", border=0, ln=0, align="", fill=False, link=""):
        return super().cell(w, h, _safe(str(txt)), border, ln, align, fill, link)
    def multi_cell(self, w, h, txt="", border=0, align="J", fill=False):
        return super().multi_cell(w, h, _safe(str(txt)), border, align, fill)
    def write(self, h, txt="", link=""):
        return super().write(h, _safe(str(txt)), link)


# MAIN --------------------------------------
def main(
    real_path:         str,
    synth_path:        str,
    config_path:       str,
    output_path:       str,
    preprocessor_path: str | None = None,
):
    # ── 1. Load data + config ─────────────────────────────────────────
    print("[1/14] Caricamento dati e config...")
    real_raw = pd.read_excel(real_path)
    synth    = pd.read_excel(synth_path)

    time_cfg, variables, _, prep_cfg = load_config(
        data_path  = config_path, model_path = "config/model_config.json") #load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)

    num_vars      = [v.name for v in data_cfg.static_cont  + data_cfg.temporal_cont]
    cat_vars      = [v.name for v in data_cfg.static_cat   + data_cfg.temporal_cat]
    temporal_vars = [v.name for v in data_cfg.temporal_cont]

    time_col    = data_cfg.time_col
    patient_col = data_cfg.patient_id_col
    fup_col     = data_cfg.fup_col
    max_len     = getattr(time_cfg, "max_visits", 12)

    # Ordiniamo per paziente e tempo e prendiamo solo le prime max_len visite
    print(f"     Troncamento dataset reale a max {max_len} visite...")
    real = real_raw.sort_values([patient_col, time_col])
    real = real.groupby(patient_col).head(max_len).reset_index(drop=True)

    # --- IMPUTAZIONE DIFFERENZIATA ---
    print("     Esecuzione imputazione (MICE cont, KNN cat) su dataset reale...")
    num_ok  = [v for v in num_vars      if v in real.columns and v in synth.columns]
    cat_ok  = [v for v in cat_vars      if v in real.columns and v in synth.columns]
    #real = apply_custom_imputation(real, num_ok, cat_ok)
    #real = real_raw
    #processor = Preprocessor(data_cfg, embedding_configs = prep_cfg.emb_vars, log_vars = prep_cfg.log_vars)
    
    processor = torch.load("processing/preprocessor_fitted.pt", weights_only=False)['preprocessor']  # ← CARICA fitted
    inverse_maps = processor.inverse_maps 
    print(f"Loaded inverse_maps keys: {list(inverse_maps.keys())}")

    real_df = processor._force_types(real)
    real = processor._impute(real_df)
    
    #for v in ['SEX', 'DEATH']:  # binari sospetti
    #    print(f"{v} PRE-impute: {real_df[v].value_counts(normalize=True).sort_index()}")
    #    print(f"{v} POST-impute: {real[v].value_counts(normalize=True).sort_index()}")
    #    print(f"  synth: {synth[v].value_counts(normalize=True).sort_index()}")
    #    print(f"  inverse_maps[{v}]: {processor.inverse_maps.get(v, 'None')}")

    '''
    import json 

    with open(config_path, 'r') as f:
        full_config = json.load(f)

    # Estraiamo solo la parte delle variabili categoriche
    cat_config = full_config.get("baseline", {}).get("categorical", {})

    inverse_maps = {}

    for col_name, info in cat_config.items():
        mapping = info.get("mapping", {})
        
        if not mapping:
            continue
            
        # Creiamo il mapping inverso standard dal JSON {Valore: "Etichetta"}
        inv_map = {int(v): str(k) for k, v in mapping.items()}
        
        # Controlliamo i dati reali per vedere da dove partono veramente
        if col_name in real.columns:
            actual_min = real[col_name].min()
            config_min = min(inv_map.keys())
            
            # FIX: Se il config parte da 1 ma i dati partono da 0
            if config_min == 1 and actual_min == 0:
                inv_map = {k - 1: v for k, v in inv_map.items()}
                print(f" [!] Allineamento: {col_name} scalata da 1-based a 0-based")
            
            # FIX: Se il config parte da 0 ma i dati partono da 1 (raro ma possibile)
            elif config_min == 0 and actual_min == 1:
                inv_map = {k + 1: v for k, v in inv_map.items()}
                print(f" [!] Allineamento: {col_name} scalata da 0-based a 1-based")
            
            if not set(real[col_name].unique()).issubset(set(inv_map.keys())):
                print(f"[WARNING] {col_name} valori fuori mapping:", set(real[col_name].unique()))
        inverse_maps[col_name] = inv_map
    '''
    print(f"  Pazienti — real: {real[patient_col].nunique()}  "
          f"synth: {synth[patient_col].nunique()}")

    plot_dir = make_plot_dir(os.path.join(output_path, "plots"))
    os.makedirs(output_path, exist_ok=True)
    pdf = _SafePDF()

    # ── 2. Compute metrics ────────────────────────────────────────────
    print("[2/14] Calcolo metriche...")

    temp_ok = [v for v in temporal_vars if v in real.columns and v in synth.columns]

    dist_metrics = calculate_similarity_metrics(real, synth, num_ok, cat_ok)
    dist_metrics["Correlation Distance - Continuous (MAE Pearson)"] = \
        calculate_correlation_distance(real, synth, num_ok)
    dist_metrics["Correlation Distance - Categorical (MAE Cramer's V)"] = \
        calculate_categorical_correlation_distance(real, synth, cat_ok)
    dist_metrics.update(calculate_pca_overlap_score(real, synth, num_ok))

    priv_metrics = privacy_metrics(real, synth, num_ok)

    # Longitudinale su real TRONCATO (= real, gia' troncato)
    long_metrics = calculate_longitudinal_metrics(real, synth, temp_ok, time_col, patient_col)
    pos_metrics  = compute_visit_position_metrics(real, synth, time_col, patient_col)

    # TFS + fup: usa real_raw (t_FUP variabile statica, non cambia con imputazione)
    fup_metrics = compute_last_visit_vs_d3fup_metrics(
        real_raw, synth, time_col, patient_col, fup_col)
    tfs = compute_temporal_fidelity_score(
        real_raw, synth, temp_ok, time_col, patient_col, fup_col=fup_col)

    sfs = compute_statistical_fidelity_score(dist_metrics)

    print("  Computing TCS (may take a moment)...")
    tcs_dict = compute_temporal_coherence_score(real, synth, temp_ok, time_col, patient_col)

    iv_r = {
        "Mean inter-visit interval": long_metrics.get("Mean inter-visit interval - Real", float("nan")),
        "Std inter-visit interval":  long_metrics.get("Std inter-visit interval - Real",  float("nan")),
    }
    iv_s = {
        "Mean inter-visit interval": long_metrics.get("Mean inter-visit interval - Synthetic", float("nan")),
        "Std inter-visit interval":  long_metrics.get("Std inter-visit interval - Synthetic",  float("nan")),
    }
    lfs = compute_longitudinal_fidelity_score(
        long_metrics, pos_metrics, tcs_dict,
        interval_stats_real=iv_r, interval_stats_synth=iv_s)

    print(f"  SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")

    # ── 3. Executive Summary ──────────────────────────────────────────
    print("[3/14] Executive Summary...")
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "Synthetic Data Validation Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6,
             f"Config: {os.path.basename(config_path)}  |  "
             f"Real: {os.path.basename(real_path)}  |  "
             f"Synthetic: {os.path.basename(synth_path)}",
             ln=True, align="C")
    pdf.ln(3)

    # Score banner
    pdf.set_fill_color(230, 245, 255)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(63, 9, f"SFS: {_fmt(sfs.overall)}  {sfs.grade[:3]}", ln=False, fill=True, align="C")
    pdf.set_fill_color(255, 243, 205)
    pdf.cell(63, 9, f"LFS: {_fmt(lfs.overall)}  {lfs.grade[:3]}", ln=False, fill=True, align="C")
    pdf.set_fill_color(205, 230, 255)
    pdf.cell(64, 9, f"TFS: {_fmt(tfs.overall)}  {tfs.grade[:3]}", ln=True,  fill=True, align="C")
    pdf.ln(3)

    dashboard_path = plot_scores_dashboard(sfs, lfs, tfs, plot_dir)
    pdf.image(dashboard_path, x=10, w=190)
    pdf.ln(4)

    pdf.add_metrics_table(sfs.to_dict(), "STATISTICAL FIDELITY SCORE (SFS) -- dettaglio")
    pdf.add_metrics_table(lfs.to_dict(), "LONGITUDINAL FIDELITY SCORE (LFS) -- dettaglio")
    pdf.add_metrics_table(tcs_dict,      "TEMPORAL COHERENCE SCORE (TCS) -- dettaglio")
    pdf.add_metrics_table(tfs.to_dict(), "TEMPORAL FIDELITY SCORE (TFS) -- dettaglio")
    pdf.ln(2)
    pdf.add_metrics_table(priv_metrics,  "PRIVACY & DISTANCE METRICS (DCR)")

    pdf.add_note(
        f"SFS (6 metriche): fedelta' statistica [0-1]. "
        f"LFS (5 metriche): fedelta' longitudinale + coerenza [0-1]. "
        f"TFS: fedelta' timing + t_FUP coverage [0-1]. "
        f"Baseline: real IMPUTATO (MICE/KNN) troncato a max={max_len} visite. "
        f"Long. metrics su real troncato (confronto equo con synth max={max_len} visite). "
        f"Score: 1=perfetto, 0=pessimo. Lower-better -> 1-score. "
        f"Interval diff: 1/(1+rel_diff) senza clipping. "
        f"DCR Fraction<1=1.0: normale per dati clinici ad alta correlazione."
    )

    # ── 4. Distribuzioni numeriche ────────────────────────────────────
    print("[4/14] Distribuzioni numeriche...")
    #pdf.add_page()
    pdf.section("Distribuzioni Numeriche (KDE)")
    pdf.add_note(
        "Real = imputato (MICE/KNN). "
        "KS e p-value annotati (p<0.05 = divergenza significativa)."
    )
    for img in plot_numeric_grid(real, synth, num_ok, plot_dir):
        pdf.image(img, x=10, w=190)

    # ── 5. Distribuzioni categoriche ──────────────────────────────────
    if cat_ok:
        print("[5/14] Distribuzioni categoriche...")
        #pdf.add_page()
        pdf.section("Distribuzioni Categoriche (Frequenze)")
        pdf.add_note(
            "Barre affiancate. "
            "Label originali decodificate via inverse_maps. "
            "Cramer's V: 0=nessuna associazione, 1=identiche."
        )
        # Aggiungi prima di plot_categorical_grid:
        #for v in cat_ok:
        #    print(f"{v}: real_unique={sorted(real[v].unique())}, synth_unique={sorted(synth[v].unique())}")
        #    print(f"  inverse_maps: {inverse_maps.get(v, 'None')}")

        for img in plot_categorical_grid(
                real, synth, cat_ok, plot_dir,inverse_maps=inverse_maps):
            pdf.image(img, x=10, w=190)

    #v = "SEX"  # esempio
    #print("REAL raw:", sorted(real[v].unique()))
    #print("SYNTH raw:", sorted(synth[v].unique()))
    #print("MAP:", inverse_maps[v])
    # Frequenze su raw (senza decode)
    #print("REAL freq raw:\n", real[v].value_counts(normalize=True))
    #print("SYNTH freq raw:\n", synth[v].value_counts(normalize=True))

    # ── 6. Correlazioni ───────────────────────────────────────────────
    print("[6/14] Matrici di correlazione...")
    #pdf.add_page()
    pdf.section("Matrici di Correlazione")
    pdf.add_note(
        "Sinistra=reale, Centro=sintetico, Destra=|differenza|. "
        "Continue: Pearson r."
    )
    img = plot_correlation_comparison(real, synth, num_ok, "Continue", plot_dir)
    if img:
        pdf.image(img, w=190)
        pdf.ln(6)
    if cat_ok:
        img = plot_correlation_comparison(real, synth, cat_ok, "Categoriche", plot_dir)
        if img:
            pdf.image(img, w=190)
            pdf.ln(6)

    # ── 7. PCA ────────────────────────────────────────────────────────
    print("[7/14] PCA...")
    #pdf.add_page()
    pdf.section("Analisi Multivariata (PCA Shared Space)")
    pdf.add_note("PCA fittato sui reali; sintetici proiettati. Croci=centroidi.")
    img = plot_pca_shared_space(real, synth, num_ok, plot_dir)
    if img:
        pdf.image(img, w=140)

    # ── 8. UMAP ───────────────────────────────────────────────────────
    print("[8/14] UMAP...")
    #pdf.add_page()
    pdf.section("Analisi Multivariata -- UMAP Projection")
    pdf.add_note(
        "UMAP proietta in 2D preservando struttura locale. "
        "Riduttore fittato sui reali; sintetici proiettati. "
        "Nuvole sovrapposte = buona fedelta'. Richiede: pip install umap-learn"
    )
    img = plot_umap(real, synth, num_ok, plot_dir)
    if img:
        pdf.image(img, w=155)
    else:
        pdf.add_note("[SKIP] umap-learn non installato.")

    # ── 9. Traiettorie ────────────────────────────────────────────────
    print("[9/14] Traiettorie temporali...")
    #pdf.add_page()
    pdf.section("Traiettorie Temporali (Media +/- 95% CI)")
    pdf.add_note(
        "Curva media interpolata su griglia comune. "
        "Banda = CI 95% (dove >=5 pazienti)."
    )
    for v in temp_ok:
        img = plot_temporal_trajectory(
            real, synth, v,
            time_col=time_col, patient_col=patient_col,
            max_time=None, n_grid=60, min_patients=5, outdir=plot_dir)
        if img:
            pdf.image(img, w=190)
            pdf.ln(4)

    # ── 10. Variable-by-visit ─────────────────────────────────────────
    print("[10/14] Variable-by-visit...")
    #pdf.add_page()
    pdf.section("Variabili Temporali per Bin di Visita")
    pdf.add_note(
        "Visite in bin di tempo equidistanti. "
        "Boxplot affiancati + KS per bin. "
        "Verde KS<0.15, arancio KS<0.30, rosso KS>=0.30. 3 variabili/pag."
    )
    for img in plot_variable_by_visit(
            real, synth, temp_ok, time_col, patient_col, plot_dir,
            n_visit_bins=6, max_vars_per_page=3):
        pdf.image(img, x=10, w=190)

    # ── 11. Dinamiche longitudinali ───────────────────────────────────
    print("[11/14] Dinamiche longitudinali...")
    #pdf.add_page()
    pdf.section("Analisi Dinamiche Longitudinali")
    pdf.add_metrics_table(long_metrics, "METRICHE DI SIMILARITA' LONGITUDINALE")
    pdf.add_note(
        "Pendenze, varianza intra-paziente e autocorrelazione confrontati con KS. "
        "KS basso = dinamiche piu' simili. "
        "Calcolato su real TRONCATO a max_len (confronto equo)."
    )

    img_vd = plot_visit_distribution(real, synth, patient_col, plot_dir)
    img_vt = plot_visit_timing(real, synth, time_col, plot_dir)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 8, "Distribuzione Visite e Timing", ln=True)
    pdf.image(img_vd, w=190)
    pdf.ln(4)
    pdf.image(img_vt, w=190)
    pdf.ln(6)

    img_tcc = plot_temporal_cross_correlation(real, synth, temp_ok, time_col, plot_dir)
    if img_tcc:
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Correlazioni Temporali (pooled)", ln=True)
        pdf.add_note("Correlazioni di Pearson pairwise su tutte le visite.")
        pdf.image(img_tcc, w=190)

    slope_imgs = plot_slope_grid(real, synth, temp_ok, time_col, patient_col, plot_dir)
    if slope_imgs:
        pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Distribuzione Pendenze per Paziente", ln=True)
        pdf.add_note("KDE pendenza lineare per paziente per ogni variabile temporale.")
        for img in slope_imgs:
            pdf.image(img, w=190)

    var_imgs = plot_variance_grid(real, synth, temp_ok, patient_col, plot_dir)
    if var_imgs:
        #pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Varianza Intra-paziente", ln=True)
        pdf.add_note("KDE varianza intra-paziente per ogni variabile temporale.")
        for img in var_imgs:
            pdf.image(img, w=190)

    ac_imgs = plot_autocorrelation_grid(real, synth, temp_ok, time_col, patient_col, plot_dir)
    if ac_imgs:
        #pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Autocorrelazione Lag-1", ln=True)
        pdf.add_note("KDE autocorrelazione tra visite consecutive per paziente.")
        for img in ac_imgs:
            pdf.image(img, w=190)

    # ── 12. Timing per posizione ──────────────────────────────────────
    print("[12/14] Timing per posizione di visita...")
    #pdf.add_page()
    pdf.section("Timing per Posizione di Visita  [componente TFS]")
    pdf.add_note(
        "Distribuzione tempo assoluto di visita per posizione sequenziale. "
        "PBC atteso: pos1=0, pos2~6, pos3~12 mesi. "
        "Verde KS<0.15, arancio<0.30, rosso>=0.30."
    )
    if pos_metrics:
        pdf.add_metrics_table(pos_metrics, "METRICHE TIMING PER POSIZIONE")

    pos_imgs = plot_visit_position_timing(
        real, synth, time_col=time_col, patient_col=patient_col, outdir=plot_dir)
    if pos_imgs:
        pdf.image(pos_imgs[0], w=190)
        pdf.ln(4)
        if len(pos_imgs) > 1:
            #pdf.add_page()
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, "KDE per posizione di visita", ln=True)
            pdf.add_note(
                "Pos 1 deve piccare a 0, pos 2 ~6 mesi, pos 3 ~12 mesi.")
            for img in pos_imgs[1:]:
                pdf.image(img, w=190)
                pdf.ln(4)

    # ── 13. Last visit vs t_FUP ───────────────────────────────────────
    print(f"[13/14] Last-visit vs {fup_col}...")
    #pdf.add_page()
    pdf.section(f"Last Visit vs {fup_col}: Allineamento Temporale  [componente TFS]")
    pdf.add_note(
        f"Coverage = last_visit / {fup_col}: ideale = 1.0. "
        f"Gap > 0 = compressione temporale. "
        f"Usa real_raw (t_FUP non cambia con imputazione)."
    )
    if fup_metrics:
        pdf.add_metrics_table(fup_metrics, f"METRICHE ALLINEAMENTO {fup_col.upper()}")

    fup_imgs = plot_last_visit_vs_d3fup(
        real_raw, synth, time_col=time_col, patient_col=patient_col,
        d3fup_col=fup_col, outdir=plot_dir)
    if fup_imgs:
        pdf.image(fup_imgs[0], w=190)
        pdf.ln(4)
        if len(fup_imgs) > 1:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 8, f"Coverage per decile di {fup_col}", ln=True)
            pdf.add_note("Coverage=1.0 = allineamento perfetto con t_FUP.")
            for img in fup_imgs[1:]:
                pdf.image(img, w=170)
                pdf.ln(4)

    # ── 14. Kaplan-Meier ──────────────────────────────────────────────
    print("[14/14] KM overall + log-rank...")
    #pdf.add_page()
    pdf.section("Kaplan-Meier Overall (non stratificato) + Log-rank Test")
    pdf.add_note(
        f"KM per Real vs Synthetic usando {fup_col}. "
        "Log-rank: p>>0.05 = curve compatibili. Richiede: pip install lifelines"
    )
    img = plot_km_overall(
        real_raw, synth, time_col=fup_col, patient_col=patient_col,
        event_col=None, outdir=plot_dir)
    if img:
        pdf.image(img, w=160)
        pdf.ln(4)
    else:
        pdf.add_note("[SKIP] lifelines non installato.")

    img = plot_km_responder(real_raw, synth, time_col, patient_col, plot_dir)
    if img:
        #pdf.add_page()
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Kaplan-Meier: Risposta POISE (ALP<=2 e BIL<=1 a mese 12)", ln=True)
        pdf.add_note(
            "Responder = ALP<=2 E BIL<=1 al mese 12 (+/-1 mese). "
            "Buona sintesi: separazione simile tra curve R e NR.")
        pdf.image(img, w=160)

    # ── Save ──────────────────────────────────────────────────────────
    out_file = os.path.join(output_path, "Synthetic_Data_Validation_Report.pdf")
    pdf.output(out_file)
    print(f"\n[OK] Report salvato: {out_file}")
    print(f"     SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")
    return {"sfs": sfs, "lfs": lfs, "tfs": tfs}


# ======================================================
if __name__ == "__main__":
    OUTPUT_PATH = "output/exp_5"
    CONFIG_PATH = "config/data_config2.json"

    main(
        real_path         = "DGAN_PBC.xlsx",
        synth_path        = f"{OUTPUT_PATH}/synthetic_data.xlsx",
        config_path       = CONFIG_PATH,
        output_path       = OUTPUT_PATH,
        preprocessor_path = f"{OUTPUT_PATH}/preprocessor.pt",
    )