# eval/report.py
#
# run_validation_report():
#   Orchestratore centrale chiamato da main
#   1. Calcola tutte le metriche (fidelity / utility / privacy)
#   2. Genera tutti i plot
#   3. Assembla il PDF
#
# Il PDF è strutturato in 4 macro-sezioni:
#   Pag. 1 -  Executive Summary (3 radar + utility + privacy)
#   Sez. A -  Fidelity: distribuzioni, correlazioni, PCA, UMAP,
#              traiettorie, varianza, LME, autocorr, visite,
#   Sez. B -  Utility: KM (da spostare e rendere score)
#   Sez. C -  Privacy: DCR/NNDR
# ======================================================

import os
import numpy as np
import pandas as pd
from fpdf import FPDF

from eval.metrics import (
    compute_fidelity_metrics,
    compute_privacy_metrics,
)
from eval.plots import (
    plot_scores_dashboard,
    plot_summary_dashboard,
    plot_fidelity_section,
    plot_privacy_section,
    plot_radar_component,
)

_UNICODE_REPLACEMENTS = {
    '\u2014': '--', '\u2013': '-',
    '\u2264': '<=', '\u2265': '>=', '\u2260': '!=',
    '\u00b1': '+/-', '\u00b2': '^2', '\u00b5': 'u',
    '\u03b1': 'alpha', '\u03bb': 'lambda',
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    '\u2022': '-', '\u2192': '->', '\u2190': '<-',
}

def _safe(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for ch, rep in _UNICODE_REPLACEMENTS.items():
        text = text.replace(ch, rep)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _fmt(v) -> str:
    try:
        f = float(v)
        return f"{f:.3f}" if not np.isnan(f) else "N/A"
    except Exception:
        return "N/A"


def make_plot_dir(path: str = "plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path


class ReportPDF(FPDF):
    def header(self):
        #self.set_font("Arial", "B", 12)
        #self.cell(0, 10, "Synthetic Data Validation Report", ln=True, align="C")
        self.ln(3)

    def section(self, title: str):
        self.add_page()
        self.set_font("Arial", "B", 13)
        self.set_fill_color(220, 230, 245)
        self.cell(0, 11, title, ln=True, fill=True)

    def add_metrics_table(self, metrics_dict: dict, title: str):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)
        for k, v in metrics_dict.items():
            text = f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}"
            self.multi_cell(0, 6, text)
        self.ln(4)

    def add_note(self, text: str):
        self.set_font("Arial", "I", 9)
        self.multi_cell(0, 5, text)
        self.ln(3)


class _SafePDF(ReportPDF):
    def cell(self, w=0, h=0, txt="", border=0, ln=0, align="", fill=False, link=""):
        return super().cell(w, h, _safe(str(txt)), border, ln, align, fill, link)

    def multi_cell(self, w, h, txt="", border=0, align="J", fill=False):
        return super().multi_cell(w, h, _safe(str(txt)), border, align, fill)

    def write(self, h, txt="", link=""):
        return super().write(h, _safe(str(txt)), link)

    def add_metrics_table(self, metrics_dict: dict, title: str):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, _safe(title), ln=True)
        self.set_font("Arial", "", 10)
        for k, v in metrics_dict.items():
            text = f"  - {_safe(str(k))}: {_safe(str(v))}"
            self.multi_cell(0, 6, text)
        self.ln(4)


def _add_image_safe(pdf: _SafePDF, path, w: float = 190, ln: int = 4):
    if path and os.path.exists(path):
        pdf.image(path, w=w)
        pdf.ln(ln)


def _add_image_list(pdf: _SafePDF, paths, w: float = 190, ln: int = 4):
    for p in (paths or []):
        _add_image_safe(pdf, p, w=w, ln=ln)


#Report

def run_validation_report(
    real:           pd.DataFrame,
    synth:          pd.DataFrame,
    real_raw:       pd.DataFrame,
    num_ok:         list,
    cat_ok:         list,
    temporal_vars:  list,
    time_col:       str,
    patient_col:    str,
    fup_col:        str,
    max_len:        int,
    inverse_maps:   dict,
    output_path:    str,
    config_path:    str  = "",
    real_path:      str  = "",
    synth_path:     str  = "",
    umap_color_vars: list = None,
    irr_vars:       list = None,
) -> dict:
    """
    Esegue la pipeline completa di validazione:
      1. Metriche fidelity  (SFS, LFS, TFS)
      2. Metriche utility   (log-rank test KM) → UtilityScore
      3. Metriche privacy   (DCR, NNDR) → PrivacyScore
      4. Plot per sezione
      5. PDF report
    """
    os.makedirs(output_path, exist_ok=True)
    plot_dir = make_plot_dir(os.path.join(output_path, "plots"))

    # metriche
    print("[report] Calcolo metriche fidelity...")
    fidelity = compute_fidelity_metrics(
        real=real, synth=synth, real_raw=real_raw,
        num_ok=num_ok, cat_ok=cat_ok,
        temporal_vars=temporal_vars,
        time_col=time_col, patient_col=patient_col, fup_col=fup_col,
    )
    sfs = fidelity["sfs"]
    lfs = fidelity["lfs"]
    tfs = fidelity["tfs"]
    print(f"  SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")

    print("[report] Calcolo metriche privacy...")
    privacy = compute_privacy_metrics(
        real=real, synth=synth,
        numeric_features=num_ok,
        patient_col=patient_col,
        time_col=time_col,
    )

    # plot
    print("[report] Generazione plot...")

    dashboard_path = plot_summary_dashboard(sfs, lfs, tfs,plot_dir)

    fidelity_plots = plot_fidelity_section(
        real=real, synth=synth, real_raw=real_raw,
        num_ok=num_ok, cat_ok=cat_ok, temporal_vars=temporal_vars,
        time_col=time_col, patient_col=patient_col, fup_col=fup_col,
        lfs=lfs, inverse_maps=inverse_maps,
        outdir=plot_dir, umap_color_vars=umap_color_vars,
        death_col="DEATH",
        transp_col="TRANSP",
    )

    privacy_plots = plot_privacy_section(real, synth, num_ok, plot_dir)

    # pdf
    print("[report] Assemblaggio PDF...")
    pdf = _SafePDF()

    #Executive Summary 
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
    _add_image_safe(pdf, dashboard_path, w=190, ln=6)
    
    pdf.add_metrics_table(privacy.to_dict(), "PRIVACY METRICS")
    pdf.add_metrics_table(sfs.to_dict(),     "STATISTICAL FIDELITY SCORE (SFS)")
    pdf.add_metrics_table(lfs.to_dict(),     "LONGITUDINAL FIDELITY SCORE (LFS)")
    pdf.add_metrics_table(tfs.to_dict(),     "TEMPORAL FIDELITY SCORE (TFS)")
    

    pdf.add_note(
        f"SFS: fedelta' statistica.  "
        f"LFS: fedelta' longitudinale (slope, variance, autocorr, TCS, DTW).  "
        f"TFS: struttura visite (interval, timing, visit count, FUP coverage).  "
        f"Real troncato a max={max_len} visite e imputato.  "
        f"Score: 1=perfetto, 0=pessimo."
    )

    # region Fidelity
    pdf.section("A. FIDELITY - Distribuzioni Numeriche (KDE)")
    pdf.add_note("Real = imputato. KS e p-value annotati (p<0.05 = divergenza significativa).")
    _add_image_list(pdf, fidelity_plots.get("numeric", []))

    if fidelity_plots.get("categorical"):
        pdf.section("A. FIDELITY - Distribuzioni Categoriche (Chi2 / Fisher)")
        pdf.add_note(
            "Barre affiancate. Label decodificate via inverse_maps. "
            "Test Chi² se min(freq_attese) >= 5, Fisher esatto se < 5 (tabella 2x2). "
            "Effect size = Cramer's V (corretto per bias).")
        _add_image_list(pdf, fidelity_plots["categorical"])

    pdf.section("A. FIDELITY - Matrici di Correlazione")
    pdf.add_note("Sinistra=Real, Centro=Synthetic, Destra=|Diff|. Pearson per continue.")
    _add_image_list(pdf, fidelity_plots.get("correlation", []))

    pdf.section("A. FIDELITY - PCA ")
    pdf.add_note("PCA fittato sui reali, sintetici proiettati. Croci = centroidi.")
    _add_image_list(pdf, fidelity_plots.get("pca", []))
    pdf.section("A. FIDELITY - UMAP")
    pdf.add_note("UMAP: nuvole sovrapposte = buona fedelta'. Plot colorati per variabile.")
    _add_image_list(pdf, fidelity_plots.get("umap", []))

    pdf.section("A. FIDELITY - Traiettorie Temporali (Media ± 95% CI)")
    pdf.add_note("Curva media interpolata su griglia comune. Banda = CI 95%.")
    _add_image_list(pdf, fidelity_plots.get("traj_mean", []))

    pdf.section("A. FIDELITY - Traiettorie Individuali Campionate (n=20)")
    pdf.add_note("10 reali + 10 sintetici campionati casualmente.")
    _add_image_list(pdf, fidelity_plots.get("traj_sample", []))

    pdf.section("A. FIDELITY - Varianza Intra-Paziente")
    pdf.add_note("KDE varianza intra-paziente per ogni variabile temporale.")
    _add_image_list(pdf, fidelity_plots.get("variance_intra", []))

    pdf.section("A. FIDELITY - Varianza Inter-Paziente")
    pdf.add_note("Std delle medie per-paziente per bin di tempo.")
    _add_image_list(pdf, fidelity_plots.get("variance_inter", []))

    pdf.section("A. FIDELITY - Autocorrelazione Lag-1..5")
    pdf.add_note("KDE autocorrelazione per-paziente per lag k=1,2,3,4,5.")
    _add_image_list(pdf, fidelity_plots.get("autocorr", []))

    pdf.section("A. FIDELITY - Variabili Temporali per Bin di Visita")
    pdf.add_note("Boxplot affiancati + KS per bin. Verde<0.15, Arancio<0.30, Rosso>=0.30.")
    _add_image_list(pdf, fidelity_plots.get("var_by_visit", []))

    pdf.section("A. FIDELITY - Struttura delle Visite")
    pdf.add_note("Distribuzione numero visite per paziente. KS annotato.")
    _add_image_list(pdf, fidelity_plots.get("visit_dist", []))
    pdf.ln(3)
    pdf.add_note("Distribuzione timing per le prime 12 posizioni sequenziali.")
    _add_image_list(pdf, fidelity_plots.get("visit_pos", []))

    pdf.section(f"A. FIDELITY - Last Visit vs {fup_col}: Gap Analysis")
    pdf.add_note(
        f"Scatter t_FUP vs t_last_obs per paziente (reale e sintetico).\n"
        f"Il gap (t_FUP - t_last_obs) cattura la censura: pazienti troncati hanno gap>0.")
    _add_image_list(pdf, fidelity_plots.get("fup", []))

    # region Utility
    pdf.section("B. UTILITY")

    pdf.section("B. UTILITY - Kaplan-Meier")
    pdf.add_note(
        "KM DEATH: sopravvivenza libera da decesso. "
        "KM TRANSP: sopravvivenza libera da trapianto. "
        "Log-rank p>>0.05 = curve reale/sintetica compatibili."
    )
    _add_image_list(pdf, fidelity_plots.get("km", []))

    # region Privacy 
    pdf.section("C. PRIVACY")
    pdf.add_note(
        "DCR = distanza del sintetico al record reale piu' vicino (spazio standardizzato). "
        "Alto = basso rischio di copying.  "
        "NNDR[i] = d1nn / d2nn: valori < 0.5 indicano rischio re-identificazione.  "
        "Attribute Inference: AUC per predire una feature sensibile dalle altre (sintetico). "
        "AUC=0.5 -> non inferibile -> score=1.  "
        "Membership Inference: fraction(DCR < soglia p5 real-real). "
        "0 violazioni -> score=1.")
    pdf.add_metrics_table(privacy.to_dict(), "PRIVACY METRICS")
    _add_image_list(pdf, privacy_plots)

    # Salvataggio 
    out_file = os.path.join(output_path, "Synthetic_Data_Validation_Report.pdf")
    pdf.output(out_file)
    print(f"\n[OK] Report salvato: {out_file}")
    print(f"     SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")
    print(f"     Privacy={_fmt(privacy.overall)}")

    return {
        "sfs": sfs, "lfs": lfs, "tfs": tfs,
        "privacy": privacy,
    }