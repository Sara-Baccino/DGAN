# ======================================================
# eval/report.py  [v4]
#
# run_validation_report():
#   Orchestratore centrale chiamato da main_eval.py.
#   1. Calcola tutte le metriche (fidelity / utility / privacy)
#   2. Genera tutti i plot
#   3. Assembla il PDF
#
# Il PDF è strutturato in 4 macro-sezioni:
#   Pag. 1  — Executive Summary (3 radar + utility + privacy)
#   Sec. A  — Fidelity: distribuzioni, correlazioni, PCA, UMAP,
#              traiettorie, varianza, LME, autocorr, visite, KM
#   Sec. B  — Utility: score bar + note TSTR
#   Sec. C  — Privacy: DCR/NNDR
# ======================================================

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from eval.report_pdf import ReportPDF, make_plot_dir
from eval.metrics import (
    compute_fidelity_metrics,
    compute_utility_metrics,
    compute_privacy_metrics,
)
from eval.plots import (
    plot_scores_dashboard,
    plot_summary_dashboard,
    plot_fidelity_section,
    plot_utility_section,
    plot_privacy_section,
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


# ── SafePDF ───────────────────────────────────────────────────────────────────

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


# ── Helpers PDF ───────────────────────────────────────────────────────────────

def _add_image_safe(pdf: _SafePDF, path: str | None,
                     w: float = 190, ln: int = 4):
    if path and os.path.exists(path):
        pdf.image(path, w=w)
        pdf.ln(ln)


def _add_image_list(pdf: _SafePDF, paths: list[str],
                     w: float = 190, ln: int = 4):
    for p in paths:
        _add_image_safe(pdf, p, w=w, ln=ln)


# ── Orchestratore principale ──────────────────────────────────────────────────

def run_validation_report(
    real:           pd.DataFrame,
    synth:          pd.DataFrame,
    real_raw:       pd.DataFrame,
    num_ok:         list[str],
    cat_ok:         list[str],
    temporal_vars:  list[str],
    time_col:       str,
    patient_col:    str,
    fup_col:        str,
    max_len:        int,
    inverse_maps:   dict,
    output_path:    str,
    config_path:    str    = "",
    real_path:      str    = "",
    synth_path:     str    = "",
    umap_color_vars: list[str] | None = None,
    tstr_targets:   list[str] | None  = None,
) -> dict:
    """
    Esegue la pipeline completa di validazione:
      1. Metriche fidelity  (SFS, LFS, TFS)
      2. Metriche utility   (discriminator, pMSE, TSTR)
      3. Metriche privacy   (DCR, NNDR)
      4. Plot per sezione
      5. PDF report

    Parametri
    ----------
    tstr_targets : colonne binarie usate come outcome per il test TSTR
                   (es. ["DEATH", "TRANSP"]). Se None → TSTR saltato.
    umap_color_vars : variabili per UMAP colorato (es. ["SEX", "Risk_Level_Label"]).
    """
    os.makedirs(output_path, exist_ok=True)
    plot_dir = make_plot_dir(os.path.join(output_path, "plots"))

    # ── 1. METRICHE ───────────────────────────────────────────────────────────
    print("[report] Calcolo metriche fidelity...")
    fidelity = compute_fidelity_metrics(
        real          = real,
        synth         = synth,
        real_raw      = real_raw,
        num_ok        = num_ok,
        cat_ok        = cat_ok,
        temporal_vars = temporal_vars,
        time_col      = time_col,
        patient_col   = patient_col,
        fup_col       = fup_col,
    )
    sfs = fidelity["sfs"]
    lfs = fidelity["lfs"]
    tfs = fidelity["tfs"]
    print(f"  SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")

    print("[report] Calcolo metriche utility...")
    utility = compute_utility_metrics(
        real         = real,
        synth        = synth,
        num_ok       = num_ok,
        tstr_targets = tstr_targets or ["DEATH", "TRANSP"],
    )

    print("[report] Calcolo metriche privacy...")
    privacy = compute_privacy_metrics(real, synth, num_ok)

    # ── 2. PLOT ───────────────────────────────────────────────────────────────
    print("[report] Generazione plot...")

    # Executive summary dashboard
    dashboard_path = plot_summary_dashboard(sfs, lfs, tfs, utility, privacy, plot_dir)

    
    # Fidelity plots
    fidelity_plots = plot_fidelity_section(
        real           = real, synth          = synth, real_raw       = real_raw,
        num_ok         = num_ok, cat_ok         = cat_ok, temporal_vars  = temporal_vars,
        time_col       = time_col, patient_col    = patient_col, fup_col        = fup_col,
        lfs            = lfs, inverse_maps   = inverse_maps,
        outdir         = plot_dir, umap_color_vars = umap_color_vars,
    )
    

    # Utility plots
    utility_plots = plot_utility_section(utility, plot_dir)

    # Privacy plots
    privacy_plots = plot_privacy_section(real, synth, num_ok, plot_dir)

    # ── 3. PDF ────────────────────────────────────────────────────────────────
    print("[report] Assemblaggio PDF...")
    pdf = _SafePDF()

    # ── Pagina 1: Executive Summary ───────────────────────────────────────────
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

    '''
    # Score banner 3 + 2
    for score_obj, label, fill_rgb in [
        (sfs, "SFS", (230, 245, 255)),
        (lfs, "LFS", (255, 243, 205)),
        (tfs, "TFS", (205, 230, 255)),
    ]:
        pdf.set_fill_color(*fill_rgb)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(63, 9,
                 f"{label}: {_fmt(score_obj.overall)}  {score_obj.grade[:3]}",
                 ln=False, fill=True, align="C")
    pdf.ln(5)

    # Utility + Privacy overall
    u_overall = utility.get("*** Utility overall [0-1]", "N/A")
    p_overall = privacy.get("*** Privacy overall [0-1]", "N/A")
    pdf.set_fill_color(240, 240, 220)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(95, 9, f"Utility: {_fmt(u_overall)}",  ln=False, fill=True, align="C")
    pdf.cell(95, 9, f"Privacy: {_fmt(p_overall)}",  ln=True,  fill=True, align="C")
    #pdf.ln(3)
    '''
    _add_image_safe(pdf, dashboard_path, w=190, ln=6)

    # Dettaglio score
    pdf.add_metrics_table(sfs.to_dict(),  "STATISTICAL FIDELITY SCORE (SFS)")
    pdf.add_metrics_table(lfs.to_dict(),  "LONGITUDINAL FIDELITY SCORE (LFS)")
    pdf.add_metrics_table(tfs.to_dict(),  "TEMPORAL FIDELITY SCORE (TFS)")
    pdf.add_metrics_table(utility,         "UTILITY METRICS")
    pdf.add_metrics_table(privacy,         "PRIVACY METRICS (DCR / NNDR)")

    pdf.add_note(
        f"SFS: fedelta' statistica [0-1].  "
        f"LFS: fedelta' longitudinale [0-1] (include DTW e LME).  "
        f"TFS: fedelta' temporale + t_FUP [0-1].  "
        f"Real troncato a max={max_len} visite e imputato (MICE/KNN).  "
        f"Score: 1=perfetto, 0=pessimo. lower-better -> 1-score (lineare)."
    )

    
    # ── Sezione A: Fidelity ───────────────────────────────────────────────────
    pdf.section("A. FIDELITY — Distribuzioni Numeriche (KDE)")
    pdf.add_note("Real = imputato. KS e p-value annotati (p<0.05 = divergenza significativa).")
    _add_image_list(pdf, fidelity_plots.get("numeric", []))

    if fidelity_plots.get("categorical"):
        pdf.section("A. FIDELITY — Distribuzioni Categoriche")
        pdf.add_note("Barre affiancate. Label decodificate via inverse_maps. Cramer's V annotato.")
        _add_image_list(pdf, fidelity_plots["categorical"])

    pdf.section("A. FIDELITY — Matrici di Correlazione")
    pdf.add_note("Sinistra=Real, Centro=Synthetic, Destra=|Diff|. Pearson per continue.")
    _add_image_list(pdf, fidelity_plots.get("correlation", []))

    pdf.section("A. FIDELITY — PCA + UMAP")
    pdf.add_note("PCA fittato sui reali, sintetici proiettati. Croci = centroidi.")
    _add_image_list(pdf, fidelity_plots.get("pca", []))
    pdf.add_note("UMAP: nuvole sovrapposte = buona fedelta'. Plot colorati per variabile.")
    _add_image_list(pdf, fidelity_plots.get("umap", []))

    pdf.section("A. FIDELITY — Traiettorie Temporali (Media ± 95% CI)")
    pdf.add_note("Curva media interpolata su griglia comune. Banda = CI 95%.")
    _add_image_list(pdf, fidelity_plots.get("traj_mean", []))

    pdf.section("A. FIDELITY — Traiettorie Individuali Campionate (n=20)")
    pdf.add_note("10 reali + 10 sintetici campionati casualmente.")
    _add_image_list(pdf, fidelity_plots.get("traj_sample", []))

    pdf.section("A. FIDELITY — Varianza Intra-Paziente")
    pdf.add_note("KDE varianza intra-paziente per ogni variabile temporale.")
    _add_image_list(pdf, fidelity_plots.get("variance_intra", []))

    pdf.section("A. FIDELITY — Varianza Inter-Paziente")
    pdf.add_note("Std delle medie per-paziente per bin di tempo.")
    _add_image_list(pdf, fidelity_plots.get("variance_inter", []))

    pdf.section("A. FIDELITY — LME Slope Comparison")
    pdf.add_note(
        "Coefficiente fisso del tempo (beta) fittato con Linear Mixed Effects. Rappresenta la velocità media di variazione di una variabile nel tempo. Dati raggruppati per paziente." \
        "Il segno positivo indica che la variabile tende ad aumentare col passare del tempo."
        "(var ~ time + (1|patient)). Confronta real vs sintetico.")
    _add_image_list(pdf, fidelity_plots.get("lme", []))
    if lfs.lme_betas:
        table = {}
        for v, b in lfs.lme_betas.items():
            table[f"{v} beta_real"]  = b.get("beta_real",  "N/A")
            table[f"{v} beta_synth"] = b.get("beta_synth", "N/A")
            table[f"{v} |Δbeta|"]    = b.get("beta_diff_abs", "N/A")
        pdf.add_metrics_table(table, "LME betas per variabile")

    pdf.section("A. FIDELITY — Autocorrelazione Lag-1..5")
    pdf.add_note("KDE autocorrelazione per-paziente per lag k=1,2,3,4,5.")
    _add_image_list(pdf, fidelity_plots.get("autocorr", []))

    pdf.section("A. FIDELITY — Variabili Temporali per Bin di Visita")
    pdf.add_note("Boxplot affiancati + KS per bin. Verde<0.15, Arancio<0.30, Rosso>=0.30.")
    _add_image_list(pdf, fidelity_plots.get("var_by_visit", []))

    pdf.section("A. FIDELITY — Struttura delle Visite")
    pdf.add_note("Distribuzione numero visite per paziente. KS annotato.")
    _add_image_list(pdf, fidelity_plots.get("visit_dist", []))
    pdf.add_note("Distribuzione timing per le prime 7 posizioni sequenziali.")
    _add_image_list(pdf, fidelity_plots.get("visit_pos", []))

    pdf.section(f"A. FIDELITY — Last Visit vs {fup_col}: Allineamento Temporale")
    pdf.add_note(
        f"Coverage = last_visit / {fup_col}: ideale = 1.0.  "
        "Usa real_raw (t_FUP non cambia con imputazione).")
    _add_image_list(pdf, fidelity_plots.get("fup", []))

    pdf.section("A. FIDELITY — Kaplan-Meier")
    pdf.add_note(
        "KM Overall: log-rank p>>0.05 = curve compatibili.  "
        "KM POISE: stratificato per risposta (ALP<=2 & BIL<=1 a mese 12), "
        "evento = DEATH o TRANSP.")
    _add_image_list(pdf, fidelity_plots.get("km", []))

    # ── Sezione B: Utility ────────────────────────────────────────────────────
    pdf.section("B. UTILITY")
    pdf.add_note("Discriminator score: AUC RF real-vs-synth -> score=1-|AUC-0.5|*2.  Classification con Random Forest.")
    pdf.add_note("pMSE: indistinguibilita' globale. propensity -> score=1-clip(pMSE/0.25,0,1).  ")
    pdf.add_note("TSTR: AUC_TSTR vs AUC_TRTR; score=1-gap.  Viene usato un modello predittivo.")
    pdf.add_metrics_table(utility, "UTILITY SCORES")
    _add_image_list(pdf, utility_plots)

    # ── Sezione C: Privacy ────────────────────────────────────────────────────
    pdf.section("C. PRIVACY")
    pdf.add_note(
        "DCR = distanza minima al record reale piu' vicino (spazio standardizzato).  Alto: basso rischio copying."
        "NNDR = dist_1nn / dist_2nn: valori bassi indicano rischio re-identificazione.  "
        "DCR_score = clip(median_DCR / typical_dist, 0, 2) / 2.  "
        "NNDR_score = fraction(NNDR >= 0.5).")
    pdf.add_metrics_table(privacy, "PRIVACY METRICS")
    _add_image_list(pdf, privacy_plots)
    
    # ── Salvataggio ───────────────────────────────────────────────────────────
    out_file = os.path.join(output_path, "Synthetic_Data_Validation_Report.pdf")
    pdf.output(out_file)
    print(f"\n[OK] Report salvato: {out_file}")
    print(f"     SFS={_fmt(sfs.overall)} | LFS={_fmt(lfs.overall)} | TFS={_fmt(tfs.overall)}")
    print(f"     Utility={_fmt(utility.get('*** Utility overall [0-1]', float('nan')))} | "
          f"Privacy={_fmt(privacy.get('*** Privacy overall [0-1]', float('nan')))}")

    return {
        "sfs": sfs, "lfs": lfs, "tfs": tfs,
        "utility": utility, "privacy": privacy,
    }