import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from config.config_loader import load_config, build_data_config

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from fpdf import FPDF

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, sem
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pandas.api.types import is_numeric_dtype, is_string_dtype, CategoricalDtype


# ======================================================
# CONFIGURAZIONE COLORI
# ======================================================
COLOR_REAL  = "#3B5998"   # Bluette
COLOR_SYNTH = "#FF7F50"   # Rosso Corallo


# ======================================================
# REPORT PDF
# ======================================================
class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Synthetic Data Validation Report", ln=True, align="C")
        self.ln(5)

    def section(self, title):
        self.add_page()
        self.set_font("Arial", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(5)

    def add_metrics_table(self, metrics_dict, title):
        self.set_font("Arial", "B", 11)
        self.cell(0, 8, title, ln=True)
        self.set_font("Arial", "", 10)
        for k, v in metrics_dict.items():
            text = f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            self.cell(0, 6, text, ln=True)
        self.ln(5)


# ======================================================
# METRICHE DI SIMILARITÀ
# ======================================================
def calculate_similarity_metrics(real, synth, num_vars, cat_vars):
    metrics = {}

    # KS Test per variabili numeriche
    ks_scores = []
    for v in num_vars:
        r_data = real[v].dropna()
        s_data = synth[v].dropna()
        if len(r_data) > 1 and len(s_data) > 1:
            try:
                score, _ = ks_2samp(r_data, s_data)
                ks_scores.append(score)
            except Exception:
                continue
    metrics["Avg Kolmogorov-Smirnov Dist (Lower is better)"] = (
        np.mean(ks_scores) if ks_scores else "N/A"
    )

    # Overlap frequenze categoriche
    cat_overlap = []
    for v in cat_vars:
        if v in real.columns and v in synth.columns:
            r_dist = real[v].value_counts(normalize=True)
            s_dist = synth[v].value_counts(normalize=True)
            all_cats = r_dist.index.union(s_dist.index)
            r_dist = r_dist.reindex(all_cats, fill_value=0)
            s_dist = s_dist.reindex(all_cats, fill_value=0)
            cat_overlap.append(np.sum(np.minimum(r_dist, s_dist)))
    metrics["Avg Categorical Overlap (Higher is better)"] = (
        np.mean(cat_overlap) if cat_overlap else "N/A"
    )

    return metrics


def calculate_ks_numeric(real, synth, numeric_vars):
    ks_vals = []
    for v in numeric_vars:
        r = real[v].dropna()
        s = synth[v].dropna()
        if len(r) > 10 and len(s) > 10:
            d, _ = ks_2samp(r, s)
            ks_vals.append(d)
    return float(np.mean(ks_vals)) if ks_vals else np.nan


# ======================================================
# METRICHE PCA NORMALIZZATE [0, 1]
# ======================================================
def calculate_pca_overlap_score(real, synth, num_vars):
    """
    Restituisce due metriche PCA normalizzate in [0, 1]:

    1. PCA Centroid Similarity (0 = lontani, 1 = sovrapposti)
       Basata sulla distanza euclidea tra centroidi PC1-PC2, normalizzata
       rispetto alla deviazione standard del real nel PCA space.

    2. PCA Distribution Overlap (0 = nessuna sovrapposizione, 1 = identiche)
       Stima dell'overlap tra le due distribuzioni nel PCA space usando
       un approccio basato su distanza mediana nearest-neighbor.
    """
    if len(num_vars) < 2:
        return {"PCA Centroid Similarity [0-1] (Higher is better)": "N/A",
                "PCA Distribution Overlap [0-1] (Higher is better)": "N/A"}

    r = real[num_vars].dropna(how="all").fillna(0)
    s = synth[num_vars].dropna(how="all").fillna(0)

    if len(r) < 5 or len(s) < 5:
        return {"PCA Centroid Similarity [0-1] (Higher is better)": "N/A",
                "PCA Distribution Overlap [0-1] (Higher is better)": "N/A"}

    scaler = StandardScaler()
    Xr = scaler.fit_transform(r)
    Xs = scaler.transform(s)

    pca = PCA(n_components=2)
    Zr = pca.fit_transform(Xr)
    Zs = pca.transform(Xs)

    # --- 1. Centroid Similarity ---
    # Distanza euclidea tra centroidi
    centroid_dist = np.linalg.norm(Zr.mean(axis=0) - Zs.mean(axis=0))
    # Normalizzazione: usiamo la deviazione standard media del real come scala
    scale = np.sqrt(Zr.var(axis=0).sum())  # norma della std nel PCA space
    # Sigmoid-like: exp(-d/scale) → 1 quando d=0, → 0 quando d>>scale
    centroid_similarity = float(np.exp(-centroid_dist / (scale + 1e-9)))

    # --- 2. Distribution Overlap (via pairwise distances) ---
    # Per ogni punto sintetico troviamo il nearest real; confrontiamo
    # con la distanza mediana intra-real (self-distance k=2)
    nn_real = NearestNeighbors(n_neighbors=2).fit(Zr)

    # Distanza di ogni sintetico dal suo real più vicino
    d_sr, _ = NearestNeighbors(n_neighbors=1).fit(Zr).kneighbors(Zs)
    d_sr = d_sr[:, 0]

    # Distanza tipica intra-real (escludiamo sé stessi, k=2)
    d_rr, _ = nn_real.kneighbors(Zr)
    d_rr = d_rr[:, 1]  # distanza al 1° vicino reale (non sé stesso)

    # Ratio: se i sintetici sono "vicini" quanto i reali tra loro → overlap ~ 1
    # overlap_ratio = P(d_sr <= median(d_rr))
    threshold = np.median(d_rr)
    overlap_score = float(np.mean(d_sr <= threshold))

    return {
        "PCA Centroid Similarity [0-1] (Higher is better)": centroid_similarity,
        "PCA Distribution Overlap [0-1] (Higher is better)": overlap_score,
    }


# ======================================================
# METRICHE CORRELAZIONE
# ======================================================
def calculate_correlation_distance(real, synth, num_vars):
    """MAE tra le matrici di correlazione."""
    if not num_vars:
        return 0.0
    r_corr = real[num_vars].corr().fillna(0).values
    s_corr = synth[num_vars].corr().fillna(0).values
    return float(np.mean(np.abs(r_corr - s_corr)))


# ======================================================
# PRIVACY METRICS
# ======================================================
def privacy_metrics(real, synth, num_vars):
    r = real[num_vars].dropna()
    s = synth[num_vars].dropna()
    n = min(len(r), len(s))
    r = r.iloc[:n]
    s = s.iloc[:n]

    scaler = StandardScaler()
    r = scaler.fit_transform(r)
    s = scaler.transform(s)

    nn = NearestNeighbors(n_neighbors=2).fit(r)
    dist, _ = nn.kneighbors(s)
    dcr = dist[:, 0] / (dist[:, 1] + 1e-8)

    return {
        "mean_DCR": float(np.mean(dcr)),
        "median_DCR": float(np.median(dcr)),
        "fraction_DCR_lt_1": float(np.mean(dcr < 1)),
    }


# ======================================================
# PLOTTING: DISTRIBUZIONI NUMERICHE (KDE)
# ======================================================
def plot_numeric_grid(real, synth, variables, outdir):
    paths = []
    ncols, max_rows = 3, 5
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
        axes = axes.flatten()

        for ax, v in zip(axes, chunk):
            sns.kdeplot(real[v].dropna(),  ax=ax, color=COLOR_REAL,  label="Real",  fill=True, alpha=0.3)
            sns.kdeplot(synth[v].dropna(), ax=ax, color=COLOR_SYNTH, label="Synth", fill=True, alpha=0.3)
            ax.set_title(f"Dist: {v}")

        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.tight_layout()
        path = os.path.join(outdir, f"num_dist_{i}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        paths.append(path)
    return paths


# ======================================================
# PLOTTING: DISTRIBUZIONI CATEGORICHE (BAR)
# ======================================================
def plot_categorical_grid(real, synth, variables, outdir):
    paths = []
    ncols, max_rows = 3, 4
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axes = axes.flatten()

        for ax, v in zip(axes, chunk):
            r_df = real[v].value_counts(normalize=True).rename("Percentage").reset_index()
            r_df["Dataset"] = "Real"
            s_df = synth[v].value_counts(normalize=True).rename("Percentage").reset_index()
            s_df["Dataset"] = "Synth"
            combined = pd.concat([r_df, s_df])

            sns.barplot(
                data=combined, x=v, y="Percentage", hue="Dataset",
                ax=ax, palette=[COLOR_REAL, COLOR_SYNTH], alpha=0.8,
            )
            ax.set_title(f"Dist: {v}")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(prop={"size": 8})

        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.tight_layout()
        path = os.path.join(outdir, f"cat_dist_{i}.png")
        plt.savefig(path, dpi=120)
        plt.close()
        paths.append(path)
    return paths


# ======================================================
# PLOTTING: MATRICI DI CORRELAZIONE (Real | Synth | Diff)
# ======================================================
def plot_correlation_comparison(real, synth, vars_list, title_suffix, outdir):
    """
    Genera una figura con TRE heatmap affiancate:
      1. Matrice di correlazione Real
      2. Matrice di correlazione Synthetic
      3. Differenza assoluta |Real - Synthetic|
    """
    if not vars_list:
        return None

    def prep_corr(df, cols):
        temp = df[cols].copy()
        for col in temp.select_dtypes(include=["object", "category"]).columns:
            temp[col] = pd.factorize(temp[col])[0].astype(float)
        return temp.corr()

    r_corr = prep_corr(real, vars_list)
    s_corr = prep_corr(synth, vars_list)
    diff_corr = (r_corr - s_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # 1. Real
    sns.heatmap(
        r_corr, ax=axes[0], cmap="coolwarm", center=0, vmin=-1, vmax=1,
        cbar=True, square=True,
    )
    axes[0].set_title(f"Real – {title_suffix}", fontsize=13, color=COLOR_REAL, fontweight="bold")

    # 2. Synthetic
    sns.heatmap(
        s_corr, ax=axes[1], cmap="coolwarm", center=0, vmin=-1, vmax=1,
        cbar=True, square=True,
    )
    axes[1].set_title(f"Synthetic – {title_suffix}", fontsize=13, color=COLOR_SYNTH, fontweight="bold")

    # 3. |Differenza|
    # Palette separata per evidenziare le discrepanze: bianco (0) → arancione scuro (1)
    sns.heatmap(
        diff_corr, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1,
        cbar=True, square=True,
        annot=(len(vars_list) <= 12),   # annotiamo solo se poche variabili
        fmt=".2f",
        annot_kws={"size": 7},
    )
    axes[2].set_title(f"|Real − Synthetic| – {title_suffix}", fontsize=13, color="#444444", fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, f"corr_{title_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================
# PLOTTING: PCA SHARED SPACE
# ======================================================
def plot_pca_shared_space(real, synth, num_vars, outdir):
    r = real[num_vars].dropna(how="all")
    s = synth[num_vars].dropna(how="all")

    if len(r) < 2 or len(s) < 2:
        print("⚠️  PCA skipped: insufficient data")
        return None

    scaler = StandardScaler()
    Xr = scaler.fit_transform(r.fillna(0))
    Xs = scaler.transform(s.fillna(0))

    pca = PCA(n_components=2)
    Zr = pca.fit_transform(Xr)
    Zs = pca.transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.35, s=8,  color=COLOR_REAL,  label="Real")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.35, s=8,  color=COLOR_SYNTH, label="Synthetic")

    # Centroidi
    ax.scatter(*Zr.mean(axis=0), marker="X", s=120, color=COLOR_REAL,  zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.scatter(*Zs.mean(axis=0), marker="X", s=120, color=COLOR_SYNTH, zorder=5,
               edgecolors="white", linewidths=0.8)

    var_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%} var)")
    ax.set_title("PCA Shared Space (fit on Real)\nCrosses = centroids")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(outdir, "pca_shared.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================
# PLOTTING: TRAIETTORIE TEMPORALI CON CI 95%
# ======================================================
def _build_interpolated_curves(df, var, time_col, patient_col, grid):
    """
    Per ciascun paziente, interpola il valore di `var` sulla griglia
    temporale comune `grid` e restituisce una matrice (n_pazienti × n_grid).

    Gestisce tempi irregolari: usa np.interp con extrapolazione ai bordi
    disabilitata (left/right = NaN se fuori range del paziente).
    """
    curves = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[time_col], errors="coerce").values
        y = pd.to_numeric(g[var],      errors="coerce").values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 2:
            continue
        xv, yv = x[mask], y[mask]
        # Interpola solo nell'intervallo del paziente (NaN fuori range)
        interp_y = np.where(
            (grid >= xv.min()) & (grid <= xv.max()),
            np.interp(grid, xv, yv),
            np.nan,
        )
        curves.append(interp_y)
    return np.array(curves) if curves else np.empty((0, len(grid)))


def _ci95(curves):
    """
    Restituisce (media, lower_CI95, upper_CI95) ignorando NaN.
    Usa SEM * 1.96 per il CI 95%.
    Dove ci sono meno di 3 osservazioni valide, CI = NaN.
    """
    n_valid = np.sum(~np.isnan(curves), axis=0)
    mean    = np.nanmean(curves, axis=0)
    stderr  = np.where(n_valid >= 3,
                       np.nanstd(curves, axis=0) / np.sqrt(np.maximum(n_valid, 1)),
                       np.nan)
    lo = mean - 1.96 * stderr
    hi = mean + 1.96 * stderr
    # Dove n < 3, lo/hi = NaN → il fill_between si interrompe correttamente
    lo = np.where(n_valid >= 3, lo, np.nan)
    hi = np.where(n_valid >= 3, hi, np.nan)
    return mean, lo, hi, n_valid


def plot_temporal_trajectory(
    real,
    synth,
    var,
    time_col="MONTHS_FROM_BASELINE",
    patient_col="RECORD_ID",
    max_time=None,
    n_grid=50,
    min_patients=5,
    outdir="plots",
):
    """
    Traccia la traiettoria media ± CI 95% per `var` nel tempo.

    Parametri
    ---------
    real, synth : DataFrame con colonne [patient_col, time_col, var]
    max_time    : limite superiore della griglia temporale.
                  Se None viene calcolato come max(real, synth).
    n_grid      : numero di punti nella griglia temporale comune.
    min_patients: numero minimo di pazienti per mostrare il CI.
    """
    # Griglia temporale comune che copre entrambi i dataset
    t_max_real  = pd.to_numeric(real[time_col],  errors="coerce").max()
    t_max_synth = pd.to_numeric(synth[time_col], errors="coerce").max()
    t_max_all   = max(
        t_max_real  if not np.isnan(t_max_real)  else 0,
        t_max_synth if not np.isnan(t_max_synth) else 0,
    )
    if max_time is None:
        max_time = t_max_all
    if max_time <= 0:
        return None

    grid = np.linspace(0.0, float(max_time), int(n_grid))

    r_curves = _build_interpolated_curves(real,  var, time_col, patient_col, grid)
    s_curves = _build_interpolated_curves(synth, var, time_col, patient_col, grid)

    if len(r_curves) == 0 or len(s_curves) == 0:
        return None

    r_mean, r_lo, r_hi, r_n = _ci95(r_curves)
    s_mean, s_lo, s_hi, s_n = _ci95(s_curves)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Real
    ax.plot(grid, r_mean, color=COLOR_REAL,  lw=2, label=f"Real (n={len(r_curves)})")
    valid_r = r_n >= min_patients
    ax.fill_between(grid, r_lo, r_hi, where=valid_r,
                    color=COLOR_REAL, alpha=0.2, label="CI 95% Real")

    # Synthetic
    ax.plot(grid, s_mean, color=COLOR_SYNTH, lw=2, label=f"Synthetic (n={len(s_curves)})")
    valid_s = s_n >= min_patients
    ax.fill_between(grid, s_lo, s_hi, where=valid_s,
                    color=COLOR_SYNTH, alpha=0.2, label="CI 95% Synth")

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel(var)
    ax.set_title(f"Temporal Trajectory – {var}\n(mean ± 95% CI, interpolated on common grid)")
    ax.legend(fontsize=8)
    plt.tight_layout()

    path = os.path.join(outdir, f"trajectory_{var}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================
# UTILITY
# ======================================================
def ensure_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def get_variable_types(df, exclude):
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_dtype(df[c]):
            num.append(c)
        elif is_string_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype):
            cat.append(c)
    return num, cat


def filter_valid_visits(df, visit_mask_col="VISIT_MASK"):
    if visit_mask_col in df.columns:
        return df[df[visit_mask_col] == 1].copy()
    return df.copy()


# ======================================================
# MAIN
# ======================================================
def main(real_path, synth_path, config_path, output_path):
    # --------------------------------------------------
    # 1. CARICAMENTO
    # --------------------------------------------------
    print("Caricamento dati...")
    real  = pd.read_excel(real_path)
    synth = pd.read_excel(synth_path)

    time_cfg, variables, _ = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)

    num_vars = [v.name for v in data_cfg.static_cont] + [v.name for v in data_cfg.temporal_cont]
    cat_vars = [v.name for v in data_cfg.static_cat]  + [v.name for v in data_cfg.temporal_cat]

    time_col    = time_cfg.visit_column
    patient_col = "RECORD_ID"   # adatta se necessario

    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    pdf = ReportPDF()

    # --------------------------------------------------
    # 2. CALCOLO METRICHE
    # --------------------------------------------------
    print("Calcolo metriche di validazione...")

    dist_metrics = calculate_similarity_metrics(real, synth, num_vars, cat_vars)

    corr_dist = calculate_correlation_distance(real, synth, num_vars)
    dist_metrics["Correlation Matrix Distance (Lower is better)"] = corr_dist

    # Metriche PCA normalizzate [0, 1]
    pca_scores = calculate_pca_overlap_score(real, synth, num_vars)
    dist_metrics.update(pca_scores)

    priv_metrics = privacy_metrics(real, synth, num_vars)

    # --------------------------------------------------
    # 3. EXECUTIVE SUMMARY
    # --------------------------------------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "Executive Summary: Synthetic Data Fidelity", ln=True, align="C")
    pdf.ln(5)

    pdf.add_metrics_table(dist_metrics,  "STATISTICAL FIDELITY METRICS")
    pdf.add_metrics_table(priv_metrics,  "PRIVACY & DISTANCE METRICS (DCR)")

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 5,
        "Note: KS distance measures distributional divergence (0 = identical). "
        "PCA Centroid Similarity and PCA Distribution Overlap are normalised to [0, 1] "
        "(1 = perfect overlap). Correlation Distance is the mean absolute error between "
        "correlation matrices. DCR monitors real-data copying risk.",
    )

    # --------------------------------------------------
    # 4. DISTRIBUZIONI NUMERICHE
    # --------------------------------------------------
    print("Generazione grafici distribuzioni numeriche...")
    pdf.section("Numerical Distributions (KDE Analysis)")
    for img_path in plot_numeric_grid(real, synth, num_vars, plot_dir):
        pdf.image(img_path, w=190)

    # --------------------------------------------------
    # 5. DISTRIBUZIONI CATEGORICHE
    # --------------------------------------------------
    if cat_vars:
        print("Generazione grafici distribuzioni categoriche...")
        pdf.section("Categorical Distributions (Frequency Analysis)")
        for img_path in plot_categorical_grid(real, synth, cat_vars, plot_dir):
            pdf.image(img_path, w=190)

    # --------------------------------------------------
    # 6. MATRICI DI CORRELAZIONE (Real | Synth | |Diff|)
    # --------------------------------------------------
    print("Generazione matrici di correlazione...")
    pdf.section("Correlation Matrices Comparison")

    img_corr_num = plot_correlation_comparison(real, synth, num_vars, "Numerical", plot_dir)
    if img_corr_num:
        pdf.image(img_corr_num, w=190)

    if cat_vars:
        pdf.ln(10)
        img_corr_cat = plot_correlation_comparison(real, synth, cat_vars, "Categorical", plot_dir)
        if img_corr_cat:
            pdf.image(img_corr_cat, w=190)

    # --------------------------------------------------
    # 7. PCA SHARED SPACE
    # --------------------------------------------------
    print("Generazione PCA...")
    pdf.section("Multivariate Analysis (PCA Shared Space)")

    img_pca = plot_pca_shared_space(real, synth, num_vars, plot_dir)
    if img_pca:
        pdf.image(img_pca, w=140)

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 5,
        "PCA fitted exclusively on real data; synthetic data is projected onto the same axes. "
        "Crosses mark the centroid of each cloud. "
        "PCA Centroid Similarity and Distribution Overlap are reported in the Executive Summary.",
    )

    # --------------------------------------------------
    # 8. TEMPORAL TRAJECTORIES (sezione dedicata)
    # --------------------------------------------------
    print("Generazione traiettorie temporali...")
    pdf.section("Temporal Trajectories (Mean ± 95% CI)")

    pdf.set_font("Arial", "I", 9)
    pdf.multi_cell(
        0, 5,
        "Each curve shows the mean value of the marker over time, "
        "interpolated onto a common time grid that spans both real and synthetic datasets. "
        "Shaded bands represent the 95% confidence interval (mean ± 1.96 × SEM). "
        "Bands are shown only where at least 5 patients contribute data. "
        "Irregular visit schedules are handled by per-patient linear interpolation.",
    )
    pdf.ln(4)

    temporal_vars = [v.name for v in data_cfg.temporal_cont]
    for v in temporal_vars:
        img_traj = plot_temporal_trajectory(
            real, synth, v,
            time_col=time_col,
            patient_col=patient_col,
            max_time=None,   # auto: max(real, synth)
            n_grid=60,
            min_patients=5,
            outdir=plot_dir,
        )
        if img_traj:
            pdf.image(img_traj, w=170)
            pdf.ln(5)

    # --------------------------------------------------
    # 9. SALVATAGGIO
    # --------------------------------------------------
    output_name = f"{output_path}/Synthetic_Data_Validation_Report.pdf"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)
    pdf.output(output_name)
    print(f"✅ REPORT COMPLETATO: {output_name}")


if __name__ == "__main__":

    output_path = "output/exp_3"

    main(
        real_path="PBC_UDCA_long_stratificato.xlsx",
        synth_path= f"{output_path}/synthetic_data.xlsx",
        config_path="config/data_config.json",
        output_path=output_path
    )