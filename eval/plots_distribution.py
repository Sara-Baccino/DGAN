# ======================================================
# eval/plots_distribution.py  [v2]
# Distribution plots: numeric KDE, categorical bar,
# correlation matrices, PCA shared space.
#
# Fix v2:
#   - Categorical labels: mostra stringhe originali invece di codici int.
#     Se una colonna contiene int, usa inverse_maps dal preprocessore
#     (se disponibile) oppure converte a stringa per leggibilità.
#   - Bar categoriche: barre AFFIANCATE (dodge) invece di sovrapposte.
#     Usa matplotlib direttamente con posizioni calcolate, nessuna
#     dipendenza da hue= di seaborn (che causa overlap con valori numerici).
# ======================================================

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from eval.config import COLOR_REAL, COLOR_SYNTH
except ImportError:
    COLOR_REAL  = "#457B9D"
    COLOR_SYNTH = "#E63946"


# ======================================================
# Utility: decodifica colonne categoriche int -> stringa
# ======================================================

def _try_int(x):
    """Converti x in int se possibile, altrimenti None."""
    try:
        f = float(x)
        return int(f)
    except (ValueError, TypeError):
        return None


def decode_categoricals(
    df: pd.DataFrame,
    cat_vars: list,
    inverse_maps: dict | None = None,
) -> pd.DataFrame:
    df = df.copy()
    
    # Se inverse_maps è None, lo trasformiamo in un dizionario vuoto per evitare il TypeError
    if inverse_maps is None:
        inverse_maps = {}

    for col in cat_vars:
        if col not in df.columns:
            continue

        # Caso A: Abbiamo una mappa di decodifica (es. 0 -> "Maschio")
        if col in inverse_maps:
            imap = inverse_maps[col]

            def _decode(x, m=imap):
                if pd.isna(x) or x is None:
                    return "NA"
                as_int = _try_int(x)
                if as_int is not None and as_int in m:
                    return str(m[as_int])
                return str(x)

            df[col] = df[col].map(_decode).astype(str)
        
        # Caso B: Non abbiamo mappa (es. la colonna è già stringa "PBC001")
        else:
            def _clean(x):
                if pd.isna(x) or x is None:
                    return "NA"
                # Se è un float .0 (es 1.0), lo puliamo in "1"
                as_int = _try_int(x)
                if as_int is not None:
                    return str(as_int)
                return str(x)

            df[col] = df[col].map(_clean).astype(str)
            
    return df


# ======================================================
# Numeric KDE grid
# ======================================================

def plot_numeric_grid(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    variables: list,
    outdir: str,
    ncols: int = 3,
    max_rows: int = 5,
) -> list:
    paths = []
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = np.array(axes).flatten()

        for ax, v in zip(axes, chunk):
            r_data = pd.to_numeric(real[v],  errors="coerce").dropna()
            s_data = pd.to_numeric(synth[v], errors="coerce").dropna()
            sns.kdeplot(r_data, ax=ax, color=COLOR_REAL,  label="Real",  fill=True, alpha=0.3)
            sns.kdeplot(s_data, ax=ax, color=COLOR_SYNTH, label="Synth", fill=True, alpha=0.3)
            ax.legend(fontsize=8)
            if len(r_data) > 1 and len(s_data) > 1:
                ks_stat, p_val = ks_2samp(r_data, s_data)
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.text(
                    0.02, 0.97, f"KS={ks_stat:.3f}, {p_str}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            ax.set_title(v, fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelsize=7)

        for ax in axes[len(chunk):]:
            ax.axis("off")

        plt.subplots_adjust(hspace=0.55, wspace=0.35)
        path = os.path.join(outdir, f"num_dist_{i}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


# ======================================================
# Categorical bar grid — side-by-side, string labels
# ======================================================

def _cramers_v_chi2(r_vals: pd.Series, s_vals: pd.Series):
    """Cramer's V e chi2 tra due distribuzioni categoriche."""
    all_cats = sorted(set(r_vals.unique()) | set(s_vals.unique()))
    rc = r_vals.value_counts().reindex(all_cats, fill_value=0)
    sc = s_vals.value_counts().reindex(all_cats, fill_value=0)
    contingency = np.array([rc.values, sc.values])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    n  = contingency.sum()
    k, r_dim = contingency.shape[1], contingency.shape[0]
    phi2 = max(0, chi2_stat / (n + 1e-9) - ((k - 1) * (r_dim - 1)) / (n - 1 + 1e-9))
    rcorr = r_dim - ((r_dim - 1) ** 2) / (n - 1 + 1e-9)
    kcorr = k   - ((k   - 1) ** 2) / (n - 1 + 1e-9)
    denom = min(kcorr - 1, rcorr - 1)
    cv = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
    return cv, chi2_stat, p_val


def plot_categorical_grid(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    variables: list,
    outdir: str,
    ncols: int = 3,
    max_rows: int = 4,
    inverse_maps: dict | None = None,
) -> list:
    """
    Griglia di bar chart per variabili categoriche.
    Le barre sono AFFIANCATE (side-by-side), non sovrapposte.
    Le label sull'asse X mostrano le stringhe originali (non int codes).

    Parameters
    ----------
    inverse_maps : preprocessor.inverse_maps per decodifica int->stringa.
                   Se None, converte con str().
    """
    # Decodifica int -> stringa originale
    #real_dec  = decode_categoricals(real,  variables, inverse_maps)
    #synth_dec = decode_categoricals(synth, variables, inverse_maps)

    paths   = []
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5 * nrows))
        axes = np.array(axes).flatten()

        for ax, v in zip(axes, chunk):
            r_raw = real[v].astype(str).replace('nan', np.nan).dropna() #real_dec[v]
            s_raw = synth[v].astype(str).replace('nan', np.nan).dropna()

            # Ordina categorie: prima per frequenza reale, poi alfabeticamente
            all_cats = sorted(
                set(r_raw.unique()) | set(s_raw.unique()),
                key=lambda c: (-r_raw.value_counts().get(c, 0), c),
            )

            r_freq = r_raw.value_counts(normalize=True).reindex(all_cats, fill_value=0)
            s_freq = s_raw.value_counts(normalize=True).reindex(all_cats, fill_value=0)

            # Barre affiancate con matplotlib
            x     = np.arange(len(all_cats))
            width = 0.38
            ax.bar(x - width / 2, r_freq.values, width,
                   color=COLOR_REAL,  alpha=0.80, label="Real",  edgecolor="white", lw=0.5)
            ax.bar(x + width / 2, s_freq.values, width,
                   color=COLOR_SYNTH, alpha=0.80, label="Synth", edgecolor="white", lw=0.5)

            ax.set_xticks(x)
            # Tronca le label lunghe per leggibilità
            tick_labels = [str(c)[:14] for c in all_cats]
            ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel("Percentuale", fontsize=8)
            ax.set_title(v, fontsize=9, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)

            # Cramer's V annotation
            try:
                cv, chi2_stat, p_val = _cramers_v_chi2(r_raw, s_raw)
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.text(
                    0.02, 0.97,
                    f"chi2={chi2_stat:.1f}, {p_str}\nCramér's V={cv:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            except Exception:
                pass

        for ax in axes[len(chunk):]:
            ax.axis("off")

        plt.subplots_adjust(hspace=0.65, wspace=0.35)
        path = os.path.join(outdir, f"cat_dist_{i}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


# ======================================================
# Correlation matrices
# ======================================================

def plot_correlation_comparison(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    vars_list: list,
    title_suffix: str,
    outdir: str,
) -> str | None:
    if not vars_list:
        return None

    def prep_corr(df, cols):
        if not cols:
            return pd.DataFrame()
        
        temp = df[cols].copy()
        for col in cols:
            # Controllo robusto per capire se la colonna è numerica
            # Gestisce i nuovi StringDtype di Pandas che rompevano np.issubdtype
            is_num = pd.api.types.is_numeric_dtype(temp[col])
            
            if not is_num:
                # Se è una stringa (es. 'PBC0001'), la trasformiamo in numeri (0, 1, 2...)
                # solo per poter calcolare la correlazione di Pearson/Spearman
                temp[col] = pd.factorize(temp[col])[0]
            else:
                # Se è numerica, assicuriamoci che sia float per gestire i NaN
                temp[col] = temp[col].astype(float)
                
        return temp.corr(method='pearson')

    r_corr    = prep_corr(real,  vars_list)
    s_corr    = prep_corr(synth, vars_list)
    diff_corr = (r_corr - s_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    kw = dict(cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar=True, square=True)
    sns.heatmap(r_corr,    ax=axes[0], **kw)
    axes[0].set_title(f"Real — {title_suffix}", fontsize=13,
                      color=COLOR_REAL, fontweight="bold")
    sns.heatmap(s_corr,    ax=axes[1], **kw)
    axes[1].set_title(f"Synthetic — {title_suffix}", fontsize=13,
                      color=COLOR_SYNTH, fontweight="bold")
    sns.heatmap(diff_corr, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1,
                cbar=True, square=True,
                annot=(len(vars_list) <= 12), fmt=".2f", annot_kws={"size": 7})
    axes[2].set_title(f"|Real − Synthetic| — {title_suffix}", fontsize=13,
                      fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, f"corr_{title_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ======================================================
# PCA shared space
# ======================================================

def plot_pca_shared_space(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_vars: list,
    outdir: str,
) -> str | None:
    r = real[num_vars].dropna(how="all")
    s = synth[num_vars].dropna(how="all")
    if len(r) < 2 or len(s) < 2:
        return None

    scaler = StandardScaler()
    Xr = scaler.fit_transform(r.fillna(0))
    Xs = scaler.transform(s.fillna(0))
    pca = PCA(n_components=2)
    Zr  = pca.fit_transform(Xr)
    Zs  = pca.transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.35, s=8,
               color=COLOR_REAL,  label="Real")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.35, s=8,
               color=COLOR_SYNTH, label="Synthetic")
    ax.scatter(*Zr.mean(0), marker="X", s=120, color=COLOR_REAL,
               zorder=5, edgecolors="white", lw=0.8)
    ax.scatter(*Zs.mean(0), marker="X", s=120, color=COLOR_SYNTH,
               zorder=5, edgecolors="white", lw=0.8)
    ve = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ve[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ve[1]:.1%} var)")
    ax.set_title("PCA Shared Space (fit on Real)\nCrosses = centroids")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(outdir, "pca_shared.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path