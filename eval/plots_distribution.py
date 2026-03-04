# ======================================================
# eval/plots_distribution.py
# Distribution plots: numeric KDE, categorical bar,
# correlation matrices, PCA shared space
# ======================================================

import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from eval.config import COLOR_REAL, COLOR_SYNTH


# -- Numeric KDE grid ---------------------------------

def plot_numeric_grid(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    variables: list[str],
    outdir: str,
    ncols: int = 3,
    max_rows: int = 5,
) -> list[str]:
    paths = []
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = axes.flatten()

        for ax, v in zip(axes, chunk):
            r_data = real[v].dropna()
            s_data = synth[v].dropna()
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


# -- Categorical bar grid -----------------------------

def _cramers_v_chi2(r_vals, s_vals):
    all_cats = sorted(set(r_vals.unique()) | set(s_vals.unique()))
    rc = r_vals.value_counts().reindex(all_cats, fill_value=0)
    sc = s_vals.value_counts().reindex(all_cats, fill_value=0)
    contingency = np.array([rc.values, sc.values])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency)
    n = contingency.sum()
    k, r_dim = contingency.shape[1], contingency.shape[0]
    phi2corr = max(0, chi2_stat / (n + 1e-9) - ((k - 1) * (r_dim - 1)) / (n - 1 + 1e-9))
    rcorr = r_dim - ((r_dim - 1) ** 2) / (n - 1 + 1e-9)
    kcorr = k - ((k - 1) ** 2) / (n - 1 + 1e-9)
    denom = min(kcorr - 1, rcorr - 1)
    cv = float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0
    return cv, chi2_stat, p_val


def plot_categorical_grid(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    variables: list[str],
    outdir: str,
    ncols: int = 3,
    max_rows: int = 4,
) -> list[str]:
    paths = []
    per_page = ncols * max_rows

    for i in range(0, len(variables), per_page):
        chunk = variables[i : i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.5 * nrows))
        axes = axes.flatten()

        for ax, v in zip(axes, chunk):
            r_df = real[v].value_counts(normalize=True).rename("Percentage").reset_index()
            r_df["Dataset"] = "Real"
            s_df = synth[v].value_counts(normalize=True).rename("Percentage").reset_index()
            s_df["Dataset"] = "Synth"
            combined = pd.concat([r_df, s_df])

            sns.barplot(data=combined, x=v, y="Percentage", hue="Dataset",
                        ax=ax, palette=[COLOR_REAL, COLOR_SYNTH], alpha=0.8)
            try:
                r_vals = real[v].fillna("NA").astype(str)
                s_vals = synth[v].fillna("NA").astype(str)
                cv, chi2_stat, p_val = _cramers_v_chi2(r_vals, s_vals)
                p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.text(
                    0.02, 0.97, f"chi2={chi2_stat:.1f}, {p_str}\nCramér's V={cv:.3f}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
                )
            except Exception:
                pass
            ax.set_title(v, fontsize=9, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.legend(prop={"size": 8})

        for ax in axes[len(chunk):]:
            ax.axis("off")

        plt.subplots_adjust(hspace=0.6, wspace=0.35)
        path = os.path.join(outdir, f"cat_dist_{i}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


# -- Correlation matrices -----------------------------

def plot_correlation_comparison(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    vars_list: list[str],
    title_suffix: str,
    outdir: str,
) -> str | None:
    if not vars_list:
        return None

    def prep_corr(df, cols):
        temp = df[cols].copy()
        for col in temp.select_dtypes(include=["object", "str", "category"]).columns:
            temp[col] = pd.factorize(temp[col])[0].astype(float)
        return temp.corr()

    r_corr    = prep_corr(real, vars_list)
    s_corr    = prep_corr(synth, vars_list)
    diff_corr = (r_corr - s_corr).abs()

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    kw = dict(cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar=True, square=True)
    sns.heatmap(r_corr, ax=axes[0], **kw)
    axes[0].set_title(f"Real - {title_suffix}", fontsize=13, color=COLOR_REAL, fontweight="bold")
    sns.heatmap(s_corr, ax=axes[1], **kw)
    axes[1].set_title(f"Synthetic - {title_suffix}", fontsize=13, color=COLOR_SYNTH, fontweight="bold")
    sns.heatmap(diff_corr, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1, cbar=True, square=True,
                annot=(len(vars_list) <= 12), fmt=".2f", annot_kws={"size": 7})
    axes[2].set_title(f"|Real - Synthetic| - {title_suffix}", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(outdir, f"corr_{title_suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# -- PCA shared space ---------------------------------

def plot_pca_shared_space(
    real: pd.DataFrame, synth: pd.DataFrame, num_vars: list[str], outdir: str
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
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.35, s=8, color=COLOR_REAL,  label="Real")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.35, s=8, color=COLOR_SYNTH, label="Synthetic")
    ax.scatter(*Zr.mean(0), marker="X", s=120, color=COLOR_REAL,  zorder=5, edgecolors="white", lw=0.8)
    ax.scatter(*Zs.mean(0), marker="X", s=120, color=COLOR_SYNTH, zorder=5, edgecolors="white", lw=0.8)
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