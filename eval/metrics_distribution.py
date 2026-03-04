# ======================================================
# eval/metrics_distribution.py
# Statistical fidelity metrics: KS, Wasserstein, JS,
# categorical overlap, correlation distances, PCA scores
# ======================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# -- Distributional similarity ------------------------

def calculate_similarity_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    num_vars: list[str],
    cat_vars: list[str],
) -> dict:
    metrics = {}

    ks_scores, wass_scores, wass_exp_scores, js_scores = [], [], [], []

    for v in num_vars:
        r = real[v].dropna().values
        s = synth[v].dropna().values
        if len(r) < 2 or len(s) < 2:
            continue
        try:
            ks_stat, _ = ks_2samp(r, s)
            ks_scores.append(ks_stat)

            w = wasserstein_distance(r, s)
            wass_scores.append(w)
            scale = np.std(np.concatenate([r, s])) + 1e-9
            wass_exp_scores.append(np.exp(-w / scale))

            combined = np.concatenate([r, s])
            bins = np.histogram_bin_edges(combined, bins=50)
            r_hist, _ = np.histogram(r, bins=bins, density=True)
            s_hist, _ = np.histogram(s, bins=bins, density=True)
            r_hist = (r_hist + 1e-10) / (r_hist + 1e-10).sum()
            s_hist = (s_hist + 1e-10) / (s_hist + 1e-10).sum()
            js_scores.append(jensenshannon(r_hist, s_hist))
        except Exception:
            continue

    metrics["Avg Kolmogorov-Smirnov Distance ((lower) better)"] = (
        np.mean(ks_scores) if ks_scores else "N/A"
    )
    metrics["Avg Wasserstein Distance ((lower) better)"] = (
        np.mean(wass_scores) if wass_scores else "N/A"
    )
    metrics["Avg Wasserstein Exp Similarity [0-1] ((higher) better)"] = (
        np.mean(wass_exp_scores) if wass_exp_scores else "N/A"
    )
    metrics["Avg Jensen-Shannon Divergence [0-1] ((lower) better)"] = (
        np.mean(js_scores) if js_scores else "N/A"
    )

    # Categorical overlap
    cat_overlap = []
    for v in cat_vars:
        if v not in real.columns or v not in synth.columns:
            continue
        r_dist = real[v].value_counts(normalize=True)
        s_dist = synth[v].value_counts(normalize=True)
        all_cats = r_dist.index.union(s_dist.index)
        r_dist = r_dist.reindex(all_cats, fill_value=0)
        s_dist = s_dist.reindex(all_cats, fill_value=0)
        cat_overlap.append(np.sum(np.minimum(r_dist, s_dist)))

    metrics["Avg Categorical Overlap [0-1] ((higher) better)"] = (
        np.mean(cat_overlap) if cat_overlap else "N/A"
    )
    return metrics


# -- Correlation distances ----------------------------

def calculate_correlation_distance(
    real: pd.DataFrame, synth: pd.DataFrame, num_vars: list[str]
) -> float:
    """MAE between Pearson correlation matrices (continuous vars)."""
    if not num_vars:
        return 0.0
    r_corr = real[num_vars].corr().fillna(0).values
    s_corr = synth[num_vars].corr().fillna(0).values
    return float(np.mean(np.abs(r_corr - s_corr)))


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    confusion = pd.crosstab(x, y)
    chi2, _, _, _ = chi2_contingency(confusion)
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2corr = max(0, chi2 / (n + 1e-9) - ((k - 1) * (r - 1)) / (n - 1 + 1e-9))
    rcorr = r - ((r - 1) ** 2) / (n - 1 + 1e-9)
    kcorr = k - ((k - 1) ** 2) / (n - 1 + 1e-9)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def _build_assoc_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    mat = np.zeros((len(cols), len(cols)))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                mat[i, j] = 1.0
            elif j > i:
                v = _cramers_v(
                    df[ci].fillna("NA").astype(str),
                    df[cj].fillna("NA").astype(str),
                )
                mat[i, j] = mat[j, i] = v
    return mat


def calculate_categorical_correlation_distance(
    real: pd.DataFrame, synth: pd.DataFrame, cat_vars: list[str]
) -> float | str:
    """MAE between pairwise Cramér's V matrices (categorical vars)."""
    if not cat_vars or len(cat_vars) < 2:
        return "N/A"
    r_mat = _build_assoc_matrix(real, cat_vars)
    s_mat = _build_assoc_matrix(synth, cat_vars)
    return float(np.mean(np.abs(r_mat - s_mat)))


# -- PCA overlap scores -------------------------------

def calculate_pca_overlap_score(
    real: pd.DataFrame, synth: pd.DataFrame, num_vars: list[str]
) -> dict:
    if len(num_vars) < 2:
        return {
            "PCA Centroid Similarity [0-1] ((higher) better)": "N/A",
            "PCA Distribution Overlap [0-1] ((higher) better)": "N/A",
        }
    r = real[num_vars].dropna(how="all").fillna(0)
    s = synth[num_vars].dropna(how="all").fillna(0)
    if len(r) < 5 or len(s) < 5:
        return {
            "PCA Centroid Similarity [0-1] ((higher) better)": "N/A",
            "PCA Distribution Overlap [0-1] ((higher) better)": "N/A",
        }
    scaler = StandardScaler()
    Xr = scaler.fit_transform(r)
    Xs = scaler.transform(s)
    pca = PCA(n_components=2)
    Zr = pca.fit_transform(Xr)
    Zs = pca.transform(Xs)

    centroid_dist = np.linalg.norm(Zr.mean(axis=0) - Zs.mean(axis=0))
    scale = np.sqrt(Zr.var(axis=0).sum())
    centroid_sim = float(np.exp(-centroid_dist / (scale + 1e-9)))

    d_sr, _ = NearestNeighbors(n_neighbors=1).fit(Zr).kneighbors(Zs)
    d_rr, _ = NearestNeighbors(n_neighbors=2).fit(Zr).kneighbors(Zr)
    threshold  = np.median(d_rr[:, 1])
    overlap    = float(np.mean(d_sr[:, 0] <= threshold))

    return {
        "PCA Centroid Similarity [0-1] ((higher) better)": centroid_sim,
        "PCA Distribution Overlap [0-1] ((higher) better)": overlap,
    }


# -- Privacy metrics (DCR) ----------------------------

def privacy_metrics(
    real: pd.DataFrame, synth: pd.DataFrame, num_vars: list[str]
) -> dict:
    r = real[num_vars].dropna()
    s = synth[num_vars].dropna()
    n = min(len(r), len(s))
    scaler = StandardScaler()
    r_s = scaler.fit_transform(r.iloc[:n])
    s_s = scaler.transform(s.iloc[:n])
    nn   = NearestNeighbors(n_neighbors=2).fit(r_s)
    dist, _ = nn.kneighbors(s_s)
    dcr  = dist[:, 0] / (dist[:, 1] + 1e-8)
    return {
        "Mean DCR ((higher) better)":              float(np.mean(dcr)),
        "Median DCR ((higher) better)":            float(np.median(dcr)),
        "Fraction DCR < 1 ((lower) better)":      float(np.mean(dcr < 1)),
    }