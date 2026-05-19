# eval/plots.py 

import os
import math
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import ks_2samp as _ks

from scipy.stats import ks_2samp, chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


COLOR_REAL  = "#3B5998"
COLOR_SYNTH = "#FF7F50"

logger = logging.getLogger(__name__)


# utils
def _save(fig, path: str, dpi: int = 120) -> str:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _safe_mkdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _try_int(x):
    try: return int(float(x))
    except Exception: return None


def _decode_cat(series: pd.Series, inv_map: dict | None) -> pd.Series:
    if inv_map is None:
        return series.astype(str)
    def _d(x):
        if pd.isna(x): return "NA"
        k = _try_int(x)
        if k is not None and k in inv_map: return str(inv_map[k])
        return str(x)
    return series.map(_d)


# radar chart
def _radar(ax, values_dict: dict, title: str,
            overall: float, grade: str, color: str):
    """
    Radar chart.
    """
    labels = list(values_dict.keys())
    vals   = [float(v) if not np.isnan(float(v)) else 0.0
              for v in values_dict.values()]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    plot_v = vals + [vals[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])

    # Label radiali con sfondo bianco per evitare sovrapposizioni
    ax.set_xticklabels([])           # rimuove default
    for angle, label in zip(angles[:-1], labels):
        x = np.cos(angle - np.pi / 2)
        y = np.sin(angle - np.pi / 2)
        ha = "left" if x >= 0 else "right"
        va = "bottom" if y >= 0 else "top"
        ax.text(angle, 1.18, label,
                ha="center", va="center", size=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.85))

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.55, 0.70, 0.85, 1.0])
    ax.set_yticklabels(["0.25", "0.55", "0.70", "0.85", "1.0"],
                        size=5.5, color="gray")

    # Bande di sfondo
    for thr, col, alpha in [
        (1.00, "#d4f1d4", 0.22),
        (0.70, "#fff3cd", 0.33),
        (0.55, "#f8d7da", 0.38),
    ]:
        ax.fill(angles, [thr] * (N + 1), color=col, alpha=alpha, zorder=0)

    # Griglia radiale leggera
    for thr in [0.25, 0.55, 0.70, 0.85]:
        ax.plot(angles, [thr] * (N + 1), color="gray", lw=0.4, ls="--", zorder=1)

    # Dati
    ax.plot(angles, plot_v, "o-", lw=2.2, color=color, zorder=3)
    ax.fill(angles, plot_v, alpha=0.25, color=color, zorder=2)

    ov = f"{overall:.3f}" if not np.isnan(overall) else "N/A"
    ax.set_title(f"{title}\nOverall={ov}  {grade}",
                 size=9, fontweight="bold", pad=28)


def plot_scores_dashboard(sfs, lfs, tfs, outdir: str) -> str:
    fig = plt.figure(figsize=(21, 7))
    fig.suptitle("Synthetic Data Quality — Fidelity Three-Score Overview\n"
                 "(tutti gli assi: 1=perfetto, 0=pessimo)",
                 fontsize=12, fontweight="bold", y=1.02)
    configs = [
        (sfs, "Statistical Fidelity (SFS)", "#2A9D8F"),
        (lfs, "Longitudinal Fidelity (LFS)", "#E9C46A"),
        (tfs, "Temporal Fidelity (TFS)",     "#1a6faf"),
    ]
    for idx, (score_obj, title, color) in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx + 1, projection="polar")
        _radar(ax, score_obj.radar_values(), title,
               score_obj.overall, score_obj.grade, color)
    plt.tight_layout()
    path = os.path.join(outdir, "scores_dashboard.png")
    return _save(fig, path, dpi=150)


def plot_radar_component(scores_obj, title, outdir, filename,
                          color: str = "#2A9D8F"):
    """Radar chart generico per una singola sezione (stile unificato)."""
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    _radar(ax, scores_obj.radar_values(), title,
           scores_obj.overall, scores_obj.grade, color)
    plt.tight_layout()
    path = os.path.join(outdir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_summary_dashboard(sfs, lfs, tfs, outdir: str) -> str:
    """Executive summary: 3 radar SFS/LFS/TFS.
    """
    
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle("Synthetic Data Validation — Executive Summary",
                 fontsize=14, fontweight="bold", y=0.95)

    gs_radar = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3,
                                 top=0.9, bottom=0.55, left=0.05, right=0.95)
    colors_fidelity = ["#2A9D8F", "#E9C46A", "#1a6faf"]
    fills_fidelity  = ["#E6F5FF", "#FFF3CD", "#CDE6FF"]
    labels_fidelity = ["SFS", "LFS", "TFS"]

    for idx, (score_obj, title, color, fill, lbl) in enumerate(zip(
        [sfs, lfs, tfs],
        ["Statistical\nFidelity (SFS)", "Longitudinal\nFidelity (LFS)",
         "Temporal\nFidelity (TFS)"],
        colors_fidelity, fills_fidelity, labels_fidelity,
    )):
        ax = fig.add_subplot(gs_radar[idx], projection="polar")
        _radar(ax, score_obj.radar_values(), title,
               score_obj.overall, score_obj.grade, color)
        ax.text(0.5, 1.30,
                f"{lbl}: {score_obj.overall:.2f}  {score_obj.grade[:3]}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, fontweight="bold",
                bbox=dict(facecolor=fill, edgecolor="none", boxstyle="round,pad=0.35"))

    path = os.path.join(outdir, "summary_dashboard.png")
    return _save(fig, path, dpi=150)


# fidelity plots
def plot_numeric_grid(real, synth, variables, outdir,
                       ncols=3, max_rows=4):
    paths = []
    per_page = ncols * max_rows
    for i in range(0, len(variables), per_page):
        chunk = variables[i: i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = np.array(axes).flatten()
        for ax, v in zip(axes, chunk):
            r = pd.to_numeric(real[v],  errors="coerce").dropna()
            s = pd.to_numeric(synth[v], errors="coerce").dropna()
            sns.kdeplot(r, ax=ax, color=COLOR_REAL,  label="Real",  fill=True, alpha=0.3)
            sns.kdeplot(s, ax=ax, color=COLOR_SYNTH, label="Synth", fill=True, alpha=0.3)
            ax.legend(fontsize=8)
            if len(r) > 1 and len(s) > 1:
                ks, p = ks_2samp(r, s)
                ax.text(0.02, 0.97, f"KS={ks:.3f}, p={p:.3f}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            ax.set_title(v, fontsize=9, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.subplots_adjust(hspace=0.55, wspace=0.35)
        p = os.path.join(outdir, f"num_dist_{i}.png")
        paths.append(_save(fig, p))
    return paths


def _cramers_v_chi2(r, s):
    cats = sorted(set(r.unique()) | set(s.unique()))
    rc = r.value_counts().reindex(cats, fill_value=0)
    sc = s.value_counts().reindex(cats, fill_value=0)
    cont = np.array([rc.values, sc.values])
    chi2, pv, _, _ = chi2_contingency(cont)
    n = cont.sum(); k = cont.shape[1]; rd = cont.shape[0]
    phi2 = max(0, chi2 / (n + 1e-9) - ((k - 1) * (rd - 1)) / (n - 1 + 1e-9))
    rc_ = rd - (rd - 1) ** 2 / (n - 1 + 1e-9)
    kc_ = k  - (k  - 1) ** 2 / (n - 1 + 1e-9)
    denom = min(kc_ - 1, rc_ - 1)
    cv = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
    return cv, chi2, pv


def plot_categorical_grid(real, synth, variables, outdir,
                           inverse_maps=None, ncols=3, max_rows=4):
    inv = inverse_maps or {}
    paths = []
    per_page = ncols * max_rows
    for i in range(0, len(variables), per_page):
        chunk = variables[i: i + per_page]
        nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
        axes = np.array(axes).flatten()
        for ax, v in zip(axes, chunk):
            r_raw = real[v].astype(str).replace("nan", np.nan).dropna()
            s_raw = synth[v].astype(str).replace("nan", np.nan).dropna()
            all_cats = sorted(set(r_raw.unique()) | set(s_raw.unique()),
                              key=lambda c: (-r_raw.value_counts().get(c, 0), c))
            r_freq = r_raw.value_counts(normalize=True).reindex(all_cats, fill_value=0)
            s_freq = s_raw.value_counts(normalize=True).reindex(all_cats, fill_value=0)
            x = np.arange(len(all_cats)); w = 0.38
            ax.bar(x - w / 2, r_freq.values, w, color=COLOR_REAL,  alpha=0.8, label="Real")
            ax.bar(x + w / 2, s_freq.values, w, color=COLOR_SYNTH, alpha=0.8, label="Synth")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c)[:14] for c in all_cats], rotation=45,
                                ha="right", fontsize=7)
            ax.set_ylim(0, 1.0)
            ax.set_title(v, fontsize=9, fontweight="bold")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
            try:
                cv, chi2, pv = _cramers_v_chi2(r_raw, s_raw)
                p_str = f"p={pv:.3f}" if pv >= 0.001 else "p<0.001"
                ax.text(0.02, 0.97, f"chi2={chi2:.1f}, {p_str}\nV={cv:.3f}",
                        transform=ax.transAxes, fontsize=7, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            except Exception:
                pass
        for ax in axes[len(chunk):]:
            ax.axis("off")
        plt.subplots_adjust(hspace=0.65, wspace=0.35)
        p = os.path.join(outdir, f"cat_dist_{i}.png")
        paths.append(_save(fig, p))
    return paths


def plot_correlation_comparison(real, synth, vars_list, title_suffix, outdir):
    if not vars_list: return None
    def prep(df):
        tmp = df[vars_list].copy()
        for c in vars_list:
            if not pd.api.types.is_numeric_dtype(tmp[c]):
                tmp[c] = pd.factorize(tmp[c])[0].astype(float)
            else:
                tmp[c] = tmp[c].astype(float)
        return tmp.corr(method="pearson")
    rc, sc = prep(real), prep(synth)
    diff   = (rc - sc).abs()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    kw = dict(cmap="coolwarm", center=0, vmin=-1, vmax=1, cbar=True, square=True)
    sns.heatmap(rc,   ax=axes[0], **kw)
    axes[0].set_title(f"Real — {title_suffix}", color=COLOR_REAL, fontweight="bold")
    sns.heatmap(sc,   ax=axes[1], **kw)
    axes[1].set_title(f"Synthetic — {title_suffix}", color=COLOR_SYNTH, fontweight="bold")
    sns.heatmap(diff, ax=axes[2], cmap="YlOrRd", vmin=0, vmax=1, cbar=True, square=True,
                annot=(len(vars_list) <= 12), fmt=".2f", annot_kws={"size": 7})
    axes[2].set_title(f"|Real - Synth| — {title_suffix}", fontweight="bold")
    plt.tight_layout()
    p = os.path.join(outdir, f"corr_{title_suffix}.png")
    return _save(fig, p, dpi=150)


def plot_pca_shared_space(real, synth, num_vars, outdir):
    r = real[num_vars].dropna(how="all").fillna(0)
    s = synth[num_vars].dropna(how="all").fillna(0)
    if len(r) < 2 or len(s) < 2: return None
    scaler = StandardScaler()
    Xr = scaler.fit_transform(r); Xs = scaler.transform(s)
    pca = PCA(n_components=2, random_state=42)
    Zr  = pca.fit_transform(Xr); Zs = pca.transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.3, s=8, color=COLOR_REAL,  label="Real")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.3, s=8, color=COLOR_SYNTH, label="Synthetic")
    ax.scatter(*Zr.mean(0), marker="X", s=120, color=COLOR_REAL,  zorder=5, edgecolors="white")
    ax.scatter(*Zs.mean(0), marker="X", s=120, color=COLOR_SYNTH, zorder=5, edgecolors="white")
    ve = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ve[0]:.1%} var)"); ax.set_ylabel(f"PC2 ({ve[1]:.1%} var)")
    ax.set_title("PCA Shared Space"); ax.legend(); plt.tight_layout()
    p = os.path.join(outdir, "pca_shared.png")
    return _save(fig, p, dpi=150)


def _dtw_feature_matrix(df, temporal_vars, time_col, patient_col, n_lags=5):
    pids, rows = [], []
    for pid, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        feats = []
        for v in temporal_vars:
            y = pd.to_numeric(g[v], errors="coerce").dropna().values
            if len(y) == 0: feats.extend([0.0] * 7); continue
            feats.append(float(np.mean(y)))
            feats.append(float(np.std(y)) if len(y) > 1 else 0.0)
            feats.append(float(np.corrcoef(y[:-1], y[1:])[0, 1]) if len(y) > 2 else 0.0)
            feats.append(float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]) if len(y) > 1 else 0.0)
            feats.append(float(np.percentile(y, 10)))
            feats.append(float(np.percentile(y, 50)))
            feats.append(float(np.percentile(y, 90)))
        rows.append(feats); pids.append(pid)
    mat = np.array(rows, dtype=np.float32)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, pids


def plot_umap(real, synth, num_vars, outdir, color_vars=None,
              inverse_maps=None, temporal_vars=None, time_col=None, patient_col=None):
    try:
        import umap as umap_lib
    except ImportError:
        logger.warning("umap-learn non installato, UMAP saltato.")
        return []

    use_dtw = (temporal_vars is not None and time_col is not None
               and patient_col is not None and len(temporal_vars) > 0)
    if use_dtw:
        Xr_raw, r_pids = _dtw_feature_matrix(real,  temporal_vars, time_col, patient_col)
        Xs_raw, s_pids = _dtw_feature_matrix(synth, temporal_vars, time_col, patient_col)
        if len(Xr_raw) < 5 or len(Xs_raw) < 5: return []
        scaler = StandardScaler()
        Xr = scaler.fit_transform(Xr_raw); Xs = scaler.transform(Xs_raw)
        real_pp  = real.groupby(patient_col, sort=False).first().loc[r_pids].reset_index()
        synth_pp = synth.groupby(patient_col, sort=False).first().loc[s_pids].reset_index()
        label_suffix = "(DTW features)"
    else:
        r = real[num_vars].dropna(how="all").fillna(0)
        s = synth[num_vars].dropna(how="all").fillna(0)
        if len(r) < 5 or len(s) < 5: return []
        scaler = StandardScaler()
        Xr = scaler.fit_transform(r); Xs = scaler.transform(s)
        real_pp = real.iloc[r.index]; synth_pp = synth.iloc[s.index]
        label_suffix = "(static features)"

    reducer = umap_lib.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Zr = reducer.fit_transform(Xr); Zs = reducer.transform(Xs)

    paths = []
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.3, s=8, color=COLOR_REAL,  label=f"Real (n={len(Zr)})")
    ax.scatter(Zs[:, 0], Zs[:, 1], alpha=0.3, s=8, color=COLOR_SYNTH, label=f"Synth (n={len(Zs)})")
    ax.scatter(*Zr.mean(0), marker="X", s=140, color=COLOR_REAL,  zorder=5, edgecolors="white", linewidths=1.5)
    ax.scatter(*Zs.mean(0), marker="X", s=140, color=COLOR_SYNTH, zorder=5, edgecolors="white", linewidths=1.5)
    ax.set_title(f"UMAP Projection {label_suffix}\nfit on Real — crosses = centroids")
    ax.legend(fontsize=9); plt.tight_layout()
    p = os.path.join(outdir, "umap_plain.png")
    paths.append(_save(fig, p, dpi=150))

    inv = inverse_maps or {}
    for cv in (color_vars or []):
        if cv not in real_pp.columns: continue
        try:
            decoded_r = _decode_cat(real_pp[cv],  inv.get(cv))
            decoded_s = _decode_cat(synth_pp[cv], inv.get(cv))
            cats = sorted(set(decoded_r.unique()) | set(decoded_s.unique()))
            custom_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                              "#9467bd", "#8c564b", "#e377c2", "#bcbd22"]
            cmap    = mcolors.ListedColormap(custom_colors[:len(cats)])
            cat2idx = {c: i for i, c in enumerate(cats)}
            fig_comb, ax_comb = plt.subplots(figsize=(10, 8))
            for dataset_name, (Z, col_vals, marker) in [
                ("Real",      (Zr, decoded_r, "o")),
                ("Synthetic", (Zs, decoded_s, "x")),
            ]:
                for cat in cats:
                    mask = (col_vals == cat).values
                    if mask.sum() > 0:
                        ax_comb.scatter(Z[mask, 0], Z[mask, 1],
                                        marker=marker, s=22, alpha=0.6,
                                        color=cmap(cat2idx[cat]),
                                        edgecolors="white", linewidth=0.4,
                                        label=f"{dataset_name}: {cat}")
            ax_comb.scatter(*Zr.mean(0), marker="P", s=200, color=COLOR_REAL,  zorder=10, edgecolors="black", lw=2)
            ax_comb.scatter(*Zs.mean(0), marker="X", s=200, color=COLOR_SYNTH, zorder=10, edgecolors="black", lw=2)
            ax_comb.set_title(f"UMAP {label_suffix} — Real (o) vs Synthetic (x)\nColored by {cv}",
                               fontsize=12, fontweight="bold")
            ax_comb.legend(fontsize=8, markerscale=1.5, loc="best"); plt.tight_layout()
            p_comb = os.path.join(outdir, f"umap_{cv}_combined.png")
            paths.append(_save(fig_comb, p_comb, dpi=150))
        except Exception as e:
            logger.warning("UMAP color plot for %s failed: %s", cv, e)
    return paths


def plot_trajectory_mean_ci(real, synth, var, time_col, patient_col,
                              outdir, n_grid=60, min_patients=5):
    def _interp(df):
        t_max = pd.to_numeric(df[time_col], errors="coerce").max()
        grid  = np.linspace(0, t_max, n_grid)
        vals  = []
        for _, g in df.groupby(patient_col):
            g = g.sort_values(time_col)
            t = pd.to_numeric(g[time_col], errors="coerce").values
            y = pd.to_numeric(g[var],      errors="coerce").values
            mask = ~(np.isnan(t) | np.isnan(y))
            if mask.sum() >= 2: vals.append(np.interp(grid, t[mask], y[mask]))
        if not vals: return grid, None, None, None
        mat = np.vstack(vals)
        m   = np.nanmean(mat, axis=0)
        se  = 1.96 * np.nanstd(mat, axis=0) / np.sqrt(np.sum(~np.isnan(mat), axis=0).clip(1))
        return grid, m, se, None
    gr, rm, rse, _ = _interp(real)
    _,  sm, sse, _ = _interp(synth)
    if rm is None or sm is None: return None
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gr, rm, color=COLOR_REAL,  lw=2, label=f"Real (n={real[patient_col].nunique()})")
    ax.fill_between(gr, rm - rse, rm + rse, color=COLOR_REAL,  alpha=0.2)
    ax.plot(gr, sm, color=COLOR_SYNTH, lw=2, label=f"Synth (n={synth[patient_col].nunique()})")
    ax.fill_between(gr, sm - sse, sm + sse, color=COLOR_SYNTH, alpha=0.2)
    ax.set_title(f"Trajectory — {var}"); ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel(var); ax.legend(fontsize=8); plt.tight_layout()
    p = os.path.join(outdir, f"traj_{var}.png")
    return _save(fig, p)


def plot_sampled_trajectories(real, synth, var, time_col, patient_col,
                               outdir, n_sample=20):
    rng = np.random.default_rng(42)
    def _sample(df, n):
        pids = df[patient_col].unique()
        return rng.choice(pids, size=min(n, len(pids)), replace=False)
    r_pids = _sample(real,  n_sample // 2)
    s_pids = _sample(synth, n_sample // 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(r_pids):
        g = real[real[patient_col] == pid].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").values
        y = pd.to_numeric(g[var],      errors="coerce").values
        ax.plot(t, y, color=COLOR_REAL, alpha=0.4, lw=1, label="Real" if i == 0 else "")
    for i, pid in enumerate(s_pids):
        g = synth[synth[patient_col] == pid].sort_values(time_col)
        t = pd.to_numeric(g[time_col], errors="coerce").values
        y = pd.to_numeric(g[var],      errors="coerce").values
        ax.plot(t, y, color=COLOR_SYNTH, alpha=0.4, lw=1, label="Synth" if i == 0 else "")
    ax.set_title(f"Sample Trajectories ({n_sample}) — {var}")
    ax.set_xlabel(f"Time ({time_col})"); ax.set_ylabel(var)
    ax.legend(fontsize=9); plt.tight_layout()
    p = os.path.join(outdir, f"traj_sample_{var}.png")
    return _save(fig, p)


def plot_variance_grid(real, synth, temporal_vars, patient_col, outdir):
    paths = []
    per_page = 6
    for i in range(0, len(temporal_vars), per_page):
        chunk = temporal_vars[i: i + per_page]
        ncols = 3; nrows = math.ceil(len(chunk) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows))
        axes = np.array(axes).flatten()
        for ax, v in zip(axes, chunk):
            def _var(df):
                out = []
                for _, g in df.groupby(patient_col):
                    y = pd.to_numeric(g[v], errors="coerce").dropna()
                    if len(y) >= 2: out.append(float(np.var(y, ddof=1)))
                return np.array(out)
            rv, sv = _var(real), _var(synth)
            if len(rv) > 1 and len(sv) > 1:
                ks, _ = ks_2samp(rv, sv)
                sns.kdeplot(rv, ax=ax, color=COLOR_REAL,  label="Real",  fill=True, alpha=0.3)
                sns.kdeplot(sv, ax=ax, color=COLOR_SYNTH, label="Synth", fill=True, alpha=0.3)
                ax.text(0.02, 0.97, f"KS={ks:.3f}", transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            ax.set_title(f"Intra-var — {v}", fontsize=9, fontweight="bold")
            ax.legend(fontsize=8)
        for ax in axes[len(chunk):]: ax.axis("off")
        plt.subplots_adjust(hspace=0.55, wspace=0.35)
        p = os.path.join(outdir, f"variance_intra_{i}.png")
        paths.append(_save(fig, p))
    return paths


def plot_inter_patient_variance(real, synth, temporal_vars, patient_col, time_col, outdir):
    n_bins = 6
    n_cols = min(3, len(temporal_vars))
    n_rows = math.ceil(len(temporal_vars) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    for ax, v in zip(axes, temporal_vars):
        r_times = pd.to_numeric(real[time_col], errors="coerce").dropna()
        t_min, t_max = float(r_times.min()), float(r_times.max())
        bin_edges = np.linspace(t_min, t_max, n_bins + 1)
        bin_labels = [f"{bin_edges[k]:.0f}-{bin_edges[k+1]:.0f}" for k in range(n_bins)]
        def _inter(df, edges):
            df2 = df[[patient_col, time_col, v]].copy()
            df2[time_col] = pd.to_numeric(df2[time_col], errors="coerce")
            df2[v]        = pd.to_numeric(df2[v],        errors="coerce")
            df2 = df2.dropna()
            df2["_bin"] = pd.cut(df2[time_col], bins=edges, include_lowest=True, labels=bin_labels)
            per_pat = df2.groupby(["_bin", patient_col], observed=True)[v].mean()
            inter   = per_pat.groupby(level=0, observed=True).std()
            return inter.reindex(bin_labels).fillna(0.0)
        ri = _inter(real, bin_edges); si = _inter(synth, bin_edges)
        x = np.arange(len(bin_labels))
        ax.bar(x - 0.2, ri.values, 0.4, color=COLOR_REAL,  alpha=0.8, label="Real")
        ax.bar(x + 0.2, si.values, 0.4, color=COLOR_SYNTH, alpha=0.8, label="Synth")
        ax.set_xticks(x); ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"Inter-patient std — {v}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(temporal_vars):]: ax.axis("off")
    plt.tight_layout()
    p = os.path.join(outdir, "variance_inter.png")
    return _save(fig, p)


def plot_lme_slopes(lme_betas, outdir):
    if not lme_betas: return None
    vars_list = [v for v, b in lme_betas.items()
                 if not (np.isnan(b.get("beta_real", float("nan"))) and
                         np.isnan(b.get("beta_synth", float("nan"))))]
    if not vars_list: return None
    x = np.arange(len(vars_list)); w = 0.35
    betas_r = [lme_betas[v].get("beta_real",  float("nan")) for v in vars_list]
    betas_s = [lme_betas[v].get("beta_synth", float("nan")) for v in vars_list]
    fig, ax = plt.subplots(figsize=(max(8, len(vars_list) * 1.2), 5))
    ax.bar(x - w / 2, betas_r, w, color=COLOR_REAL,  alpha=0.8, label="Real (LME β)")
    ax.bar(x + w / 2, betas_s, w, color=COLOR_SYNTH, alpha=0.8, label="Synth (LME β)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(vars_list, rotation=30, ha="right")
    ax.set_ylabel("Fixed-effect slope (β time)"); ax.set_title("LME Slope Comparison")
    ax.legend(); plt.tight_layout()
    p = os.path.join(outdir, "lme_slopes.png")
    return _save(fig, p)


def plot_autocorrelation_lagk(real, synth, temporal_vars, time_col, patient_col,
                                outdir, k_max=5):
    from eval.metrics import _autocorrelation_lagk as _acf
    paths = []
    for v in temporal_vars:
        fig, axes = plt.subplots(1, k_max, figsize=(4 * k_max, 4), sharey=False)
        for k, ax in enumerate(axes, start=1):
            ar  = _acf(real,  v, time_col, patient_col, k=k)
            as_ = _acf(synth, v, time_col, patient_col, k=k)
            if len(ar) > 1 and len(as_) > 1:
                sns.kdeplot(ar,  ax=ax, color=COLOR_REAL,  fill=True, alpha=0.3, label="Real")
                sns.kdeplot(as_, ax=ax, color=COLOR_SYNTH, fill=True, alpha=0.3, label="Synth")
                ks, _ = ks_2samp(ar, as_)
                ax.text(0.02, 0.97, f"KS={ks:.3f}", transform=ax.transAxes, fontsize=8, va="top",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            ax.set_title(f"Lag-{k}", fontsize=9); ax.legend(fontsize=7)
        fig.suptitle(f"Autocorrelation lag-1..{k_max} — {v}", fontweight="bold")
        plt.tight_layout()
        p = os.path.join(outdir, f"autocorr_{v}.png")
        paths.append(_save(fig, p))
    return paths


def plot_variable_by_visit(real, synth, temporal_vars, time_col, patient_col,
                            outdir, n_bins=6, max_vars_per_page=3):
    t_max = max(pd.to_numeric(real[time_col],  errors="coerce").max(),
                pd.to_numeric(synth[time_col], errors="coerce").max())
    bins  = np.linspace(0, t_max, n_bins + 1)
    paths = []
    for i in range(0, len(temporal_vars), max_vars_per_page):
        chunk = temporal_vars[i: i + max_vars_per_page]
        fig, axes_all = plt.subplots(len(chunk), 2, figsize=(14, 5 * len(chunk)))
        if len(chunk) == 1: axes_all = [axes_all]
        fig.suptitle(f"Variables by visit time bin (0-{t_max:.0f} months, {n_bins} bins)",
                     fontweight="bold")
        for row, v in enumerate(chunk):
            ax_bp, ax_ks = axes_all[row]
            r_binned = real.copy();  s_binned = synth.copy()
            r_binned["_bin"] = pd.cut(pd.to_numeric(real[time_col],  errors="coerce"), bins)
            s_binned["_bin"] = pd.cut(pd.to_numeric(synth[time_col], errors="coerce"), bins)
            ks_per_bin, bin_labels = [], []
            for b in sorted(r_binned["_bin"].cat.categories):
                rv = pd.to_numeric(r_binned[r_binned["_bin"] == b][v], errors="coerce").dropna()
                sv = pd.to_numeric(s_binned[s_binned["_bin"] == b][v], errors="coerce").dropna()
                label = f"{b.left:.0f}-{b.right:.0f}"; bin_labels.append(label)
                if len(rv) > 1 and len(sv) > 1:
                    ks, _ = ks_2samp(rv, sv); ks_per_bin.append(ks)
                    pos_r = list(r_binned["_bin"].cat.categories).index(b) * 2
                    ax_bp.boxplot(rv, positions=[pos_r],       widths=0.6, patch_artist=True,
                                  boxprops=dict(facecolor=COLOR_REAL, alpha=0.6))
                    ax_bp.boxplot(sv, positions=[pos_r + 0.7], widths=0.6, patch_artist=True,
                                  boxprops=dict(facecolor=COLOR_SYNTH, alpha=0.6))
                else:
                    ks_per_bin.append(float("nan"))
            ax_bp.set_title(f"{v} — boxplot per bin", fontsize=9); ax_bp.set_ylabel(v)
            colors_ks = ["green" if k < 0.15 else "orange" if k < 0.30 else "red"
                         for k in ks_per_bin if not np.isnan(k)]
            valid = [k for k in ks_per_bin if not np.isnan(k)]
            ax_ks.bar(range(len(valid)), valid, color=colors_ks)
            ax_ks.axhline(0.15, ls="--", color="orange", lw=1)
            ax_ks.axhline(0.30, ls="--", color="red",    lw=1)
            ax_ks.set_xticks(range(len(valid)))
            ax_ks.set_xticklabels(bin_labels[:len(valid)], rotation=30, ha="right")
            ax_ks.set_ylim(0, 1); ax_ks.set_title(f"{v} — KS per bin", fontsize=9)
        plt.tight_layout()
        p = os.path.join(outdir, f"var_by_visit_{i}.png")
        paths.append(_save(fig, p))
    return paths


def plot_visit_distribution(real, synth, patient_col, outdir):
    vc_r = real.groupby(patient_col).size()
    vc_s = synth.groupby(patient_col).size()
    ks, _ = ks_2samp(vc_r.values, vc_s.values)
    all_v  = sorted(set(vc_r.unique()) | set(vc_s.unique()))
    r_freq = vc_r.value_counts(normalize=True).reindex(all_v, fill_value=0)
    s_freq = vc_s.value_counts(normalize=True).reindex(all_v, fill_value=0)
    x = np.arange(len(all_v)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(all_v) * 0.6), 4))
    ax.bar(x - w / 2, r_freq.values, w, color=COLOR_REAL,  alpha=0.8, label="Real")
    ax.bar(x + w / 2, s_freq.values, w, color=COLOR_SYNTH, alpha=0.8, label="Synth")
    ax.set_xticks(x); ax.set_xticklabels(all_v)
    ax.set_xlabel("Number of visits"); ax.set_ylabel("Fraction")
    ax.set_title(f"Visit count distribution  KS={ks:.3f}")
    ax.legend(); plt.tight_layout()
    p = os.path.join(outdir, "visit_distribution.png")
    return _save(fig, p)


def plot_visit_position_timing(real, synth, time_col, patient_col, outdir, n_pos=12):
    def _extract(df, max_p):
        pt: dict = {p: [] for p in range(1, max_p + 1)}
        for _, g in df.groupby(patient_col):
            ts = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            for idx, t in enumerate(ts[:max_p]): pt[idx + 1].append(float(t))
        return {k: np.array(v) for k, v in pt.items() if len(v) >= 3}
    rp = _extract(real, n_pos); sp = _extract(synth, n_pos)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    positions = sorted(set(rp) & set(sp))
    rm = [np.mean(rp[p]) for p in positions]; sm = [np.mean(sp[p]) for p in positions]
    ax1.plot(positions, rm, "o-",  color=COLOR_REAL,  label="Real mean")
    ax1.plot(positions, sm, "s--", color=COLOR_SYNTH, label="Synth mean")
    ax1.fill_between(positions,
                     [np.percentile(rp[p], 25) for p in positions],
                     [np.percentile(rp[p], 75) for p in positions], alpha=0.2, color=COLOR_REAL)
    ax1.fill_between(positions,
                     [np.percentile(sp[p], 25) for p in positions],
                     [np.percentile(sp[p], 75) for p in positions], alpha=0.2, color=COLOR_SYNTH)
    ax1.set_xlabel("Visit position"); ax1.set_ylabel("Months from baseline")
    ax1.set_title("Mean visit time per sequential position (IQR shaded)"); ax1.legend()
    ks_vals = []
    for p in positions:
        if len(rp.get(p, [])) >= 2 and len(sp.get(p, [])) >= 2:
            ks_vals.append(ks_2samp(rp[p], sp[p])[0])
        else: ks_vals.append(float("nan"))
    colors = ["green" if k < 0.15 else "orange" if k < 0.30 else "red"
              for k in ks_vals if not np.isnan(k)]
    valid_ks = [k for k in ks_vals if not np.isnan(k)]
    ax2.bar(range(len(valid_ks)), valid_ks, color=colors)
    ax2.axhline(0.15, ls="--", color="orange", lw=1); ax2.axhline(0.30, ls="--", color="red", lw=1)
    ax2.set_xticks(range(len(valid_ks)))
    ax2.set_xticklabels([f"pos {p}" for p in positions[:len(valid_ks)]], rotation=30)
    ax2.set_title("KS distance per visit position")
    plt.tight_layout()
    p_overview = os.path.join(outdir, "visit_position_overview.png")
    paths = [_save(fig, p_overview)]
    ncols = min(n_pos, 4); nrows = math.ceil(n_pos / ncols)
    fig2, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, pos in zip(axes, sorted(set(rp) & set(sp))[:n_pos]):
        if len(rp[pos]) > 1: sns.kdeplot(rp[pos], ax=ax, color=COLOR_REAL,  fill=True, alpha=0.3, label="Real")
        if len(sp[pos]) > 1: sns.kdeplot(sp[pos], ax=ax, color=COLOR_SYNTH, fill=True, alpha=0.3, label="Synth")
        ks, _ = ks_2samp(rp[pos], sp[pos]) if (len(rp[pos]) > 1 and len(sp[pos]) > 1) else (float("nan"), None)
        ax.set_title(f"Visit pos {pos}  KS={ks:.3f}" if not np.isnan(ks) else f"Visit pos {pos}")
        ax.set_xlabel("Months"); ax.legend(fontsize=8)
    for ax in axes[n_pos:]: ax.axis("off")
    plt.suptitle(f"KDE — first {n_pos} visit positions", fontweight="bold"); plt.tight_layout()
    p_kde = os.path.join(outdir, "visit_position_kde.png")
    paths.append(_save(fig2, p_kde))
    return paths


def plot_last_visit_vs_fup(real_raw, synth, time_col, patient_col, fup_col,
                            outdir, death_col="DEATH"):
    paths = []
    def _last(df): return df.groupby(patient_col)[time_col].apply(lambda s: pd.to_numeric(s, errors="coerce").max()).rename("last")
    def _fup(df):
        if fup_col not in df.columns: return None
        return df.groupby(patient_col)[fup_col].first().rename("fup")
    def _death(df):
        if death_col not in df.columns: return None
        return (df.groupby(patient_col)[death_col].max()
                .pipe(lambda s: pd.to_numeric(s, errors="coerce").fillna(0)).astype(int).rename("death"))
    def _build(df):
        parts = [_last(df)]
        f = _fup(df)
        if f is not None: parts.append(f)
        d = _death(df)
        if d is not None: parts.append(d)
        m = pd.concat(parts, axis=1).dropna(subset=["last"])
        if "fup" in m.columns:
            m["fup"] = pd.to_numeric(m["fup"], errors="coerce")
            m = m[m["fup"] > 0]
            m["gap"] = (m["fup"] - m["last"]).clip(lower=0)
            m["cov"] = (m["last"] / m["fup"]).clip(0, 1)
        return m
    mr = _build(real_raw); ms = _build(synth)
    has_fup   = "fup"   in mr.columns and "fup"   in ms.columns
    has_death = "death" in mr.columns and "death" in ms.columns
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    fig1.suptitle("t_last_obs vs t_FUP  (scatter per paziente)", fontweight="bold")
    for ax, (m, label, color) in zip(axes1, [(mr, "Real", COLOR_REAL), (ms, "Synthetic", COLOR_SYNTH)]):
        if has_fup:
            if has_death:
                for dval, marker, alpha in [(0, "o", 0.35), (1, "*", 0.7)]:
                    sub = m[m["death"] == dval]
                    ax.scatter(sub["fup"], sub["last"], s=10, alpha=alpha, marker=marker, color=color, label=f"DEATH={dval} (n={len(sub)})")
            else:
                ax.scatter(m["fup"], m["last"], s=8, alpha=0.35, color=color)
            mx = max(float(m["fup"].max()), float(m["last"].max()))
            ax.plot([0, mx], [0, mx], "k--", lw=1, label="y=x")
            med_cov = float(m["cov"].median()) if "cov" in m.columns else float("nan")
            ax.set_title(f"{label}  median coverage={med_cov:.2f}")
        else:
            ax.hist(m["last"].values, bins=30, color=color, alpha=0.7)
            ax.set_title(f"{label} — t_last distribution")
        ax.set_xlabel(f"t_FUP [{fup_col}]"); ax.set_ylabel(f"t_last_obs [{time_col}]"); ax.legend(fontsize=8)
    plt.tight_layout()
    paths.append(_save(fig1, os.path.join(outdir, "last_visit_fup_scatter.png")))
    if has_fup:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Gap = t_FUP - t_last_obs", fontweight="bold")
        for ax, (m, label) in zip(axes2, [(mr, "Real"), (ms, "Synthetic")]):
            if has_death:
                for dval, ls, lbl in [(0, "-", "DEATH=0"), (1, "--", "DEATH=1")]:
                    sub = m[m["death"] == dval]["gap"].dropna()
                    if len(sub) > 1: sns.kdeplot(sub, ax=ax, label=f"{lbl} n={len(sub)}", linestyle=ls, fill=True, alpha=0.2)
            else:
                sns.kdeplot(m["gap"].dropna(), ax=ax, fill=True, alpha=0.3)
            ax.set_xlabel("Gap (months)"); ax.set_title(label); ax.legend(fontsize=8); ax.set_xlim(left=0)
        gap_r = mr["gap"].dropna().values; gap_s = ms["gap"].dropna().values
        if len(gap_r) > 1 and len(gap_s) > 1:
            ks, _ = ks_2samp(gap_r, gap_s)
            fig2.text(0.5, 0.02, f"KS(gap) = {ks:.3f}", ha="center", fontsize=10,
                      bbox=dict(boxstyle="round", facecolor="lightyellow"))
        plt.tight_layout(); paths.append(_save(fig2, os.path.join(outdir, "last_visit_fup_gap.png")))
    lr = mr["last"].dropna().values; ls_ = ms["last"].dropna().values
    ks_last, _ = ks_2samp(lr, ls_) if (len(lr) > 1 and len(ls_) > 1) else (float("nan"), None)
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    if len(lr) > 1: sns.kdeplot(lr, ax=ax3, color=COLOR_REAL,  fill=True, alpha=0.3, label=f"Real  median={np.median(lr):.1f}")
    if len(ls_) > 1: sns.kdeplot(ls_, ax=ax3, color=COLOR_SYNTH, fill=True, alpha=0.3, label=f"Synth median={np.median(ls_):.1f}")
    ax3.set_title(f"t_last_obs distribution  KS={ks_last:.3f}"); ax3.set_xlabel("t_last_obs (months)"); ax3.legend(fontsize=9); plt.tight_layout()
    paths.append(_save(fig3, os.path.join(outdir, "last_visit_kde.png")))
    return paths


def plot_km_followup(real_raw, synth, time_col, patient_col, fup_col, outdir):
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError: return None
    def _prep_fup(df):
        if fup_col and fup_col in df.columns: t = df.groupby(patient_col)[fup_col].first()
        else: t = df.groupby(patient_col)[time_col].max()
        return pd.to_numeric(t, errors="coerce").dropna().values
    tr = _prep_fup(real_raw); ts = _prep_fup(synth)
    ev = np.zeros(len(tr)); es = np.zeros(len(ts))
    try: result = logrank_test(tr, ts, event_observed_A=ev, event_observed_B=es); p_val = result.p_value
    except Exception: p_val = float("nan")
    fig, ax = plt.subplots(figsize=(9, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(tr, ev, label=f"Real (n={len(tr)})"); kmf.plot_survival_function(ax=ax, color=COLOR_REAL)
    kmf.fit(ts, es, label=f"Synthetic (n={len(ts)})"); kmf.plot_survival_function(ax=ax, color=COLOR_SYNTH, ls="--")
    p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    ax.text(0.98, 0.98, f"Log-rank p={p_str}", transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("KM — Follow-up duration"); ax.set_xlabel("Months"); plt.tight_layout()
    return _save(fig, os.path.join(outdir, "km_followup.png"))


def plot_km_event(real_raw, synth, time_col, patient_col, fup_col, event_col, outdir):
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError: return None
    def _prep(df):
        if fup_col and fup_col in df.columns: t = df.groupby(patient_col)[fup_col].first().rename("t")
        else: t = df.groupby(patient_col)[time_col].max().rename("t")
        t = pd.to_numeric(t, errors="coerce")
        if event_col not in df.columns: ev = pd.Series(0, index=t.index, name="ev")
        else: ev = (df.groupby(patient_col)[event_col].max().pipe(lambda s: pd.to_numeric(s, errors="coerce").fillna(0)).astype(int).rename("ev"))
        m = pd.concat([t, ev], axis=1).dropna()
        return m["t"].values, m["ev"].values
    tr, er = _prep(real_raw); ts, es = _prep(synth)
    if len(tr) < 3 or len(ts) < 3: return None
    try: result = logrank_test(tr, ts, event_observed_A=er, event_observed_B=es); p_val = result.p_value
    except Exception: p_val = float("nan")
    fig, ax = plt.subplots(figsize=(9, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(tr, er, label=f"Real (n={len(tr)}, ev={int(er.sum())})"); kmf.plot_survival_function(ax=ax, color=COLOR_REAL)
    kmf.fit(ts, es, label=f"Synth (n={len(ts)}, ev={int(es.sum())})"); kmf.plot_survival_function(ax=ax, color=COLOR_SYNTH, ls="--")
    p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
    ax.text(0.98, 0.98, f"Log-rank p={p_str}", transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title(f"KM — {event_col}"); ax.set_xlabel("Months"); plt.tight_layout()
    return _save(fig, os.path.join(outdir, f"km_{event_col.lower()}.png"))


def plot_km_overall(real_raw, synth, time_col, patient_col, outdir):
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError: return None
    def _prepare(df):
        last = df.groupby(patient_col)[time_col].apply(lambda s: pd.to_numeric(s, errors="coerce").max())
        return last.dropna().values
    lr = _prepare(real_raw); ls_ = _prepare(synth)
    ev_r = np.ones(len(lr)); ev_s = np.ones(len(ls_))
    result = logrank_test(lr, ls_, event_observed_A=ev_r, event_observed_B=ev_s)
    fig, ax = plt.subplots(figsize=(9, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(lr, ev_r, label=f"Real (n={len(lr)})"); kmf.plot_survival_function(ax=ax, color=COLOR_REAL)
    kmf.fit(ls_, ev_s, label=f"Synthetic (n={len(ls_)})"); kmf.plot_survival_function(ax=ax, color=COLOR_SYNTH, ls="--")
    ax.text(0.98, 0.98, f"Log-rank p={result.p_value:.4f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.set_title("Kaplan-Meier Overall"); plt.tight_layout()
    return _save(fig, os.path.join(outdir, "km_overall.png"))


def plot_km_poise(real_raw, synth, time_col, patient_col, alp_col="ALP", bil_col="BIL",
                  death_col="DEATH", transp_col="TRANSP", outdir="."):
    try: from lifelines import KaplanMeierFitter
    except ImportError: return None
    def _classify(df):
        v12 = df[(pd.to_numeric(df[time_col], errors="coerce") >= 11) &
                 (pd.to_numeric(df[time_col], errors="coerce") <= 13)].copy()
        v12[alp_col] = pd.to_numeric(v12[alp_col], errors="coerce")
        v12[bil_col] = pd.to_numeric(v12[bil_col], errors="coerce")
        return v12.groupby(patient_col).apply(
            lambda g: bool((g[alp_col].mean() <= 2) and (g[bil_col].mean() <= 1)))
    def _survival(df):
        last = df.groupby(patient_col)[time_col].apply(lambda s: pd.to_numeric(s, errors="coerce").max())
        ev = pd.Series(False, index=last.index)
        for col in [death_col, transp_col]:
            if col in df.columns:
                flag = df.groupby(patient_col)[col].max()
                ev   = ev | (pd.to_numeric(flag, errors="coerce").fillna(0) > 0)
        return last, ev.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (df, label) in zip(axes, [(real_raw, "Real"), (synth, "Synthetic")]):
        resp  = _classify(df); t, ev = _survival(df)
        data  = pd.DataFrame({"time": t, "event": ev}).join(resp.rename("responder")).dropna()
        kmf   = KaplanMeierFitter()
        for is_resp, lbl, color in [(True, "Responder", "green"), (False, "Non-responder", "red")]:
            sub = data[data["responder"] == is_resp]
            if len(sub) < 3: continue
            kmf.fit(sub["time"], sub["event"], label=f"{lbl} (n={len(sub)})")
            kmf.plot_survival_function(ax=ax, color=color)
        ax.set_title(f"KM POISE — {label}")
    plt.suptitle("KM: POISE Responder vs Non-Responder", fontweight="bold"); plt.tight_layout()
    return _save(fig, os.path.join(outdir, "km_poise.png"))

# region Privacy plot

# Colore sezione privacy
_COLOR_PRIVACY = "#8B1A1A"


def plot_privacy_section(real: pd.DataFrame, synth: pd.DataFrame,
                          num_ok: list, outdir: str,
                          privacy_score=None) -> list:
    """
    Plot sezione privacy:
      1. Radar chart (stile unificato, rosso scuro)
      2. DCR row-level histogram + baseline real-real
      3. NNDR distribution con percentili
    """
    paths = []

    # dcr + nndr
    r = real[num_ok].dropna()
    s = synth[num_ok].dropna()
    n = min(len(r), len(s))
    if n < 5:
        return paths

    scaler = StandardScaler()
    Xr = scaler.fit_transform(r.sample(n, random_state=42))
    Xs = scaler.transform(s.sample(n, random_state=42))

    nn2 = NearestNeighbors(n_neighbors=2).fit(Xr)
    d_sr, _ = nn2.kneighbors(Xs)
    dcr  = d_sr[:, 0]
    nndr = d_sr[:, 0] / (d_sr[:, 1] + 1e-8)
    d_rr, _ = nn2.kneighbors(Xr)
    baseline = d_rr[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    # DCR
    ax1.hist(dcr,      bins=50, color=_COLOR_PRIVACY, alpha=0.75, edgecolor="white",
             label="Synth → Real (DCR)")
    ax1.hist(baseline, bins=50, color="#888888",       alpha=0.5,  edgecolor="white",
             label="Real → Real (baseline)")
    ax1.axvline(float(np.median(dcr)),      ls="--", color="red",    lw=1.5,
                label=f"Median DCR={np.median(dcr):.3f}")
    ax1.axvline(float(np.median(baseline)), ls="--", color="black",  lw=1.2,
                label=f"Median baseline={np.median(baseline):.3f}")
    ax1.set_xlabel("DCR (standardised distance)")
    ax1.set_title("DCR Distribution vs Real-Real Baseline", fontweight="bold")
    ax1.legend(fontsize=8)

    # NNDR con percentili
    p5  = float(np.percentile(nndr, 5))
    p50 = float(np.percentile(nndr, 50))
    ax2.hist(nndr, bins=50, color="#2A9D8F", alpha=0.8, edgecolor="white")
    ax2.axvline(0.5, ls="--", color="red",    lw=1.5, label="NNDR=0.5 (risk)")
    ax2.axvline(1.0, ls="--", color="#888888", lw=1,   label="NNDR=1.0")
    ax2.axvline(p5,  ls=":",  color="orange",  lw=1.2, label=f"p5={p5:.3f}")
    ax2.axvline(p50, ls=":",  color="green",   lw=1.2, label=f"p50={p50:.3f}")
    frac_lt05 = float(np.mean(nndr < 0.5))
    ax2.set_xlabel("NNDR (dist_1nn / dist_2nn)")
    ax2.set_title(f"NNDR Distribution  (fraction<0.5: {frac_lt05:.3f})", fontweight="bold")
    ax2.legend(fontsize=8)

    plt.suptitle("Privacy — DCR & NNDR (row-level)", fontweight="bold")
    plt.tight_layout()
    paths.append(_save(fig, os.path.join(outdir, "privacy_dcr_nndr.png")))

    
    return paths


# region plot fidelity section
def plot_fidelity_section(
    real, synth, real_raw,
    num_ok, cat_ok, temporal_vars,
    time_col, patient_col, fup_col,
    lfs, inverse_maps, outdir,
    umap_color_vars=None,
    death_col="DEATH", transp_col="TRANSP",
) -> dict:
    _safe_mkdir(outdir)
    out: dict = {}

    print("    [plots] Distribuzioni numeriche...")
    out["numeric"] = plot_numeric_grid(real, synth, num_ok, outdir)

    print("    [plots] Distribuzioni categoriche...")
    out["categorical"] = (plot_categorical_grid(real, synth, cat_ok, outdir,
                                                 inverse_maps=inverse_maps)
                           if cat_ok else [])

    print("    [plots] Matrici di correlazione...")
    corr_paths = []
    p = plot_correlation_comparison(real, synth, num_ok, "Continue", outdir)
    if p: corr_paths.append(p)
    if cat_ok:
        p = plot_correlation_comparison(real, synth, cat_ok, "Categoriche", outdir)
        if p: corr_paths.append(p)
    out["correlation"] = corr_paths

    print("    [plots] PCA...")
    p = plot_pca_shared_space(real, synth, num_ok, outdir)
    out["pca"] = [p] if p else []

    print("    [plots] UMAP...")
    out["umap"] = plot_umap(real, synth, num_ok, outdir,
                             color_vars=umap_color_vars, inverse_maps=inverse_maps,
                             temporal_vars=temporal_vars, time_col=time_col,
                             patient_col=patient_col)

    print("    [plots] Traiettorie medie + campionate...")
    traj_mean, traj_sample = [], []
    for v in temporal_vars:
        p = plot_trajectory_mean_ci(real, synth, v, time_col, patient_col, outdir)
        if p: traj_mean.append(p)
        p = plot_sampled_trajectories(real, synth, v, time_col, patient_col, outdir)
        if p: traj_sample.append(p)
    out["traj_mean"]   = traj_mean
    out["traj_sample"] = traj_sample

    print("    [plots] Varianza intra + inter paziente...")
    out["variance_intra"] = plot_variance_grid(real, synth, temporal_vars, patient_col, outdir)
    p = plot_inter_patient_variance(real, synth, temporal_vars, patient_col, time_col, outdir)
    out["variance_inter"] = [p] if p else []

    print("    [plots] Autocorrelazione lag-k...")
    out["autocorr"] = plot_autocorrelation_lagk(
        real, synth, temporal_vars, time_col, patient_col, outdir)

    print("    [plots] Variable-by-visit...")
    out["var_by_visit"] = plot_variable_by_visit(
        real, synth, temporal_vars, time_col, patient_col, outdir)

    print("    [plots] Distribuzione visite + timing...")
    out["visit_dist"] = [plot_visit_distribution(real, synth, patient_col, outdir)]
    out["visit_pos"]  = plot_visit_position_timing(real, synth, time_col, patient_col, outdir)

    print("    [plots] Last visit vs t_FUP...")
    out["fup"] = plot_last_visit_vs_fup(
        real_raw, synth, time_col, patient_col, fup_col, outdir, death_col=death_col)

    print("    [plots] Kaplan-Meier...")
    km_paths = []
    p = plot_km_event(real_raw, synth, time_col, patient_col, fup_col, death_col, outdir)
    if p: km_paths.append(p)
    p = plot_km_event(real_raw, synth, time_col, patient_col, fup_col, transp_col, outdir)
    if p: km_paths.append(p)
    out["km"] = km_paths

    return out