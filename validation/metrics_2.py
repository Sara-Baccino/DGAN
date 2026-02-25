import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from fpdf import FPDF
import os


# ======================================================
# STATISTICHE BASE
# ======================================================

def mean_ci(series, alpha=0.05):
    series = series.dropna()
    n = len(series)
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = series.mean()
    se = stats.sem(series)
    h = se * stats.t.ppf(1 - alpha / 2, n - 1)
    return mean, mean - h, mean + h


def describe_missing_and_visits(df, id_col):
    missing = df.isna().mean().sort_values(ascending=False)
    visits = df.groupby(id_col).size()
    return missing, visits.describe()


# ======================================================
# MARGINALI
# ======================================================

def marginal_numeric(real, synth, variables):
    rows = []
    for v in variables:
        r_mean, r_l, r_u = mean_ci(real[v])
        s_mean, s_l, s_u = mean_ci(synth[v])
        ks = stats.ks_2samp(
            real[v].dropna(),
            synth[v].dropna()
        ).statistic
        rows.append([v, r_mean, r_l, r_u, s_mean, s_l, s_u, ks])

    return pd.DataFrame(rows, columns=[
        "var", "real_mean", "real_l", "real_u",
        "synth_mean", "synth_l", "synth_u", "KS"
    ])


def marginal_categorical(real, synth, variables):
    rows = []
    for v in variables:
        r = real[v].value_counts(normalize=True)
        s = synth[v].value_counts(normalize=True)

        idx = r.index.union(s.index)
        r = r.reindex(idx, fill_value=0.0)
        s = s.reindex(idx, fill_value=0.0)

        if r.sum() == 0 or s.sum() == 0:
            js = np.nan
        else:
            js = jensenshannon(r.values, s.values)

        rows.append([v, js])

    return pd.DataFrame(rows, columns=["var", "JS_divergence"])


# ======================================================
# LONGITUDINALE
# ======================================================

def longitudinal_numeric(real, synth, visit_col, variables):
    out = {}
    for v in variables:
        r = real.groupby(visit_col)[v].apply(mean_ci)
        s = synth.groupby(visit_col)[v].apply(mean_ci)
        out[v] = {"real": r, "synth": s}
    return out


def transition_matrix(df, id_col, visit_col, var):
    df = df[[id_col, visit_col, var]].dropna()
    df = df.sort_values([id_col, visit_col])

    transitions = {}
    for _, g in df.groupby(id_col):
        vals = g[var].values
        for i in range(len(vals) - 1):
            transitions[(vals[i], vals[i + 1])] = \
                transitions.get((vals[i], vals[i + 1]), 0) + 1

    mat = pd.Series(transitions).unstack(fill_value=0)
    return mat / mat.values.sum()


# ======================================================
# RELAZIONI & PRIVACY
# ======================================================

def correlation_distance(real, synth, num_vars):
    r_corr = real[num_vars].corr()
    s_corr = synth[num_vars].corr()
    return (r_corr - s_corr).abs().mean().mean()


def nn_privacy(real, synth, num_vars):
    r = real[num_vars].dropna()
    s = synth[num_vars].dropna()
    n = min(len(r), len(s))
    r, s = r.iloc[:n], s.iloc[:n]

    nn = NearestNeighbors(n_neighbors=1).fit(r)
    dist, _ = nn.kneighbors(s)
    return np.mean(dist)


# ======================================================
# PLOT
# ======================================================

def plot_numeric_distribution(real, synth, var, outdir):
    plt.figure(figsize=(5, 4))
    plt.hist(real[var].dropna(), bins=30, density=True, alpha=0.5, label="Real")
    plt.hist(synth[var].dropna(), bins=30, density=True, alpha=0.5, label="Synthetic")
    plt.title(var)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"{var}_num.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_categorical_distribution(real, synth, var, outdir):
    r = real[var].value_counts(normalize=True)
    s = synth[var].value_counts(normalize=True)

    idx = r.index.union(s.index)
    r = r.reindex(idx, fill_value=0)
    s = s.reindex(idx, fill_value=0)

    plt.figure(figsize=(5, 4))
    x = np.arange(len(idx))

    plt.bar(x, r.values, width=0.4, label="Real")
    plt.bar(x + 0.4, s.values, width=0.4, label="Synthetic")

    plt.xticks(x + 0.2, idx, rotation=45)
    plt.title(var)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"{var}_cat.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_longitudinal_mean(real, synth, visit_col, var, outdir):
    r = real.groupby(visit_col)[var].mean()
    s = synth.groupby(visit_col)[var].mean()

    plt.figure(figsize=(5, 4))
    plt.plot(r.index, r.values, marker="o", label="Real")
    plt.plot(s.index, s.values, marker="o", label="Synthetic")
    plt.title(f"Mean trajectory – {var}")
    plt.xlabel("Visit")
    plt.ylabel(var)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(outdir, f"{var}_traj.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

import math
import matplotlib.pyplot as plt
import os


def plot_numeric_grid(real, synth, variables, outdir,
                      ncols=3, max_rows=5, fname="numeric_grid.png"):

    per_page = ncols * max_rows
    paths = []

    for i in range(0, len(variables), per_page):
        vars_chunk = variables[i:i + per_page]
        n = len(vars_chunk)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows)
        )
        axes = axes.flatten()

        for ax, v in zip(axes, vars_chunk):
            ax.hist(real[v].dropna(), bins=30, density=True, alpha=0.5, label="Real")
            ax.hist(synth[v].dropna(), bins=30, density=True, alpha=0.5, label="Synth")
            ax.set_title(v, fontsize=10)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.tight_layout()
        path = os.path.join(outdir, f"{fname}_{i}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths

def plot_categorical_grid(real, synth, variables, outdir,
                          ncols=3, max_rows=5, fname="cat_grid.png"):

    per_page = ncols * max_rows
    paths = []

    for i in range(0, len(variables), per_page):
        vars_chunk = variables[i:i + per_page]
        n = len(vars_chunk)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5 * ncols, 3 * nrows)
        )
        axes = axes.flatten()

        for ax, v in zip(axes, vars_chunk):
            r = real[v].value_counts(normalize=True)
            s = synth[v].value_counts(normalize=True)
            idx = r.index.union(s.index)

            r = r.reindex(idx, fill_value=0)
            s = s.reindex(idx, fill_value=0)

            x = range(len(idx))
            ax.bar(x, r.values, width=0.4, label="Real")
            ax.bar([i + 0.4 for i in x], s.values, width=0.4, label="Synth")
            ax.set_xticks([i + 0.2 for i in x])
            ax.set_xticklabels(idx, rotation=45, fontsize=8)
            ax.set_title(v, fontsize=10)

        for ax in axes[n:]:
            ax.axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")

        plt.tight_layout()
        path = os.path.join(outdir, f"{fname}_{i}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path)

    return paths

import numpy as np
import seaborn as sns


def plot_correlation_matrices(real, synth, num_vars, outdir):

    r_corr = real[num_vars].corr()
    s_corr = synth[num_vars].corr()
    diff = r_corr - s_corr

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.heatmap(r_corr, ax=axes[0], cmap="coolwarm", center=0)
    axes[0].set_title("Reale")

    sns.heatmap(s_corr, ax=axes[1], cmap="coolwarm", center=0)
    axes[1].set_title("Sintetico")

    sns.heatmap(diff, ax=axes[2], cmap="coolwarm", center=0)
    axes[2].set_title("Differenza (Real - Synth)")

    plt.tight_layout()
    path = os.path.join(outdir, "correlation_matrices.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def plot_pca_2d(real, synth, num_vars, outdir):

    r = real[num_vars].dropna()
    s = synth[num_vars].dropna()

    n = min(len(r), len(s))
    r = r.iloc[:n]
    s = s.iloc[:n]

    X = np.vstack([r.values, s.values])
    y = np.array([0]*n + [1]*n)

    X = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(Z[y == 0, 0], Z[y == 0, 1], alpha=0.4, label="Real")
    plt.scatter(Z[y == 1, 0], Z[y == 1, 1], alpha=0.4, label="Synth")
    plt.legend()
    plt.title("PCA 2D – Real vs Synthetic")
    plt.tight_layout()

    path = os.path.join(outdir, "pca_2d.png")
    plt.savefig(path, dpi=150)
    plt.close()

    return path


# ======================================================
# PDF
# ======================================================

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Valutazione Dataset Sintetico Longitudinale", ln=True)
        self.ln(3)

    def section(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def textline(self, txt):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, txt)
