# eval/metrics.py
#
# Ogni score è in [0,1], 1 = best case.

import warnings
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scipy.stats import ks_2samp, pearsonr, kendalltau, wasserstein_distance, chi2_contingency, fisher_exact
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mutual_info_score
from eval.scores import (StatisticalFidelityScore, LongitudinalFidelityScore, TemporalFidelityScore, PrivacyScore,
    _fmt, _grade, _lower_to_score, _reldiff_to_score,
)

logger = logging.getLogger(__name__)


#utils
def _ks_to_score(ks: float) -> float:
    """KS in [0,1] → score [0,1] con trasformazione lineare 1 - ks."""
    return float(np.clip(1.0 - float(ks), 0.0, 1.0))


def _patient_groups(df: pd.DataFrame, patient_col: str) -> dict:
    """Cache dei gruppi paziente per evitare re-groupby ripetuti."""
    return {pid: g for pid, g in df.groupby(patient_col)}


# cramer's v
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V corretto per bias tra due Series categoriche."""
    try:
        confusion = pd.crosstab(x.fillna("NA").astype(str),
                                y.fillna("NA").astype(str))
        chi2, _, _, _ = chi2_contingency(confusion)
        n   = confusion.sum().sum()
        r, k = confusion.shape
        phi2 = max(0, chi2 / (n + 1e-9) - ((k-1)*(r-1)) / (n - 1 + 1e-9))
        rc = r - (r-1)**2 / (n - 1 + 1e-9)
        kc = k - (k-1)**2 / (n - 1 + 1e-9)
        denom = min(kc - 1, rc - 1)
        return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
    except Exception:
        return 0.0


# Fidelity Metrics

# analisi Statistica
def _similarity_metrics_num(real: pd.DataFrame, synth: pd.DataFrame,
                              num_vars: List[str]) -> dict:
    """KS e Wasserstein per variabili continue."""
    ks_s, wass_s, wass_exp_s = [], [], []

    for v in num_vars:
        r = pd.to_numeric(real[v],  errors="coerce").dropna().values
        s = pd.to_numeric(synth[v], errors="coerce").dropna().values
        if len(r) < 2 or len(s) < 2:
            continue
        try:
            ks, _ = ks_2samp(r, s)
            ks_s.append(ks)
            w = wasserstein_distance(r, s)
            wass_s.append(w)
            scale = np.std(np.concatenate([r, s])) + 1e-9
            wass_exp_s.append(np.exp(-w / scale))
        except Exception as e:
            logger.warning("similarity_metrics_num skip %s: %s", v, e)

    return {
        "KS_avg":               float(np.mean(ks_s))      if ks_s      else float("nan"),
        "Wasserstein_avg":      float(np.mean(wass_s))    if wass_s    else float("nan"),
        "Wasserstein_exp_avg":  float(np.mean(wass_exp_s)) if wass_exp_s else float("nan"),
    }


def _categorical_stat_test(real: pd.DataFrame, synth: pd.DataFrame,
                             cat_vars: List[str]) -> dict:
    """
    Per ogni variabile categorica, seleziona il test più appropriato:
      - Fisher esatto (2×2): se il numero di categorie è 2 E min(freq_attese) < 5
      - Chi²: se min(freq_attese) >= 5  (o se ci sono > 2 categorie)

    L'effect size è Cramér's V (corretto per bias).

    Restituisce:
      {
        "p_value_avg": float,   # media dei p-value (higher = more similar)
        "details": {var: {"test": str, "p_value": float, "effect_size": float, "score": float,}}
      }
    """
    p_values = []
    sim_scores = []
    details: dict = {}

    for v in cat_vars:
        if v not in real.columns or v not in synth.columns:
            continue
        try:
            r_raw = real[v].fillna("NA").astype(str)
            s_raw = synth[v].fillna("NA").astype(str)
            cats = sorted(set(r_raw.unique()) | set(s_raw.unique()))
            r_counts = r_raw.value_counts().reindex(cats, fill_value=0)
            s_counts = s_raw.value_counts().reindex(cats, fill_value=0)
            contingency = np.array([r_counts.values, s_counts.values])  # shape (2, K)

            # Frequenze attese (formula Chi²)
            row_sums = contingency.sum(axis=1, keepdims=True)
            col_sums = contingency.sum(axis=0, keepdims=True)
            total    = contingency.sum()
            expected = (row_sums * col_sums) / (total + 1e-9)
            min_expected = expected.min()

            #if len(cats) == 2 and min_expected < 5:
                # Fisher esatto (tabella 2×2)
                #_, p_val = fisher_exact(contingency, alternative="two-sided")
                #test_name = "fisher"
            #else:
                # Chi² (tabella K×2)
            chi2, p_val, _, _ = chi2_contingency(contingency)
            test_name = "chi2"

            # Cramér's V come effect size
            cv = cramers_v(r_raw, s_raw)

            # Trasformiamo l'effetto (distanza) in punteggio di similarità
            sim_score = float(np.clip(1.0 - cv, 0.0, 1.0))

            p_values.append(float(p_val))
            sim_scores.append(sim_score)
            details[v] = {
                "test":        test_name,
                "p_value":     float(p_val),
                "effect_size": float(cv),
                "CV_avg":       float(sim_score),
            }
        except Exception as e:
            logger.warning("categorical_stat_test skip %s: %s", v, e)

    return {
        "p_value_avg": float(np.mean(p_values)) if p_values else float("nan"),
        "CV_avg": float(np.mean(sim_scores)) if sim_scores else float("nan"),
        "details":     details,
    }


def _correlation_distance_num(real: pd.DataFrame, synth: pd.DataFrame,
                               num_vars: List[str]) -> float:
    if not num_vars:
        return float("nan")
    rc = real[num_vars].corr().fillna(0).values
    sc = synth[num_vars].corr().fillna(0).values
    return float(np.mean(np.abs(rc - sc)))


def _correlation_distance_cat(real: pd.DataFrame, synth: pd.DataFrame,
                               cat_vars: List[str]) -> float:
    if not cat_vars or len(cat_vars) < 2:
        return float("nan")
    def _mat(df):
        n = len(cat_vars)
        m = np.zeros((n, n))
        for i, ci in enumerate(cat_vars):
            for j, cj in enumerate(cat_vars):
                if i == j:
                    m[i, j] = 1.0
                elif j > i:
                    v = cramers_v(df[ci], df[cj])
                    m[i, j] = m[j, i] = v
        return m
    return float(np.mean(np.abs(_mat(real) - _mat(synth))))


def _pca_overlap(real: pd.DataFrame, synth: pd.DataFrame,
                  num_vars: List[str]) -> dict:
    empty = {"PCA_centroid_sim": float("nan"), "PCA_overlap": float("nan")}
    if len(num_vars) < 2:
        return empty
    r = real[num_vars].dropna(how="all").fillna(0)
    s = synth[num_vars].dropna(how="all").fillna(0)
    if len(r) < 5 or len(s) < 5:
        return empty
    scaler = StandardScaler()
    Xr = scaler.fit_transform(r)
    Xs = scaler.transform(s)
    pca = PCA(n_components=2, random_state=42)
    Zr = pca.fit_transform(Xr)
    Zs = pca.transform(Xs)
    cd = np.linalg.norm(Zr.mean(0) - Zs.mean(0))
    scale = np.sqrt(Zr.var(0).sum()) + 1e-9
    csim  = float(np.exp(-cd / scale))
    d_sr, _ = NearestNeighbors(n_neighbors=1).fit(Zr).kneighbors(Zs)
    d_rr, _ = NearestNeighbors(n_neighbors=2).fit(Zr).kneighbors(Zr)
    thr  = np.median(d_rr[:, 1])
    ovlp = float(np.mean(d_sr[:, 0] <= thr))
    return {"PCA_centroid_sim": csim, "PCA_overlap": ovlp}

# region Statistical Fidelity Score
def compute_statistical_fidelity_score(real: pd.DataFrame, synth: pd.DataFrame,
                                        num_ok: List[str],
                                        cat_ok: List[str]) -> StatisticalFidelityScore:
    """
    Calcola StatisticalFidelityScore.

    Per le variabili categoriche usa test statistico per comparare distribuzioni:
      - Fisher esatto se il test è 2×2 e min(freq_attese) < 5
      - Chi² altrimenti
    Il p-value medio viene usato come componente dello score (higher = better).
    """
    sim  = _similarity_metrics_num(real, synth, num_ok)
    pca  = _pca_overlap(real, synth, num_ok)
    cat  = _categorical_stat_test(real, synth, cat_ok)

    sfs = StatisticalFidelityScore(
        ks_raw       = sim.get("KS_avg",          float("nan")),
        cat_test_raw = cat.get("CV_avg",      float("nan")),
        corr_c_raw   = _correlation_distance_num(real, synth, num_ok),
        corr_k_raw   = _correlation_distance_cat(real, synth, cat_ok),
        pca_c_raw    = pca.get("PCA_centroid_sim", float("nan")),
        pca_o_raw    = pca.get("PCA_overlap",      float("nan")),
        #cat_test_details = cat.get("details", {}),
    )
    sfs.compute_overall()
    # Esponi anche il dizionario raw per i plot
    sfs._raw_sim = sim
    return sfs


# analisi Longitudinale
def _patient_slopes(df: pd.DataFrame, var: str,
                    time_col: str, patient_col: str) -> np.ndarray:
    slopes = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[time_col], errors="coerce").values.reshape(-1, 1)
        y = pd.to_numeric(g[var],      errors="coerce").values
        mask = ~(np.isnan(x.flatten()) | np.isnan(y))
        if mask.sum() >= 2:
            slopes.append(LinearRegression().fit(x[mask], y[mask]).coef_[0])
    return np.array(slopes)


def _within_variance(df: pd.DataFrame, var: str, patient_col: str) -> np.ndarray:
    out = []
    for _, g in df.groupby(patient_col):
        y = pd.to_numeric(g[var], errors="coerce").dropna()
        if len(y) >= 2:
            out.append(float(np.var(y, ddof=1)))
    return np.array(out)


def _autocorrelation_lagk(df: pd.DataFrame, var: str,
                           time_col: str, patient_col: str,
                           k: int = 1) -> np.ndarray:
    """Autocorrelazione lag-k per ogni paziente."""
    out = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        y = pd.to_numeric(g[var], errors="coerce").dropna().values
        if len(y) >= k + 2:
            a, b = y[:-k], y[k:]
            if np.std(a) > 0 and np.std(b) > 0:
                c = np.corrcoef(a, b)[0, 1]
                if not np.isnan(c):
                    out.append(c)
    return np.array(out)


def _dtw_distance(real: pd.DataFrame, synth: pd.DataFrame,
                  var: str, time_col: str, patient_col: str,
                  n_sample: int = 200) -> float:
    """
    DTW medio tra traiettorie reali e sintetiche per una variabile.
    Usa implementazione O(N²) per traiettorie brevi. Campiona n_sample pazienti.
    """
    def _traj(df):
        trajs = []
        for _, g in df.groupby(patient_col):
            g  = g.sort_values(time_col)
            y  = pd.to_numeric(g[var], errors="coerce").dropna().values
            if len(y) >= 2:
                trajs.append(y)
        return trajs

    def _dtw(a, b):
        n, m = len(a), len(b)
        D = np.full((n + 1, m + 1), np.inf)
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(a[i-1] - b[j-1])
                D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])
        return D[n, m] / (n + m)

    r_trajs = _traj(real)
    s_trajs = _traj(synth)
    if not r_trajs or not s_trajs:
        return float("nan")

    rng = np.random.default_rng(42)
    r_idx = rng.choice(len(r_trajs), size=min(n_sample, len(r_trajs)), replace=False)
    s_idx = rng.choice(len(s_trajs), size=min(n_sample, len(s_trajs)), replace=False)
    dists = []
    for ri, si in zip(r_idx, s_idx):
        try:
            dists.append(_dtw(r_trajs[ri], s_trajs[si]))
        except Exception:
            pass
    return float(np.mean(dists)) if dists else float("nan")


def _visit_interval_stats(df: pd.DataFrame,
                           time_col: str, patient_col: str) -> dict:
    intervals = []
    for _, g in df.groupby(patient_col):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
        if len(t) >= 2:
            intervals.extend(np.diff(t).tolist())
    if not intervals:
        return {"mean": float("nan"), "std": float("nan")}
    return {"mean": float(np.mean(intervals)), "std": float(np.std(intervals))}


def _temporal_coherence(real: pd.DataFrame, synth: pd.DataFrame,
                         temporal_vars: List[str],
                         time_col: str, patient_col: str) -> dict:
    tc_s, ktc_s = [], []
    for v in temporal_vars:
        r_ups, s_ups = [], []
        for df, lst in [(real, r_ups), (synth, s_ups)]:
            for _, g in df.groupby(patient_col):
                y = pd.to_numeric(g.sort_values(time_col)[v],
                                   errors="coerce").dropna().values
                if len(y) >= 2:
                    lst.extend((np.diff(y) > 0).tolist())
        if r_ups and s_ups:
            tc_s.append(1.0 - abs(float(np.mean(r_ups)) - float(np.mean(s_ups))))
    for i in range(len(temporal_vars)):
        for j in range(i+1, len(temporal_vars)):
            vi, vj = temporal_vars[i], temporal_vars[j]
            tr, ts = [], []
            for df, lst in [(real, tr), (synth, ts)]:
                for _, g in df.groupby(patient_col):
                    g = g.sort_values(time_col)
                    xi = pd.to_numeric(g[vi], errors="coerce").dropna()
                    xj = pd.to_numeric(g[vj], errors="coerce").dropna()
                    n  = min(len(xi), len(xj))
                    if n >= 3:
                        tau, _ = kendalltau(xi.values[:n], xj.values[:n])
                        if not np.isnan(tau):
                            lst.append(tau)
            if len(tr) > 5 and len(ts) > 5:
                ks, _ = ks_2samp(tr, ts)
                ktc_s.append(1.0 - ks)
    tc  = float(np.mean(tc_s))  if tc_s  else float("nan")
    ktc = float(np.mean(ktc_s)) if ktc_s else float("nan")
    ws, tw = 0.0, 0.0
    for val, w in [(tc, 1.0), (ktc, 1.5)]:
        if not np.isnan(val):
            ws += val * w; tw += w
    overall = float(ws / tw) if tw > 0 else float("nan")
    return {
        "TCS_transition": _fmt(tc),
        "TCS_kendall":    _fmt(ktc),
        "TCS_overall":    _fmt(overall),
    }


def _positional_timing_ks(real: pd.DataFrame, synth: pd.DataFrame,
                           time_col: str, patient_col: str) -> tuple:
    """Avg KS del timing per posizione sequenziale di visita."""
    n_pos = max(2, int(np.median(synth.groupby(patient_col).size().values)))

    def extract(df):
        pt: dict = {p: [] for p in range(1, n_pos+1)}
        for _, g in df.groupby(patient_col):
            ts = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            for idx, t in enumerate(ts[:n_pos]):
                pt[idx+1].append(float(t))
        return {k: np.array(v) for k, v in pt.items() if len(v) >= 3}

    rp, sp = extract(real), extract(synth)
    ks_list, details = [], {}
    for pos in sorted(set(rp) & set(sp)):
        if len(rp[pos]) >= 2 and len(sp[pos]) >= 2:
            ks, _ = ks_2samp(rp[pos], sp[pos])
            ks_list.append(ks)
            details[f"pos_{pos:02d}_KS"]          = float(ks)
            details[f"pos_{pos:02d}_median_real"]  = float(np.median(rp[pos]))
            details[f"pos_{pos:02d}_median_synth"] = float(np.median(sp[pos]))
    avg_ks = float(np.mean(ks_list)) if ks_list else float("nan")
    return avg_ks, details

# region Longitudinal Fidelity Score
def compute_longitudinal_fidelity_score(
    real:          pd.DataFrame,
    synth:         pd.DataFrame,
    temporal_vars: List[str],
    time_col:      str,
    patient_col:   str,
) -> LongitudinalFidelityScore:
    """
    Calcola LongitudinalFidelityScore.

    Componenti dello score: slope, variance, autocorr lag-1, TCS, DTW.
    Componenti informative (non pesate): timing posizionale, inter-visit interval,
    visit count, autocorr lag-2..5, LME betas.
    """
    lfs = LongitudinalFidelityScore()

    ks_slopes, ks_vars, ks_ac1 = [], [], []
    dtw_vals = {}

    for v in temporal_vars:
        sr = _patient_slopes(real,  v, time_col, patient_col)
        ss = _patient_slopes(synth, v, time_col, patient_col)
        if len(sr) > 5 and len(ss) > 5:
            ks_slopes.append(ks_2samp(sr, ss)[0])

        vr = _within_variance(real,  v, patient_col)
        vs = _within_variance(synth, v, patient_col)
        if len(vr) > 5 and len(vs) > 5:
            ks_vars.append(ks_2samp(vr, vs)[0])

        ar  = _autocorrelation_lagk(real,  v, time_col, patient_col, k=1)
        as_ = _autocorrelation_lagk(synth, v, time_col, patient_col, k=1)
        if len(ar) > 5 and len(as_) > 5:
            ks_ac1.append(ks_2samp(ar, as_)[0])

        dtw_v = _dtw_distance(real, synth, v, time_col, patient_col)
        if not np.isnan(dtw_v):
            dtw_vals[v] = dtw_v

    lfs.slope_raw = float(np.mean(ks_slopes)) if ks_slopes else float("nan")
    lfs.var_raw   = float(np.mean(ks_vars))   if ks_vars   else float("nan")
    lfs.ac_raw    = float(np.mean(ks_ac1))    if ks_ac1    else float("nan")

    if dtw_vals:
        raw_dtw = float(np.mean(list(dtw_vals.values())))
        lfs.dtw_raw = float(np.clip(raw_dtw / 10.0, 0.0, 1.0))

    # autocorr lag-k (k=1..5) informativi
    ac_lagk: dict = {}
    for k in range(1, 6):
        ks_k = []
        for v in temporal_vars:
            ar  = _autocorrelation_lagk(real,  v, time_col, patient_col, k=k)
            as_ = _autocorrelation_lagk(synth, v, time_col, patient_col, k=k)
            if len(ar) > 5 and len(as_) > 5:
                ks_k.append(ks_2samp(ar, as_)[0])
        ac_lagk[k] = float(np.mean(ks_k)) if ks_k else float("nan")
    lfs.ac_lagk = ac_lagk

    # TCS
    tcs = _temporal_coherence(real, synth, temporal_vars, time_col, patient_col)
    tcs_overall = tcs.get("TCS_overall", float("nan"))
    lfs.coh_raw = float(tcs_overall) if not isinstance(tcs_overall, str) else float("nan")

    # Informativi: timing, visit count, inter-visit interval
    avg_timing, timing_details = _positional_timing_ks(real, synth, time_col, patient_col)
    lfs.timing_raw = avg_timing

    vc_r = real.groupby(patient_col).size().values
    vc_s = synth.groupby(patient_col).size().values
    if len(vc_r) > 1 and len(vc_s) > 1:
        lfs.vc_raw = float(ks_2samp(vc_r, vc_s)[0])

    iv_r = _visit_interval_stats(real,  time_col, patient_col)
    iv_s = _visit_interval_stats(synth, time_col, patient_col)
    lfs.iv_real  = iv_r["mean"]
    lfs.iv_synth = iv_s["mean"]
    if not (np.isnan(iv_r["mean"]) or np.isnan(iv_s["mean"])) and iv_r["mean"] > 0:
        lfs.iv_raw = abs(iv_r["mean"] - iv_s["mean"]) / iv_r["mean"]

    lfs.compute_overall()
    lfs._timing_details = timing_details
    lfs._tcs = tcs
    return lfs


# region Temporal Fidelity Score
def compute_temporal_fidelity_score(
    real:          pd.DataFrame,
    synth:         pd.DataFrame,
    real_raw:      pd.DataFrame,
    temporal_vars: List[str],
    time_col:      str,
    patient_col:   str,
    fup_col:       Optional[str] = None,
) -> TemporalFidelityScore:
    """
    Calcola TemporalFidelityScore.

    Componenti: visit_count_score, timing_score, interval_score, fup_coverage.
    Slope, variance, autocorr e DTW sono demandati a LFS.
    """
    tfs = TemporalFidelityScore()

    # Visit count
    vc_r = real.groupby(patient_col).size().values
    vc_s = synth.groupby(patient_col).size().values
    if len(vc_r) > 1 and len(vc_s) > 1:
        tfs.visit_count_score = _ks_to_score(ks_2samp(vc_r, vc_s)[0])

    # Timing posizionale
    avg_timing, _ = _positional_timing_ks(real, synth, time_col, patient_col)
    tfs.timing_score = _ks_to_score(avg_timing) if not np.isnan(avg_timing) else float("nan")

    # Inter-visit interval
    iv_r = _visit_interval_stats(real,  time_col, patient_col)
    iv_s = _visit_interval_stats(synth, time_col, patient_col)
    if not (np.isnan(iv_r["mean"]) or np.isnan(iv_s["mean"])) and iv_r["mean"] > 0:
        reldiff = abs(iv_r["mean"] - iv_s["mean"]) / iv_r["mean"]
        tfs.interval_score = _reldiff_to_score(reldiff)

    # FUP coverage
    if fup_col and fup_col in synth.columns and fup_col in real.columns:
        def _cov(df):
            last_t = df.groupby(patient_col)[time_col].apply(
                lambda s: pd.to_numeric(s, errors="coerce").max()).rename("lv")
            fup    = df.groupby(patient_col)[fup_col].first().rename("fup")
            m = pd.concat([last_t, fup], axis=1).dropna()
            m["fup"] = pd.to_numeric(m["fup"], errors="coerce")
            m = m[m["fup"] > 0]
            if m.empty:
                return float("nan")
            return float((m["lv"] / m["fup"]).clip(0, 1).median())
        sc = _cov(synth)
        rc = _cov(real)
        if not (np.isnan(sc) or np.isnan(rc)):
            base    = 1.0 - abs(sc - rc)
            penalty = max(0.0, (0.9 - sc) * 2.0) if sc < 0.9 else 0.0
            tfs.fup_coverage = float(np.clip(base - penalty, 0.0, 1.0))

    tfs.compute_overall()
    return tfs

# region Fidelity metrics
def compute_fidelity_metrics(
    real:          pd.DataFrame,
    synth:         pd.DataFrame,
    real_raw:      pd.DataFrame,
    num_ok:        List[str],
    cat_ok:        List[str],
    temporal_vars: List[str],
    time_col:      str,
    patient_col:   str,
    fup_col:       Optional[str] = None,
) -> dict:
    """
    Punto di ingresso unificato per le metriche di fidelity.
    Restituisce:
      {
        "sfs": StatisticalFidelityScore,
        "lfs": LongitudinalFidelityScore,
        "tfs": TemporalFidelityScore,
      }
    """
    print("  [fidelity] Statistical ...")
    sfs = compute_statistical_fidelity_score(real, synth, num_ok, cat_ok)

    print("  [fidelity] Longitudinal (DTW può richiedere tempo) ...")
    lfs = compute_longitudinal_fidelity_score(
        real, synth, temporal_vars, time_col, patient_col)

    print("  [fidelity] Temporal ...")
    tfs = compute_temporal_fidelity_score(
        real, synth, real_raw, temporal_vars, time_col, patient_col, fup_col)

    return {"sfs": sfs, "lfs": lfs, "tfs": tfs}


#utils
def _safe_mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float)) and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _prep_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    d = df[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d




# PRIVACY METRICS 

def _trajectory_embeddings(
    df: pd.DataFrame,
    patient_col: str,
    time_col: str,
    numeric_features: List[str],
):

    rows = []

    d = df.sort_values([
        patient_col,
        time_col,
    ])

    for pid, g in d.groupby(patient_col):

        vec = []

        for col in numeric_features:

            vals = pd.to_numeric(
                g[col],
                errors="coerce",
            ).dropna()

            if len(vals) == 0:

                vec.extend([
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ])

                continue

            x = np.arange(len(vals))

            try:
                slope = np.polyfit(x, vals, 1)[0]
            except Exception:
                slope = np.nan

            vec.extend([
                float(vals.mean()),
                float(vals.std()),
                float(vals.max()),
                float(slope),
            ])

        rows.append([pid] + vec)

    cols = ["patient_id"]

    for c in numeric_features:
        cols.extend([
            f"{c}_mean",
            f"{c}_std",
            f"{c}_max",
            f"{c}_slope",
        ])

    return pd.DataFrame(rows, columns=cols)


def _dcr_scores(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_features: List[str],
):

    empty = {
        "score": float("nan"),
        "mean": float("nan"),
        "median": float("nan"),
        "dists": None,
    }

    try:

        Xr = _prep_numeric(real, numeric_features)
        Xs = _prep_numeric(synth, numeric_features)

        imp = SimpleImputer(strategy="median")

        Xr = imp.fit_transform(Xr)
        Xs = imp.transform(Xs)

        scaler = StandardScaler()

        Xr = scaler.fit_transform(Xr)
        Xs = scaler.transform(Xs)

        nn = NearestNeighbors(n_neighbors=1)

        nn.fit(Xr)

        d_sr, _ = nn.kneighbors(Xs)

        d_sr = d_sr[:, 0]

        nn_rr = NearestNeighbors(n_neighbors=2)
        nn_rr.fit(Xr)

        d_rr, _ = nn_rr.kneighbors(Xr)

        spacing = float(
            np.median(d_rr[:, 1])
        ) + 1e-9

        ratio = float(
            np.median(d_sr) / spacing
        )

        score = float(
            1.0 - np.exp(-ratio)
        )

        return {
            "score": score,
            "mean": float(np.mean(d_sr)),
            "median": float(np.median(d_sr)),
            "dists": d_sr,
        }

    except Exception as e:
        logger.warning("dcr_scores error: %s", e)
        return empty


def _nndr_score(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_features: List[str],
):

    empty = {
        "score": float("nan"),
        "frac_lt05": float("nan"),
        "nndr": None,
    }

    try:

        Xr = _prep_numeric(real, numeric_features)
        Xs = _prep_numeric(synth, numeric_features)

        imp = SimpleImputer(strategy="median")

        Xr = imp.fit_transform(Xr)
        Xs = imp.transform(Xs)

        scaler = StandardScaler()

        Xr = scaler.fit_transform(Xr)
        Xs = scaler.transform(Xs)

        nn = NearestNeighbors(n_neighbors=2)

        nn.fit(Xr)

        d, _ = nn.kneighbors(Xs)

        nndr = d[:, 0] / (d[:, 1] + 1e-9)

        frac = float(
            np.mean(nndr < 0.5)
        )

        score = float(1.0 - frac)

        return {
            "score": score,
            "frac_lt05": frac,
            "nndr": nndr,
        }

    except Exception as e:
        logger.warning("nndr_score error: %s", e)
        return empty


# region Privacy Score

def compute_privacy_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    numeric_features: List[str],
    patient_col: Optional[str] = None,
    time_col: Optional[str] = None,
):

    ps = PrivacyScore()

    #row DCR
    dcr = _dcr_scores(
        real,
        synth,
        numeric_features,
    )

    ps.dcr_score = dcr["score"]
    ps.dcr_mean = dcr["mean"]
    ps.dcr_median = dcr["median"]

    # trajectory DCR 
    if patient_col and time_col:

        er = _trajectory_embeddings(
            real,
            patient_col,
            time_col,
            numeric_features,
        )

        es = _trajectory_embeddings(
            synth,
            patient_col,
            time_col,
            numeric_features,
        )

        traj_feats = [
            c for c in er.columns
            if c != "patient_id"
        ]

        tdcr = _dcr_scores(
            er,
            es,
            traj_feats,
        )

        ps.dcr_trajectory_score = tdcr["score"]

    # NNDR 
    nndr = _nndr_score(
        real,
        synth,
        numeric_features,
    )

    ps.nndr_score = nndr["score"]
    ps.nndr_frac_lt05 = nndr["frac_lt05"]

    
    ps.compute_overall()

    return ps