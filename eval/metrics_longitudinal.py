# ======================================================
# eval/metrics_longitudinal.py
# Longitudinal & temporal metrics.
#
# [NUOVO] compute_temporal_fidelity_score():
#   Score riassuntivo [0-1] che aggrega tutti i KS longitudinali
#   in un singolo indice interpretabile clinicamente.
# ======================================================

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass, field
from typing import Dict, Optional


# ======================================================
# PER-PATIENT TRAJECTORY STATISTICS
# ======================================================

def compute_patient_slopes(
    df: pd.DataFrame, var: str, time_col: str, patient_col: str
) -> np.ndarray:
    slopes = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        x = pd.to_numeric(g[time_col], errors="coerce").values.reshape(-1, 1)
        y = pd.to_numeric(g[var],      errors="coerce").values
        mask = ~(np.isnan(x.flatten()) | np.isnan(y))
        if mask.sum() >= 2:
            model = LinearRegression().fit(x[mask], y[mask])
            slopes.append(model.coef_[0])
    return np.array(slopes)


def compute_within_variance(
    df: pd.DataFrame, var: str, patient_col: str
) -> np.ndarray:
    variances = []
    for _, g in df.groupby(patient_col):
        y = pd.to_numeric(g[var], errors="coerce").dropna()
        if len(y) >= 2:
            variances.append(float(np.var(y)))
    return np.array(variances)


def compute_autocorrelation(
    df: pd.DataFrame, var: str, time_col: str, patient_col: str
) -> np.ndarray:
    ac_vals = []
    for _, g in df.groupby(patient_col):
        g = g.sort_values(time_col)
        y = pd.to_numeric(g[var], errors="coerce").dropna().values
        if len(y) >= 3:
            if np.std(y[:-1]) == 0 or np.std(y[1:]) == 0:
                continue
            corr = np.corrcoef(y[:-1], y[1:])[0, 1]
            if not np.isnan(corr):
                ac_vals.append(corr)
    return np.array(ac_vals)


# ======================================================
# VISIT STRUCTURE METRICS
# ======================================================

def compute_visit_counts(df: pd.DataFrame, patient_col: str) -> pd.Series:
    return df.groupby(patient_col).size()


def compute_visit_interval_stats(
    df: pd.DataFrame, time_col: str, patient_col: str
) -> dict:
    intervals = []
    for _, g in df.groupby(patient_col):
        t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
        if len(t) >= 2:
            intervals.extend(np.diff(t).tolist())
    if not intervals:
        return {"Mean inter-visit interval": "N/A", "Std inter-visit interval": "N/A"}
    return {
        "Mean inter-visit interval": float(np.mean(intervals)),
        "Std inter-visit interval":  float(np.std(intervals)),
    }


# ======================================================
# PER-VISIT-POSITION TIMING METRICS
# ======================================================

def compute_visit_position_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
    n_positions: Optional[int] = None,
) -> dict:
    synth_counts = synth.groupby(patient_col).size()
    if n_positions is None:
        n_positions = int(np.median(synth_counts.values))
    n_positions = max(2, min(n_positions, int(synth_counts.max())))

    def extract_by_position(df, max_pos):
        pos_times: Dict[int, list] = {p: [] for p in range(1, max_pos + 1)}
        for _, g in df.groupby(patient_col):
            times = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            for idx, t in enumerate(times):
                pos = idx + 1
                if pos > max_pos:
                    break
                pos_times[pos].append(float(t))
        return {k: np.array(v) for k, v in pos_times.items() if len(v) >= 3}

    real_pos  = extract_by_position(real,  n_positions)
    synth_pos = extract_by_position(synth, n_positions)

    metrics: dict = {}
    ks_vals, mae_medians = [], []
    positions_evaluated = sorted(set(real_pos) & set(synth_pos))

    for pos in positions_evaluated:
        r_arr = real_pos[pos]
        s_arr = synth_pos[pos]
        if len(r_arr) < 2 or len(s_arr) < 2:
            continue
        ks, _ = ks_2samp(r_arr, s_arr)
        mae   = abs(float(np.median(r_arr)) - float(np.median(s_arr)))
        ks_vals.append(ks)
        mae_medians.append(mae)

        if pos <= 3 or pos % 5 == 0:
            metrics[f"Visit pos {pos:02d}: KS (real vs synth timing)"] = float(ks)
            metrics[f"Visit pos {pos:02d}: median real (mo)"]           = float(np.median(r_arr))
            metrics[f"Visit pos {pos:02d}: median synth (mo)"]          = float(np.median(s_arr))
            metrics[f"Visit pos {pos:02d}: MAE medians (mo)"]           = float(mae)

    if ks_vals:
        metrics["Avg KS across visit positions (timing)"]         = float(np.mean(ks_vals))
        metrics["Max KS across visit positions (timing)"]         = float(np.max(ks_vals))
        metrics["Avg MAE of median visit time per position (mo)"] = float(np.mean(mae_medians))
        metrics["N positions evaluated"]                           = len(ks_vals)

    return metrics


# ======================================================
# TEMPORAL CROSS CORRELATION
# ======================================================

def compute_temporal_cross_correlation(
    df: pd.DataFrame, var1: str, var2: str, time_col: str, patient_col: str,
    n_bins: int = 10
) -> Optional[pd.DataFrame]:
    df = df.copy()
    df["_t"]  = pd.to_numeric(df[time_col], errors="coerce")
    df["_v1"] = pd.to_numeric(df[var1],     errors="coerce")
    df["_v2"] = pd.to_numeric(df[var2],     errors="coerce")
    df = df.dropna(subset=["_t", "_v1", "_v2"])
    if len(df) < 10:
        return None
    bins = pd.cut(df["_t"], bins=n_bins)
    rows = []
    for b, grp in df.groupby(bins, observed=True):
        if len(grp) >= 5:
            r, _ = pearsonr(grp["_v1"], grp["_v2"])
            rows.append({"time_bin": b.mid, "correlation": r})
    return pd.DataFrame(rows) if rows else None


# ======================================================
# AGGREGATE LONGITUDINAL METRICS
# ======================================================

def calculate_longitudinal_metrics(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    temporal_vars: list,
    time_col: str,
    patient_col: str,
) -> dict:
    metrics = {}
    all_ks_slopes, all_ks_var, all_ks_ac = [], [], []

    for v in temporal_vars:
        slopes_r = compute_patient_slopes(real,  v, time_col, patient_col)
        slopes_s = compute_patient_slopes(synth, v, time_col, patient_col)
        if len(slopes_r) > 5 and len(slopes_s) > 5:
            ks, _ = ks_2samp(slopes_r, slopes_s)
            all_ks_slopes.append(ks)

        var_r = compute_within_variance(real,  v, patient_col)
        var_s = compute_within_variance(synth, v, patient_col)
        if len(var_r) > 5 and len(var_s) > 5:
            ks, _ = ks_2samp(var_r, var_s)
            all_ks_var.append(ks)

        ac_r = compute_autocorrelation(real,  v, time_col, patient_col)
        ac_s = compute_autocorrelation(synth, v, time_col, patient_col)
        if len(ac_r) > 5 and len(ac_s) > 5:
            ks, _ = ks_2samp(ac_r, ac_s)
            all_ks_ac.append(ks)

    metrics["Avg KS - Patient Slopes ((lower) better)"] = (
        float(np.mean(all_ks_slopes)) if all_ks_slopes else "N/A"
    )
    metrics["Avg KS - Within-patient Variance ((lower) better)"] = (
        float(np.mean(all_ks_var)) if all_ks_var else "N/A"
    )
    metrics["Avg KS - Autocorrelation ((lower) better)"] = (
        float(np.mean(all_ks_ac)) if all_ks_ac else "N/A"
    )

    vc_r = compute_visit_counts(real,  patient_col).values
    vc_s = compute_visit_counts(synth, patient_col).values
    if len(vc_r) > 1 and len(vc_s) > 1:
        ks, _ = ks_2samp(vc_r, vc_s)
        metrics["KS - Visit Count Distribution ((lower) better)"] = float(ks)

    iv_r = compute_visit_interval_stats(real,  time_col, patient_col)
    iv_s = compute_visit_interval_stats(synth, time_col, patient_col)
    for k in iv_r:
        vr, vs = iv_r[k], iv_s[k]
        if isinstance(vr, float) and isinstance(vs, float):
            metrics[f"{k} - Real"]      = vr
            metrics[f"{k} - Synthetic"] = vs

    return metrics


# ======================================================
# [NUOVO] TEMPORAL FIDELITY SCORE — score riassuntivo [0-1]
# ======================================================

@dataclass
class TemporalFidelityScore:
    """
    Score riassuntivo per la fedeltà temporale dei dati sintetici.
    Ogni componente è mappato in [0,1] (1 = perfetto) e pesato.

    Componenti e pesi:
      slope_score     (1.5): fedeltà traiettorie per paziente
      variance_score  (1.5): fedeltà variabilità intra-paziente  [critico PBC]
      autocorr_score  (1.0): struttura di autocorrelazione
      visit_count_score(1.0): distribuzione numero visite
      timing_score    (2.0): timing per posizione sequenziale    [critico PBC]
      interval_score  (1.0): intervalli inter-visita
      fup_coverage    (2.0): allineamento last-visit / t_FUP     [critico PBC]

    Soglie per overall:
      >= 0.85 → Eccellente
      >= 0.70 → Buona
      >= 0.55 → Accettabile
      <  0.55 → Scadente
    """
    slope_score:       float = float("nan")
    variance_score:    float = float("nan")
    autocorr_score:    float = float("nan")
    visit_count_score: float = float("nan")
    timing_score:      float = float("nan")
    interval_score:    float = float("nan")
    fup_coverage:      float = float("nan")
    overall:           float = float("nan")
    grade:             str   = "N/A"

    _weights: Dict[str, float] = field(default_factory=lambda: {
        "slope_score":        1.5,
        "variance_score":     1.5,
        "autocorr_score":     1.0,
        "visit_count_score":  1.0,
        "timing_score":       2.0,
        "interval_score":     1.0,
        "fup_coverage":       2.0,
    })

    def to_dict(self) -> Dict:
        """Dizionario pronto per add_metrics_table() del report PDF."""
        d = {
            "TFS - Slope score [0-1]":           _fmt(self.slope_score),
            "TFS - Within-patient variance [0-1]": _fmt(self.variance_score),
            "TFS - Autocorrelation [0-1]":        _fmt(self.autocorr_score),
            "TFS - Visit-count score [0-1]":      _fmt(self.visit_count_score),
            "TFS - Timing (positional) [0-1]":    _fmt(self.timing_score),
            "TFS - Inter-visit interval [0-1]":   _fmt(self.interval_score),
            "TFS - t_FUP coverage [0-1]":         _fmt(self.fup_coverage),
        }
        # Lo score globale è in fondo, evidenziato con ***
        d["*** Temporal Fidelity Score (TFS) [0-1]"] = _fmt(self.overall)
        d["*** TFS Grade"] = self.grade
        return d

    def radar_values(self) -> Dict[str, float]:
        """Valori [0-1] puliti per un radar/spider chart."""
        return {k: (v if not np.isnan(v) else 0.0) for k, v in {
            "Slopes":        self.slope_score,
            "Variance":      self.variance_score,
            "Autocorr":      self.autocorr_score,
            "Visit counts":  self.visit_count_score,
            "Timing":        self.timing_score,
            "Intervals":     self.interval_score,
            "FUP coverage":  self.fup_coverage,
        }.items()}


def _fmt(v: float):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return round(float(v), 4)


def _ks_to_score(ks: float) -> float:
    """
    Converte KS distance [0,1] in score [0,1].
    Curva esponenziale: KS=0 → 1.0, KS=0.3 → 0.41, KS=1 → 0.05.
    """
    return float(np.exp(-3.0 * np.clip(ks, 0.0, 1.0)))


def compute_temporal_fidelity_score(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    temporal_vars: list,
    time_col: str,
    patient_col: str,
    fup_col: Optional[str] = None,
) -> TemporalFidelityScore:
    """
    Calcola il Temporal Fidelity Score (TFS) [0-1] per i dati sintetici.

    Parameters
    ----------
    real, synth   : DataFrame long (una riga per visita per paziente)
    temporal_vars : variabili continue temporali
    time_col      : colonna tempo (es. "MONTHS_FROM_BASELINE")
    patient_col   : colonna ID paziente (es. "RECORD_ID")
    fup_col       : colonna follow-up totale (es. "t_FUP").
                    Se None o non presente, fup_coverage = NaN.

    Returns TemporalFidelityScore con overall e grade.
    """
    tfs = TemporalFidelityScore()

    # ── 1. Slope score ────────────────────────────────────────────────
    ks_s = []
    for v in temporal_vars:
        r = compute_patient_slopes(real,  v, time_col, patient_col)
        s = compute_patient_slopes(synth, v, time_col, patient_col)
        if len(r) > 5 and len(s) > 5:
            ks_s.append(ks_2samp(r, s)[0])
    if ks_s:
        tfs.slope_score = _ks_to_score(float(np.mean(ks_s)))

    # ── 2. Variance score ─────────────────────────────────────────────
    ks_v = []
    for v in temporal_vars:
        r = compute_within_variance(real,  v, patient_col)
        s = compute_within_variance(synth, v, patient_col)
        if len(r) > 5 and len(s) > 5:
            ks_v.append(ks_2samp(r, s)[0])
    if ks_v:
        tfs.variance_score = _ks_to_score(float(np.mean(ks_v)))

    # ── 3. Autocorrelation score ──────────────────────────────────────
    ks_a = []
    for v in temporal_vars:
        r = compute_autocorrelation(real,  v, time_col, patient_col)
        s = compute_autocorrelation(synth, v, time_col, patient_col)
        if len(r) > 5 and len(s) > 5:
            ks_a.append(ks_2samp(r, s)[0])
    if ks_a:
        tfs.autocorr_score = _ks_to_score(float(np.mean(ks_a)))

    # ── 4. Visit count score ──────────────────────────────────────────
    vc_r = compute_visit_counts(real,  patient_col).values
    vc_s = compute_visit_counts(synth, patient_col).values
    if len(vc_r) > 1 and len(vc_s) > 1:
        tfs.visit_count_score = _ks_to_score(ks_2samp(vc_r, vc_s)[0])

    # ── 5. Timing score (posizionale) ─────────────────────────────────
    n_pos = max(2, int(np.median(synth.groupby(patient_col).size().values)))

    def _extract_pos(df, max_pos):
        pt = {p: [] for p in range(1, max_pos + 1)}
        for _, g in df.groupby(patient_col):
            times = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            for idx, t in enumerate(times[:max_pos]):
                pt[idx + 1].append(float(t))
        return {k: np.array(v) for k, v in pt.items() if len(v) >= 3}

    rp = _extract_pos(real,  n_pos)
    sp = _extract_pos(synth, n_pos)
    ks_t = []
    for pos in sorted(set(rp) & set(sp)):
        if len(rp[pos]) >= 2 and len(sp[pos]) >= 2:
            ks_t.append(ks_2samp(rp[pos], sp[pos])[0])
    if ks_t:
        tfs.timing_score = _ks_to_score(float(np.mean(ks_t)))

    # ── 6. Interval score ─────────────────────────────────────────────
    def _iv(df):
        vals = []
        for _, g in df.groupby(patient_col):
            t = pd.to_numeric(g[time_col], errors="coerce").dropna().sort_values().values
            if len(t) >= 2:
                vals.extend(np.diff(t).tolist())
        return np.array(vals)

    iv_r, iv_s = _iv(real), _iv(synth)
    if len(iv_r) > 5 and len(iv_s) > 5:
        tfs.interval_score = _ks_to_score(ks_2samp(iv_r, iv_s)[0])

    # ── 7. t_FUP coverage ────────────────────────────────────────────
    if fup_col and fup_col in synth.columns and fup_col in real.columns:
        def _cov(df):
            last_t = df.groupby(patient_col)[time_col].apply(
                lambda s: pd.to_numeric(s, errors="coerce").max()
            ).rename("last_visit")
            fup = df.groupby(patient_col)[fup_col].first().rename("fup")
            m   = pd.concat([last_t, fup], axis=1).dropna()
            m["fup"] = pd.to_numeric(m["fup"], errors="coerce")
            m = m[m["fup"] > 0]
            if m.empty:
                return float("nan")
            return float((m["last_visit"] / m["fup"]).clip(0, 1).median())

        sc = _cov(synth)
        rc = _cov(real)
        if not (np.isnan(sc) or np.isnan(rc)):
            base    = 1.0 - abs(sc - rc)
            penalty = max(0.0, (0.9 - sc) * 2.0) if sc < 0.9 else 0.0
            tfs.fup_coverage = float(np.clip(base - penalty, 0.0, 1.0))

    # ── 8. Overall weighted average ───────────────────────────────────
    weights = tfs._weights
    weighted_sum, total_w = 0.0, 0.0
    for name, w in weights.items():
        v = getattr(tfs, name)
        if not np.isnan(v):
            weighted_sum += v * w
            total_w      += w

    if total_w > 0:
        tfs.overall = float(weighted_sum / total_w)

    # ── 9. Grade ─────────────────────────────────────────────────────
    o = tfs.overall
    if not np.isnan(o):
        if o >= 0.85:   tfs.grade = "Eccellente  [****]"
        elif o >= 0.70: tfs.grade = "Buona       [***]"
        elif o >= 0.55: tfs.grade = "Accettabile [**]"
        else:           tfs.grade = "Scadente    [*]"

    return tfs