"""
eval/scores.py  
Definizioni delle classi Score con pesi, aggregazione e radar_values().

Ogni classe Score:
- dichiara i propri campi raw + _W (pesi e direzioni)
- ha compute_overall() per aggregare in [0,1]
- ha to_dict()        per il report testuale / tabella PDF
- ha radar_values()   per i grafici radar

Score:
- StatisticalFidelityScore  (SFS)           
- LongitudinalFidelityScore (LFS)            
- TemporalFidelityScore     (TFS)            
- UtilityScore              
- PrivacyScore              

Convenzione: 1 = ottimo, 0 = pessimo in tutti i campi score.
"""

import numpy as np
from dataclasses import dataclass, field


#utils
def _fmt(v) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    try:
        return round(float(v), 4)
    except Exception:
        return "N/A"


def _grade(v: float) -> str:
    if np.isnan(v):
        return "N/A"
    if v >= 0.85: return "Eccellente  [****]"
    if v >= 0.70: return "Buona       [***]"
    if v >= 0.55: return "Accettabile [**]"
    return "Scadente    [*]"


def _lower_to_score(raw: float, max_val: float = 1.0) -> float:
    if np.isnan(raw) or raw < 0:
        return float("nan")
    return float(np.clip(1.0 - raw / max_val, 0.0, 1.0))


def _reldiff_to_score(diff: float) -> float:
    if np.isnan(diff) or diff < 0:
        return float("nan")
    return float(1.0 / (1.0 + diff))


#region STATISTICAL FIDELITY SCORE
# aggiungere score UMAP
@dataclass
class StatisticalFidelityScore:
    """
    Score di fidelity statistica (marginal distributions + correlation + PCA).

    Componenti:
      ks_raw        : KS distance media sulle variabili continue (lower = better)
      cat_test_raw  : Cramer's V medio per variabili categoriche (higher=better)
      corr_c_raw    : MAE Pearson correlation matrix continue (lower = better)
      corr_k_raw    : MAE Cramér's V correlation matrix categoriche (lower = better)
      pca_c_raw     : PCA centroid similarity (higher = better)
      pca_o_raw     : PCA distribution overlap (higher = better)
    """
    ks_raw:          float = float("nan")
    cat_test_raw:    float = float("nan")
    corr_c_raw:      float = float("nan")
    corr_k_raw:      float = float("nan")
    pca_c_raw:       float = float("nan")
    pca_o_raw:       float = float("nan")
    overall:         float = float("nan")
    grade:           str   = "N/A"

    cat_test_details: dict = field(default_factory=dict)

    _W: dict = field(default_factory=lambda: {
        "ks":       (1.5, "lower"),
        "cat_test": (1.5, "higher"),
        "corr_c":   (1.5, "lower"),
        "corr_k":   (1.5, "lower"),
        "pca_c":    (1.0, "higher"),
        "pca_o":    (1.0, "higher"),
    })

    def _score(self, raw: float, direction: str) -> float:
        if np.isnan(raw):
            return float("nan")
        if direction == "lower":  return _lower_to_score(raw)
        if direction == "higher": return float(np.clip(raw, 0, 1))
        return float("nan")

    def compute_overall(self):
        pairs = [
            ("ks",       self.ks_raw),
            ("cat_test", self.cat_test_raw),
            ("corr_c",   self.corr_c_raw),
            ("corr_k",   self.corr_k_raw),
            ("pca_c",    self.pca_c_raw),
            ("pca_o",    self.pca_o_raw),
        ]
        ws, tw = 0.0, 0.0
        for key, raw in pairs:
            w, direction = self._W[key]
            s = self._score(raw, direction)
            if not np.isnan(s):
                ws += s * w; tw += w
        self.overall = float(ws / tw) if tw > 0 else float("nan")
        self.grade   = _grade(self.overall)

    def to_dict(self) -> dict:
        d = {
            "KS distance (avg) [lower=better]":                _fmt(self.ks_raw),
            "Cat stat test Cramer's V (avg) [higher=better]":  _fmt(self.cat_test_raw),
            "Corr dist continuous (MAE Pearson) [lower]":      _fmt(self.corr_c_raw),
            "Corr dist categorical (MAE Cramer's V) [lower]":  _fmt(self.corr_k_raw),
            "PCA centroid similarity [higher]":                _fmt(self.pca_c_raw),
            "PCA distribution overlap [higher]":               _fmt(self.pca_o_raw),
        }
        for var, det in self.cat_test_details.items():
            test_name = det.get("test", "chi2")
            pv        = _fmt(det.get("p_value",    float("nan")))
            es        = _fmt(det.get("effect_size", float("nan")))
            d[f"  [{var}] test"]        = test_name
            d[f"  [{var}] p-value"]     = pv
            d[f"  [{var}] effect_size"] = es
        d["*** SFS overall [0-1]"] = _fmt(self.overall)
        d["*** SFS Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        pairs = [
            ("KS dist",    self.ks_raw,       "lower"),
            ("Cat test",   self.cat_test_raw,  "higher"),
            ("Corr cont",  self.corr_c_raw,    "lower"),
            ("Corr cat",   self.corr_k_raw,    "lower"),
            ("PCA cent.",  self.pca_c_raw,     "higher"),
            ("PCA ovlp.",  self.pca_o_raw,     "higher"),
        ]
        return {
            lbl: (self._score(raw, d) if not np.isnan(self._score(raw, d)) else 0.0)
            for lbl, raw, d in pairs
        }


# region LONGITUDINAL FIDELITY SCORE 
@dataclass
class LongitudinalFidelityScore:
    """
    Score di fidelity longitudinale (traiettorie, varianza intra-paziente,
    autocorrelazione, coerenza temporale, DTW).
    """
    slope_raw:  float = float("nan")
    var_raw:    float = float("nan")
    ac_raw:     float = float("nan")
    coh_raw:    float = float("nan")
    dtw_raw:    float = float("nan")
    overall:    float = float("nan")
    grade:      str   = "N/A"

    timing_raw: float = float("nan")
    iv_raw:     float = float("nan")
    vc_raw:     float = float("nan")
    iv_real:    float = float("nan")
    iv_synth:   float = float("nan")
    ac_lagk:    dict  = field(default_factory=dict)
    lme_betas:  dict  = field(default_factory=dict)

    _W: dict = field(default_factory=lambda: {
        "slope": (1.5, "lower"),
        "var":   (1.5, "lower"),
        "ac":    (1.0, "lower"),
        "coh":   (2.5, "higher"),
        "dtw":   (1.5, "lower"),
    })

    def _score(self, raw: float, direction: str) -> float:
        if np.isnan(raw):
            return float("nan")
        if direction == "lower":   return _lower_to_score(raw)
        if direction == "higher":  return float(np.clip(raw, 0, 1))
        if direction == "reldiff": return _reldiff_to_score(raw)
        return float("nan")

    def compute_overall(self):
        pairs = [
            ("slope", self.slope_raw),
            ("var",   self.var_raw),
            ("ac",    self.ac_raw),
            ("coh",   self.coh_raw),
            ("dtw",   self.dtw_raw),
        ]
        ws, tw = 0.0, 0.0
        for key, raw in pairs:
            w, direction = self._W[key]
            s = self._score(raw, direction)
            if not np.isnan(s):
                ws += s * w; tw += w
        self.overall = float(ws / tw) if tw > 0 else float("nan")
        self.grade   = _grade(self.overall)

    def to_dict(self) -> dict:
        d = {
            "KS slopes (avg) [lower]":          _fmt(self.slope_raw),
            "KS within-variance (avg) [lower]": _fmt(self.var_raw),
            "KS autocorr lag-1 (avg) [lower]":  _fmt(self.ac_raw),
            "Temporal coherence TCS [higher]":  _fmt(self.coh_raw),
            "DTW normalised (avg) [lower]":     _fmt(self.dtw_raw),
        }
        d["*** LFS overall [0-1]"] = _fmt(self.overall)
        d["*** LFS Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        pairs = [
            ("Slopes",    self.slope_raw, "lower"),
            ("Variance",  self.var_raw,   "lower"),
            ("Autocorr",  self.ac_raw,    "lower"),
            ("Coherence", self.coh_raw,   "higher"),
            ("DTW",       self.dtw_raw,   "lower"),
        ]
        return {
            lbl: (self._score(raw, d) if not np.isnan(self._score(raw, d)) else 0.0)
            for lbl, raw, d in pairs
        }


# region TEMPORAL FIDELITY SCORE 
@dataclass
class TemporalFidelityScore:
    """
    Score di fidelity temporale (struttura delle visite e timing).
    """
    visit_count_score: float = float("nan")
    timing_score:      float = float("nan")
    interval_score:    float = float("nan")
    fup_coverage:      float = float("nan")
    overall:           float = float("nan")
    grade:             str   = "N/A"

    _W: dict = field(default_factory=lambda: {
        "visit_count_score": (1.5, "higher"),
        "timing_score":      (2.5, "higher"),
        "interval_score":    (2.0, "higher"),
        "fup_coverage":      (2.0, "higher"),
    })

    def compute_overall(self):
        ws, tw = 0.0, 0.0
        for name, (w, _) in self._W.items():
            v = getattr(self, name)
            if not np.isnan(v):
                ws += v * w; tw += w
        self.overall = float(ws / tw) if tw > 0 else float("nan")
        self.grade   = _grade(self.overall)

    def to_dict(self) -> dict:
        return {
            "TFS visit count score [0-1]":  _fmt(self.visit_count_score),
            "TFS timing score [0-1]":       _fmt(self.timing_score),
            "TFS interval score [0-1]":     _fmt(self.interval_score),
            "TFS t_FUP coverage [0-1]":     _fmt(self.fup_coverage),
            "*** TFS overall [0-1]":        _fmt(self.overall),
            "*** TFS Grade":                self.grade,
        }

    def radar_values(self) -> dict:
        return {
            "Visit cnt": self.visit_count_score if not np.isnan(self.visit_count_score) else 0.0,
            "Timing":    self.timing_score      if not np.isnan(self.timing_score)      else 0.0,
            "Interval":  self.interval_score    if not np.isnan(self.interval_score)    else 0.0,
            "FUP cov":   self.fup_coverage      if not np.isnan(self.fup_coverage)      else 0.0,
        }


# region UTILITY SCORE 
# da implementare
@dataclass
class UtilityScore:
    """
    Score di utilità dei dati sintetici. Idee:
    - TSTR 
    - Survival
    - Next step prediction score (longitudinal utility)
    """

    def compute_overall(self):
        pass


# region PRIVACY SCORE 
@dataclass
class PrivacyScore:
    """
    Score di privacy dei dati sintetici.

    Metrics:
      dcr_score            : clip(median_DCR / typical_spacing, 0, 2) / 2
                             DCR row-wise (spazio standardizzato).
      dcr_trajectory_score : DCR calcolato su feature embedding DTW per-paziente.
                             Cattura la copia di traiettorie intere.
      nndr_score           : fraction(NNDR >= 0.5). NNDR = d1nn / d2nn.

    
      overall   : DCR 25% | DCR_traj 20% | NNDR 25% 
    """

    dcr_score:             float = float("nan")
    dcr_trajectory_score:  float = float("nan")
    nndr_score:            float = float("nan")


    dcr_mean:              float = float("nan")
    dcr_median:            float = float("nan")
    nndr_frac_lt05:        float = float("nan")

    overall:               float = float("nan")
    grade:                 str   = "N/A"

    def compute_overall(self):
        def _wavg(pairs):
            ws, tw = 0.0, 0.0
            for val, w in pairs:
                if isinstance(val, (int, float)) and not np.isnan(float(val)):
                    ws += float(val) * w; tw += w
            return float(ws / tw) if tw > 0 else float("nan")

        self.overall = _wavg([
            (self.dcr_score,            0.35),
            (self.dcr_trajectory_score, 0.30),
            (self.nndr_score,           0.35)
        ])

        self.grade = _grade(self.overall)

    def to_dict(self) -> dict:
        d = {}

        d["[Privacy] DCR_mean"]                      = _fmt(self.dcr_mean)
        d["[Privacy] DCR_median"]                    = _fmt(self.dcr_median)
        d["[Privacy] NNDR_fraction_lt_0.5"]          = _fmt(self.nndr_frac_lt05)
        d["[Privacy] DCR_score (row-level) [0-1]"]   = _fmt(self.dcr_score)
        d["[Privacy] DCR_trajectory_score [0-1]"]    = _fmt(self.dcr_trajectory_score)
        d["[Privacy] NNDR_score [0-1]"]              = _fmt(self.nndr_score)

        
        d["*** Privacy overall [0-1]"] = _fmt(self.overall)
        d["*** Privacy Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        def _safe(v):
            try:
                f = float(v)
                return 0.0 if np.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0

        return {
            "DCR (row)":    _safe(self.dcr_score),
            "DCR (traj)":   _safe(self.dcr_trajectory_score),
            "NNDR":         _safe(self.nndr_score)
        }