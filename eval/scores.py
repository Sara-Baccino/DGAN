"""
eval/scores.py

Definizioni delle classi Score con pesi, aggregazione e radar_values().

Ogni classe Score:
  - dichiara i propri campi raw + _W (pesi e direzioni)
  - ha compute_overall() per aggregare in [0,1]
  - ha to_dict()        per il report testuale / tabella PDF
  - ha radar_values()   per i grafici radar

Score inclusi:
  - StatisticalFidelityScore  (SFS)
  - LongitudinalFidelityScore (LFS)
  - TemporalFidelityScore     (TFS)
  - UtilityScore
  - PrivacyScore
"""

import numpy as np
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY CONDIVISE
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# STATISTICAL FIDELITY SCORE (SFS)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StatisticalFidelityScore:
    """
    Score di fidelity statistica (marginal distributions + correlation + PCA).

    Componenti:
      ks_raw        : KS distance media sulle variabili continue (lower = better)
      cat_test_raw  : p-value medio dei test Chi²/Fisher per-variabile categorica
                      (higher = better: p alto → distribuzioni simili)
      corr_c_raw    : MAE Pearson correlation matrix continue (lower = better)
      corr_k_raw    : MAE Cramér's V correlation matrix categoriche (lower = better)
      pca_c_raw     : PCA centroid similarity (higher = better)
      pca_o_raw     : PCA distribution overlap (higher = better)
    """
    ks_raw:          float = float("nan")
    cat_test_raw:    float = float("nan")   # p-value medio Chi²/Fisher
    corr_c_raw:      float = float("nan")
    corr_k_raw:      float = float("nan")
    pca_c_raw:       float = float("nan")
    pca_o_raw:       float = float("nan")
    overall:         float = float("nan")
    grade:           str   = "N/A"

    # Dettagli per-variabile del test categorico (non incidono sullo score)
    cat_test_details: dict = field(default_factory=dict)

    _W: dict = field(default_factory=lambda: {
        "ks":       (1.5, "lower"),
        "cat_test": (1.5, "higher"),   # p-value alto = distribuzioni simili
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
        # Dettagli per-variabile (Chi² / Fisher)
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


# ──────────────────────────────────────────────────────────────────────────────
# LONGITUDINAL FIDELITY SCORE (LFS)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LongitudinalFidelityScore:
    """
    Score di fidelity longitudinale (traiettorie, varianza intra-paziente,
    autocorrelazione, coerenza temporale, DTW).

    Componenti incluse nello score:
      slope_raw  : avg KS distribuzione slopes per-paziente (lower = better)
      var_raw    : avg KS varianza intra-paziente (lower = better)
      ac_raw     : avg KS autocorrelazione lag-1 (lower = better)
      coh_raw    : Temporal Coherence Score - TCS (higher = better)
      dtw_raw    : DTW normalizzato medio (lower = better)

    Campi informativi (non pesati nello score):
      timing_raw, iv_raw, vc_raw, iv_real, iv_synth, ac_lagk, lme_betas
    """
    slope_raw:  float = float("nan")
    var_raw:    float = float("nan")
    ac_raw:     float = float("nan")
    coh_raw:    float = float("nan")
    dtw_raw:    float = float("nan")
    overall:    float = float("nan")
    grade:      str   = "N/A"

    # Informativi (non nello score)
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
            # Informativi
            "[info] KS timing positional":      _fmt(self.timing_raw),
            "[info] Interval reldiff":          _fmt(self.iv_raw),
            "[info] KS visit count":            _fmt(self.vc_raw),
            "[info] Inter-visit real (mo)":     _fmt(self.iv_real),
            "[info] Inter-visit synth (mo)":    _fmt(self.iv_synth),
        }
        for lag, ks in self.ac_lagk.items():
            d[f"KS autocorr lag-{lag}"] = _fmt(ks)
        for var, betas in self.lme_betas.items():
            d[f"LME beta real  [{var}]"]  = _fmt(betas.get("beta_real",  float("nan")))
            d[f"LME beta synth [{var}]"]  = _fmt(betas.get("beta_synth", float("nan")))
            d[f"LME |Dbeta|    [{var}]"]  = _fmt(betas.get("beta_diff_abs", float("nan")))
        d["*** LFS overall [0-1]"] = _fmt(self.overall)
        d["*** LFS Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        pairs = [
            ("Slopes",   self.slope_raw, "lower"),
            ("Variance", self.var_raw,   "lower"),
            ("Autocorr", self.ac_raw,    "lower"),
            ("Coherence",self.coh_raw,   "higher"),
            ("DTW",      self.dtw_raw,   "lower"),
        ]
        return {
            lbl: (self._score(raw, d) if not np.isnan(self._score(raw, d)) else 0.0)
            for lbl, raw, d in pairs
        }


# ──────────────────────────────────────────────────────────────────────────────
# TEMPORAL FIDELITY SCORE (TFS)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TemporalFidelityScore:
    """
    Score di fidelity temporale (struttura delle visite e timing).

    Componenti incluse nello score:
      visit_count_score : 1 - KS distribuzione numero visite per paziente
      timing_score      : 1 - KS timing posizionale medio (sequenza visite)
      interval_score    : 1 / (1 + reldiff inter-visit interval)
      fup_coverage      : coverage last_visit / t_FUP (penalizzata se < 0.9)
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


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY SCORE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UtilityScore:
    """
    Score di utilità dei dati sintetici.

    Componenti:
      discriminator_score : 1 - |AUC_RF - 0.5| * 2
                            AUC ≈ 0.5 → indistinguibili (ottimo); 1.0 → separabili (pessimo)
      pmse_score          : 1 - clip(pMSE / 0.25, 0, 1)
                            propensity MSE → 0 ideale
      tstr_scores         : {target: score} da TSTR per ogni outcome binario
      tstr_details        : {target: {AUC_TRTR, AUC_TSTR, gap}} (informativi)
    """
    discriminator_score: float = float("nan")
    pmse_score:          float = float("nan")
    tstr_scores:         dict  = field(default_factory=dict)   # {target: float}
    tstr_details:        dict  = field(default_factory=dict)   # {target: {AUC_TRTR, AUC_TSTR, gap}}
    overall:             float = float("nan")
    grade:               str   = "N/A"

    def compute_overall(self):
        scores = []
        for v in [self.discriminator_score, self.pmse_score]:
            if not np.isnan(v):
                scores.append(v)
        for s in self.tstr_scores.values():
            if not np.isnan(s):
                scores.append(s)
        self.overall = float(np.mean(scores)) if scores else float("nan")
        self.grade   = _grade(self.overall)

    def to_dict(self) -> dict:
        d = {
            "discriminator_score [0-1, 1=indistinguibili]": _fmt(self.discriminator_score),
            "pMSE_score [0-1, 1=simili]":                   _fmt(self.pmse_score),
        }
        for tgt, details in self.tstr_details.items():
            d[f"TSTR_{tgt}_AUC_TRTR"]     = _fmt(details.get("AUC_TRTR", float("nan")))
            d[f"TSTR_{tgt}_AUC_TSTR"]     = _fmt(details.get("AUC_TSTR", float("nan")))
            d[f"TSTR_{tgt}_gap"]          = _fmt(details.get("gap",       float("nan")))
            d[f"TSTR_{tgt}_score [0-1]"]  = _fmt(self.tstr_scores.get(tgt, float("nan")))
        d["*** Utility overall [0-1]"] = _fmt(self.overall)
        d["*** Utility Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        rv = {
            "Discriminator": self.discriminator_score if not np.isnan(self.discriminator_score) else 0.0,
            "pMSE":          self.pmse_score          if not np.isnan(self.pmse_score)          else 0.0,
        }
        for tgt, s in self.tstr_scores.items():
            rv[f"TSTR {tgt}"] = s if not np.isnan(s) else 0.0
        return rv


# ──────────────────────────────────────────────────────────────────────────────
# PRIVACY SCORE
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PrivacyScore:
    """
    Score di privacy dei dati sintetici.

    Componenti:
      dcr_score           : clip(median_DCR / typical_spacing, 0, 2) / 2
                            DCR = distanza del sintetico al record reale più vicino,
                            normalizzata sulla spaziatura mediana reale–reale.
                            Alto = sintetici lontani dai reali = più privati.

      nndr_score          : fraction(NNDR >= 0.5)
                            NNDR[i] = d(synth_i, NN1_real) / d(synth_i, NN2_real).
                            NNDR < 0.5 → sintetico molto vicino a UN solo reale
                            → rischio di copying/memorizzazione.

      attribute_inference : 1 - clip((AUC_ai - 0.5) / 0.5, 0, 1)
                            AUC di un modello che inferisce una feature sensibile
                            da tutte le altre sul dataset sintetico.
                            AUC = 0.5 → feature non inferibile → score = 1.

      membership_inference: 1 - fraction(sintetici con DCR < dcr_p5_real)
                            Stima della fraction di sintetici "troppo vicini"
                            ai reali di training (soglia = 5° percentile DCR reale–reale).
                            0 violazioni → score = 1.

    Campi informativi (non pesati):
      dcr_mean, dcr_median, nndr_frac_lt1, mi_threshold, ai_target
    """
    dcr_score:            float = float("nan")
    nndr_score:           float = float("nan")
    attribute_inference:  float = float("nan")
    membership_inference: float = float("nan")
    overall:              float = float("nan")
    grade:                str   = "N/A"

    # Informativi
    dcr_mean:      float = float("nan")
    dcr_median:    float = float("nan")
    nndr_frac_lt1: float = float("nan")
    mi_threshold:  float = float("nan")   # soglia DCR p5 usata per MIA
    ai_target:     str   = ""             # feature usata per attribute inference

    def compute_overall(self):
        scores = []
        for v in [self.dcr_score, self.nndr_score,
                  self.attribute_inference, self.membership_inference]:
            if not np.isnan(v):
                scores.append(v)
        self.overall = float(np.mean(scores)) if scores else float("nan")
        self.grade   = _grade(self.overall)

    def to_dict(self) -> dict:
        d = {
            "DCR_mean":                   _fmt(self.dcr_mean),
            "DCR_median":                 _fmt(self.dcr_median),
            "NNDR_fraction_lt_0.5":       _fmt(self.nndr_frac_lt1),
            "DCR_score [0-1]":            _fmt(self.dcr_score),
            "NNDR_score [0-1]":           _fmt(self.nndr_score),
            "Attribute inference [0-1]":  _fmt(self.attribute_inference),
            "Membership inference [0-1]": _fmt(self.membership_inference),
        }
        if self.ai_target:
            d["  AI target feature"] = self.ai_target
        if not np.isnan(self.mi_threshold):
            d["  MI DCR threshold (p5 real-real)"] = _fmt(self.mi_threshold)
        d["*** Privacy overall [0-1]"] = _fmt(self.overall)
        d["*** Privacy Grade"]         = self.grade
        return d

    def radar_values(self) -> dict:
        return {
            "DCR":        self.dcr_score            if not np.isnan(self.dcr_score)            else 0.0,
            "NNDR":       self.nndr_score            if not np.isnan(self.nndr_score)           else 0.0,
            "Attr. Inf.": self.attribute_inference   if not np.isnan(self.attribute_inference)  else 0.0,
            "Memb. Inf.": self.membership_inference  if not np.isnan(self.membership_inference) else 0.0,
        }