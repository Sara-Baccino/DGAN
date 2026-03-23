"""
config/config_loader.py  [v3 — compatibile con JSON utente]
================================================================================
Compatibile con il JSON esistente dell'utente senza modifiche al JSON.

STRUTTURA JSON SUPPORTATA
──────────────────────────
{
  "time": {
    "max_visits":   <int>,          ← numero massimo di visite per paziente
    "visit_column": <str>,          ← colonna tempo nel DataFrame long
    "patient_id":   <str>,          ← colonna ID paziente
    "fup_column":   <str|null>,     ← colonna t_FUP (default "t_FUP")
    "min_visits":   <int|null>      ← minimo visite per paziente generato (default 1)
  },

  "preprocessing": {                ← blocco OPZIONALE (ha tutti default)
    "mice_max_iter": <int>,         ← iterazioni MICE (default 10)
    "knn_neighbors": <int>,         ← vicini KNN categoriche (default 5)
    "log_vars":      [<str>, ...],  ← variabili da log1p prima dello scaling
    "clip_z":        <float>        ← clipping z-score inverse_transform (default 4.0)
  },

  "baseline": {
    "continuous":  [{"name": ..., "type": "float"}, ...],
    "categorical": {
      "VAR": {
        "type":    "int"|"string",
        "mapping": {"label": <int>, ...}
      }, ...
    }
  },

  "followup": {
    "continuous":  [{"name": ..., "type": "float"}, ...],
    "categorical": {
      "VAR": {
        "mapping":     {"label": <int>, ...},
        "irreversible": <bool>          ← opzionale, default false
      }, ...
    }
  },

  "outcomes": [<str>, ...],         ← ignorato dal loader (solo documentazione)

  "model": {
    "z_static_dim":   <int>,        ← dim rumore statico (default 64)
    "z_temporal_dim": <int>,        ← dim rumore temporale (default 32)
    "hidden":         <int>,        ← hidden_dim globale (default 128)

    "epochs":         <int>,
    "patience":       <int>,
    "batch_size":     <int>,
    "lr_g":           <float>,
    "lr_d_s":         <float>,
    "lr_d_t":         <float|null>, ← default = lr_d_s

    "optimizer_betas":     [<f>,<f>], ← default [0.5, 0.9]
    "dataloader_drop_last": <bool>,   ← default true
    "noise_ar_rho":         <float>,  ← correlazione AR rumore z_t (default 0.0)

    "generator": {
      "arch":           "gru",
      "hidden_dim":     <int>,
      "n_layers":       <int>,
      "dropout":        <float>,
      "bidirectional":  <bool>        ← legacy, ignorato
    },

    "static_discriminator": {
      "static_layers":  <int>,
      "mlp_hidden_dim": <int>,
      "dropout":        <float>
    },

    "temporal_discriminator": {
      "arch":         "cnn"|"gru",
      "hidden_dim":   <int>,
      "kernel_size":  <int>,
      "n_layers":     <int>,
      "dilation_base":<int>,
      "mlp_layers":   <int>,
      "dropout":      <float>
    },

    "noise_std":   <float>,
    "grad_clip":   <float>,
    "use_dp":      <bool>,

    "critic_steps":          <int>,
    "critic_steps_temporal": <int>,   ← -1 → uguale a critic_steps

    "gumbel_temperature_start": <float>,
    "temperature_min":          <float>,

    "lambda_gp_s":       <float>,
    "lambda_gp_t":       <float>,
    "lambda_gp":         <float>,     ← legacy alias per gp_s e gp_t se presenti
    "alpha_irr":         <float>,
    "lambda_aux":        <float>,
    "lambda_fm":         <float>,
    "lambda_nv":         <float>,
    "lambda_fup":        <float>,
    "lambda_static_cat": <float>,
    "lambda_var":        <float>,     ← varianza feature continue (default 0.5)

    ← Campi ignorati (presenti nel tuo JSON ma rimossi dall'architettura gretel):
    "lambda_freq_gen", "lambda_freq_disc", "freq_weight_power",
    "lambda_fc", "lambda_sc_var", "lambda_ivi",
    "lambda_coverage", "lambda_uniformity",
    "n_visits_sharpness", "fixed_visits",
    "warmup_mask_frac", "finetune_mask_frac",
    "force_full_mask", "regular"
  }
}

NOTA SU t_FUP IN BASELINE
──────────────────────────
Se "t_FUP" (o qualsiasi fup_column) è dichiarata anche in "baseline.continuous",
viene automaticamente ESCLUSA dalle variabili statiche continue per evitare
duplicazione, poiché viene già gestita internamente come target di follow-up.

RETROCOMPATIBILITÀ
──────────────────
load_config restituisce 4 valori: (TimeConfig, List[VariableConfig], ModelConfig, PreprocessingConfig)
Il tuo main.py usa:
    time_cfg, variables, model_cfg, prep_cfg = load_config(config_path)
================================================================================
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ======================================================================
# HELPER — unico punto di null-safety
# ======================================================================

def _get(cfg, key: str, default):
    """
    Legge cfg[key]. Restituisce default se:
      - cfg non è un dict
      - la chiave manca
      - il valore è None/null
    """
    if not isinstance(cfg, dict):
        return default
    val = cfg.get(key, default)
    return val if val is not None else default


# ======================================================================
# TIME CONFIG
# ======================================================================

@dataclass
class TimeConfig:
    max_visits:   int
    visit_column: str
    patient_id:   str
    fup_column:   str = "t_FUP"
    min_visits:   int = 1


# ======================================================================
# PREPROCESSING CONFIG
# ======================================================================

@dataclass
class PreprocessingConfig:
    mice_max_iter: int       = 10
    knn_neighbors: int       = 5
    log_vars:      List[str] = field(default_factory=list)
    clip_z:        float     = 4.0


# ======================================================================
# VARIABLE CONFIG
# ======================================================================

@dataclass
class VariableConfig:
    name:         str
    kind:         str           # "continuous" | "categorical"
    static:       bool
    dtype:        str
    mapping:      Optional[Dict] = None
    irreversible: bool           = False

    @property
    def n_categories(self) -> Optional[int]:
        return len(self.mapping) if self.mapping is not None else None


# ======================================================================
# GENERATOR CONFIG
# ======================================================================

@dataclass
class GeneratorConfig:
    hidden_dim:           int
    n_layers:             int   = 2
    dropout:              float = 0.1
    # Campi legacy (presenti nel JSON dell'utente, non usati dal nuovo generator.py)
    arch:                 str   = "gru"
    bidirectional:        bool  = True
    z_static_dim:         int   = 0
    z_temporal_dim:       int   = 0
    gru_layers:           int   = 2
    n_transformer_layers: int   = 2
    n_heads:              int   = 4
    pe_frequencies:       int   = 16


# ======================================================================
# DISCRIMINATOR CONFIGS
# ======================================================================

@dataclass
class DiscriminatorConfig:
    static_layers:  int
    mlp_hidden_dim: int
    dropout:        float


@dataclass
class TempDiscriminatorConfig:
    arch:           str   = "cnn"
    hidden_dim:     int   = 64
    kernel_size:    int   = 3
    n_layers:       int   = 3
    dilation_base:  int   = 2
    mlp_layers:     int   = 2
    dropout:        float = 0.1
    # GRU-only
    mlp_hidden_dim: int   = 128
    gru_hidden_dim: int   = 64
    gru_layers:     int   = 2


# ======================================================================
# MODEL CONFIG
# ======================================================================

@dataclass
class ModelConfig:
    # ── Spazio latente ────────────────────────────────────────────────
    z_static_dim:   int
    z_temporal_dim: int
    hidden:         int

    # ── Training ─────────────────────────────────────────────────────
    epochs:     int
    batch_size: int
    lr_g:       float
    lr_d_s:     float
    lr_d_t:     float

    # ── Architetture ─────────────────────────────────────────────────
    generator:              GeneratorConfig
    static_discriminator:   DiscriminatorConfig
    temporal_discriminator: TempDiscriminatorConfig

    # ── Training misc ─────────────────────────────────────────────────
    noise_std:    float
    critic_steps: int
    grad_clip:    float
    patience:     int
    use_dp:       bool

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer_betas:      List[float] = field(default_factory=lambda: [0.5, 0.9])
    dataloader_drop_last: bool        = True

    # ── Rumore AR temporale ───────────────────────────────────────────
    noise_ar_rho: float = 0.4

    # ── Gumbel temperature ───────────────────────────────────────────
    gumbel_temperature_start: float = 1.0
    temperature_min:          float = 0.5

    # ── Critic steps ─────────────────────────────────────────────────
    critic_steps_temporal: int = -1

    # ── Loss: gradient penalty ────────────────────────────────────────
    lambda_gp_s: float = 10.0
    lambda_gp_t: float = 10.0

    # ── Loss: ausiliarie ─────────────────────────────────────────────
    lambda_aux:        float = 0.2
    alpha_irr:         float = 0.1
    lambda_fup:        float = 1.0
    lambda_nv:         float = 1.0
    lambda_static_cat: float = 2.0
    lambda_fm:         float = 0.0
    lambda_var:        float = 0.5   # varianza feature continue temporali
    lambda_interval:   float = 2.0   # distribuzione intervalli inter-visita (media+std)

    # ── EMA generatore ────────────────────────────────────────────────
    # decay=0 = EMA disabilitata; 0.999 = standard per GAN su dati clinici
    ema_decay:              float = 0.0

    # ── Instance noise sulle OHE reali (anti-collapse categoriche) ────
    # Rumore additivo U(0, ε) sulle OHE reali prima del discriminatore.
    # Decade linearmente da instance_noise_start → instance_noise_end
    # nel corso del training. 0 = disabilitato.
    instance_noise_start:  float = 0.05
    instance_noise_end:    float = 0.0

    # ── Lambda_scat warmup (anti-collapse categoriche statiche) ───────
    # Per le prime lambda_scat_warmup_epochs epoche, lambda_scat è
    # moltiplicato per un fattore che cresce da 0→1 (cosine schedule).
    # 0 = nessun warmup (comportamento precedente).
    lambda_scat_warmup_epochs: int = 0

    def __post_init__(self):
        if self.critic_steps_temporal < 0:
            self.critic_steps_temporal = self.critic_steps
        if not isinstance(self.optimizer_betas, (list, tuple)) or len(self.optimizer_betas) != 2:
            self.optimizer_betas = [0.5, 0.9]

    @property
    def optimizer_beta1(self) -> float:
        return float(self.optimizer_betas[0])

    @property
    def optimizer_beta2(self) -> float:
        return float(self.optimizer_betas[1])


# ======================================================================
# DATA CONFIG
# ======================================================================

@dataclass
class DataConfig:
    max_len:        int
    min_visits:     int
    patient_id_col: str
    time_col:       str
    fup_col:        str

    static_cont:   List[VariableConfig]
    static_cat:    List[VariableConfig]
    temporal_cont: List[VariableConfig]
    temporal_cat:  List[VariableConfig]

    n_static_cont:    int
    n_static_cat:     List[int]
    n_temp_cont:      int
    n_temp_cat:       List[int]
    irreversible_idx: List[int]


# ======================================================================
# BUILD FUNCTIONS
# ======================================================================

def build_data_config(
    time_cfg:  TimeConfig,
    variables: List[VariableConfig],
) -> DataConfig:
    static_cont      = [v for v in variables if v.static     and v.kind == "continuous"]
    static_cat       = [v for v in variables if v.static     and v.kind == "categorical"]
    temporal_cont    = [v for v in variables if not v.static and v.kind == "continuous"]
    temporal_cat     = [v for v in variables if not v.static and v.kind == "categorical"]
    irreversible_idx = [i for i, v in enumerate(temporal_cat) if v.irreversible]

    return DataConfig(
        max_len          = time_cfg.max_visits,
        min_visits       = time_cfg.min_visits,
        patient_id_col   = time_cfg.patient_id,
        time_col         = time_cfg.visit_column,
        fup_col          = time_cfg.fup_column,
        static_cont      = static_cont,
        static_cat       = static_cat,
        temporal_cont    = temporal_cont,
        temporal_cat     = temporal_cat,
        n_static_cont    = len(static_cont),
        n_static_cat     = [len(v.mapping) for v in static_cat],
        n_temp_cont      = len(temporal_cont),
        n_temp_cat       = [len(v.mapping) for v in temporal_cat],
        irreversible_idx = irreversible_idx,
    )


def _build_generator_config(gen_raw: dict, global_hidden: int) -> GeneratorConfig:
    gru_layers = _get(gen_raw, "gru_layers", _get(gen_raw, "n_layers", 2))
    n_layers   = _get(gen_raw, "n_layers", gru_layers)
    return GeneratorConfig(
        hidden_dim           = _get(gen_raw, "hidden_dim",            global_hidden),
        n_layers             = n_layers,
        dropout              = _get(gen_raw, "dropout",               0.1),
        arch                 = _get(gen_raw, "arch",                  "gru"),
        bidirectional        = _get(gen_raw, "bidirectional",         True),
        z_static_dim         = _get(gen_raw, "z_static_dim",          0),
        z_temporal_dim       = _get(gen_raw, "z_temporal_dim",        0),
        gru_layers           = gru_layers,
        n_transformer_layers = _get(gen_raw, "n_transformer_layers",  n_layers),
        n_heads              = _get(gen_raw, "n_heads",               4),
        pe_frequencies       = _get(gen_raw, "pe_frequencies",        16),
    )


def _build_temp_disc_config(td_raw: dict) -> TempDiscriminatorConfig:
    return TempDiscriminatorConfig(
        arch           = _get(td_raw, "arch",            "cnn"),
        hidden_dim     = _get(td_raw, "hidden_dim",       64),
        kernel_size    = _get(td_raw, "kernel_size",       3),
        n_layers       = _get(td_raw, "n_layers",          3),
        dilation_base  = _get(td_raw, "dilation_base",     2),
        mlp_layers     = _get(td_raw, "mlp_layers",        2),
        dropout        = _get(td_raw, "dropout",          0.1),
        mlp_hidden_dim = _get(td_raw, "mlp_hidden_dim",  128),
        gru_hidden_dim = _get(td_raw, "gru_hidden_dim",   64),
        gru_layers     = _get(td_raw, "gru_layers",        2),
    )


def build_model_config(cfg: dict) -> ModelConfig:
    """
    Legge il blocco "model" del JSON e costruisce ModelConfig.
    Gestisce:
      - I campi presenti nel JSON dell'utente (inclusi quelli legacy ignorati)
      - lambda_gp come alias di gp_s e gp_t se lambda_gp_s/t non sono presenti
      - critic_steps_temporal: -1 se mancante → allineato a critic_steps in __post_init__
    """
    global_hidden = _get(cfg, "hidden", 128)
    lr_d_s        = _get(cfg, "lr_d_s", 1e-4)

    # lambda_gp: alias legacy per gp_s e gp_t se i valori specifici mancano
    lambda_gp_fallback = _get(cfg, "lambda_gp", 10.0)
    lambda_gp_s = _get(cfg, "lambda_gp_s", lambda_gp_fallback)
    lambda_gp_t = _get(cfg, "lambda_gp_t", lambda_gp_fallback)

    disc_raw = _get(cfg, "static_discriminator", {})
    disc_cfg = DiscriminatorConfig(
        static_layers  = _get(disc_raw, "static_layers",  3),
        mlp_hidden_dim = _get(disc_raw, "mlp_hidden_dim", global_hidden),
        dropout        = _get(disc_raw, "dropout",        0.05),
    )

    # Campi presenti nel JSON dell'utente ma rimossi dall'architettura gretel.
    # Vengono letti per non sollevare errori, ma non usati.
    _ignored = [
        "lambda_freq_gen", "lambda_freq_disc", "freq_weight_power",
        "lambda_fc", "lambda_sc_var", "lambda_ivi",
        "lambda_coverage", "lambda_uniformity",
        "n_visits_sharpness", "fixed_visits",
        "warmup_mask_frac", "finetune_mask_frac",
        "force_full_mask", "regular",
    ]
    ignored_present = [k for k in _ignored if k in cfg and cfg[k] not in (None, 0, False)]
    if ignored_present:
        warnings.warn(
            f"I seguenti parametri del JSON non sono utilizzati "
            f"nell'architettura gretel-style e verranno ignorati: "
            f"{ignored_present}. Puoi rimuoverli dal config senza effetti.",
            UserWarning,
            stacklevel=3,
        )

    return ModelConfig(
        z_static_dim   = _get(cfg, "z_static_dim",   64),
        z_temporal_dim = _get(cfg, "z_temporal_dim", 32),
        hidden         = global_hidden,

        epochs         = _get(cfg, "epochs",     200),
        batch_size     = _get(cfg, "batch_size", 64),
        lr_g           = _get(cfg, "lr_g",       1e-4),
        lr_d_s         = lr_d_s,
        lr_d_t         = _get(cfg, "lr_d_t",     lr_d_s),

        optimizer_betas      = _get(cfg, "optimizer_betas",      [0.5, 0.9]),
        dataloader_drop_last = _get(cfg, "dataloader_drop_last", True),

        generator              = _build_generator_config(
                                     _get(cfg, "generator", {}), global_hidden),
        static_discriminator   = disc_cfg,
        temporal_discriminator = _build_temp_disc_config(
                                     _get(cfg, "temporal_discriminator", {})),

        noise_std             = _get(cfg, "noise_std",    1.0),
        noise_ar_rho          = _get(cfg, "noise_ar_rho", 0.0),
        critic_steps          = _get(cfg, "critic_steps", 5),
        critic_steps_temporal = _get(cfg, "critic_steps_temporal", -1),
        grad_clip             = _get(cfg, "grad_clip",    1.0),
        patience              = _get(cfg, "patience",    20),
        use_dp                = _get(cfg, "use_dp",       False),

        gumbel_temperature_start = _get(cfg, "gumbel_temperature_start", 1.0),
        temperature_min          = _get(cfg, "temperature_min",          0.5),

        lambda_gp_s = lambda_gp_s,
        lambda_gp_t = lambda_gp_t,

        lambda_aux        = _get(cfg, "lambda_aux",        0.2),
        alpha_irr         = _get(cfg, "alpha_irr",         0.1),
        lambda_fup        = _get(cfg, "lambda_fup",        1.0),
        lambda_nv         = _get(cfg, "lambda_nv",         1.0),
        lambda_static_cat = _get(cfg, "lambda_static_cat", 2.0),
        lambda_fm         = _get(cfg, "lambda_fm",         0.0),
        lambda_var        = _get(cfg, "lambda_var",         0.5),
        lambda_interval   = _get(cfg, "lambda_interval",   2.0),

        ema_decay                  = _get(cfg, "ema_decay",                  0.0),
        instance_noise_start       = _get(cfg, "instance_noise_start",       0.05),
        instance_noise_end         = _get(cfg, "instance_noise_end",         0.0),
        lambda_scat_warmup_epochs  = _get(cfg, "lambda_scat_warmup_epochs",  0),
    )


def build_preprocessing_config(cfg: dict) -> PreprocessingConfig:
    """
    Legge il blocco opzionale "preprocessing" del JSON.
    Se il blocco è assente, tutti i default entrano in vigore.
    """
    raw = _get(cfg, "preprocessing", {})
    if not isinstance(raw, dict):
        raw = {}
    return PreprocessingConfig(
        mice_max_iter = _get(raw, "mice_max_iter", 10),
        knn_neighbors = _get(raw, "knn_neighbors",  5),
        log_vars      = _get(raw, "log_vars",       []),
        clip_z        = _get(raw, "clip_z",         4.0),
    )


# ======================================================================
# LOAD CONFIG — entry point principale
# ======================================================================

def load_config(
    path: str,
) -> Tuple[TimeConfig, List[VariableConfig], ModelConfig, PreprocessingConfig]:
    """
    Carica il JSON e restituisce le quattro config del pipeline.

    Uso:
        time_cfg, variables, model_cfg, prep_cfg = load_config("config.json")
        data_cfg     = build_data_config(time_cfg, variables)
        preprocessor = Preprocessor(
            data_cfg,
            log_vars      = prep_cfg.log_vars,
            mice_max_iter = prep_cfg.mice_max_iter,
            knn_neighbors = prep_cfg.knn_neighbors,
            clip_z        = prep_cfg.clip_z,
        )
        model = DGAN(data_cfg, model_cfg, preprocessor)

    NOTA SU t_FUP IN BASELINE
    ──────────────────────────
    Se la colonna fup_column (es. "t_FUP") è dichiarata anche in baseline.continuous,
    viene automaticamente esclusa dalle variabili statiche continue per evitare
    duplicazione. Rimane gestita internamente come target di follow-up.
    Non è necessario modificare il JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── Time ──────────────────────────────────────────────────────────
    t = cfg.get("time", {})
    if not t:
        raise ValueError(
            "Il JSON di configurazione deve contenere il blocco 'time' con "
            "max_visits, visit_column e patient_id."
        )
    for required in ("max_visits", "visit_column", "patient_id"):
        if required not in t:
            raise ValueError(
                f"Il blocco 'time' del JSON deve contenere '{required}'."
            )

    fup_column = _get(t, "fup_column", "t_FUP")
    time_cfg = TimeConfig(
        max_visits   = int(t["max_visits"]),
        visit_column = t["visit_column"],
        patient_id   = t["patient_id"],
        fup_column   = fup_column,
        min_visits   = int(_get(t, "min_visits", 1)),
    )

    # ── Model ─────────────────────────────────────────────────────────
    if "model" not in cfg:
        raise ValueError(
            "Il JSON di configurazione deve contenere il blocco 'model'."
        )
    model_cfg = build_model_config(cfg["model"])

    # ── Preprocessing ─────────────────────────────────────────────────
    prep_cfg = build_preprocessing_config(cfg)

    # ── Variables ─────────────────────────────────────────────────────
    # La colonna fup_column (es. "t_FUP") viene esclusa automaticamente
    # da baseline.continuous se presente, perché è già gestita come
    # target di follow-up internamente. Emette un warning informativo.
    fup_col_lower = fup_column.lower()
    variables: List[VariableConfig] = []

    baseline_cont = cfg.get("baseline", {}).get("continuous", [])
    fup_in_baseline = [s["name"] for s in baseline_cont
                       if s["name"].lower() == fup_col_lower]
    if fup_in_baseline:
        warnings.warn(
            f"La colonna '{fup_in_baseline[0]}' è dichiarata sia in 'time.fup_column' "
            f"sia in 'baseline.continuous'. Verrà esclusa dalle variabili statiche "
            f"continue per evitare duplicazione — è già gestita come target di follow-up.",
            UserWarning,
            stacklevel=2,
        )

    for spec in baseline_cont:
        # Salta la colonna fup (es. t_FUP) se presente in baseline
        if spec["name"].lower() == fup_col_lower:
            continue
        variables.append(VariableConfig(
            name   = spec["name"],
            kind   = "continuous",
            static = True,
            dtype  = spec.get("type", "float"),
        ))

    for name, spec in cfg.get("baseline", {}).get("categorical", {}).items():
        if not isinstance(spec.get("mapping"), dict) or not spec["mapping"]:
            warnings.warn(
                f"Variabile categorica baseline '{name}' ha mapping vuoto o assente. "
                f"Verrà saltata.",
                UserWarning,
                stacklevel=2,
            )
            continue
        mapping = {k: int(v) for k, v in spec["mapping"].items()}
        variables.append(VariableConfig(
            name    = name,
            kind    = "categorical",
            static  = True,
            dtype   = spec.get("type", "int"),
            mapping = mapping,
        ))

    for spec in cfg.get("followup", {}).get("continuous", []):
        variables.append(VariableConfig(
            name   = spec["name"],
            kind   = "continuous",
            static = False,
            dtype  = spec.get("type", "float"),
        ))

    for name, spec in cfg.get("followup", {}).get("categorical", {}).items():
        if not isinstance(spec.get("mapping"), dict) or not spec["mapping"]:
            warnings.warn(
                f"Variabile categorica followup '{name}' ha mapping vuoto o assente. "
                f"Verrà saltata.",
                UserWarning,
                stacklevel=2,
            )
            continue
        mapping = {k: int(v) for k, v in spec["mapping"].items()}
        variables.append(VariableConfig(
            name         = name,
            kind         = "categorical",
            static       = False,
            dtype        = spec.get("type", "int"),
            mapping      = mapping,
            irreversible = bool(spec.get("irreversible", False)),
        ))

    # outcomes è solo documentativo, non usato dal modello
    # (già presente nel JSON dell'utente, viene ignorato silenziosamente)

    return time_cfg, variables, model_cfg, prep_cfg