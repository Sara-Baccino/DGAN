"""
config/config_loader.py  [v6.4]
================================================================================
Modifiche rispetto a v6:

  [v6.4] ModelConfig — nuovi parametri aggiunti:
    - lambda_static_cat : supervisione distribuzione marginale categoriche statiche
                          default 5.0 (era mancante → getattr in dgan.py → None)
    - lambda_sc_var     : supervisione distribuzione continue statiche (quantili+var)
                          default 1.0 (era mancante → stessa situazione)

  [v6.4] Default allineati tra ModelConfig e dgan.py:
    I default in ModelConfig erano disallineati rispetto ai getattr di dgan.py.
    Quando il JSON non specifica un parametro, ModelConfig usava valori troppo
    bassi (es. lambda_fc=0.3 invece di 2.5). Ora i default sono:
      lambda_fup        : 3.0  (era 0.5)
      lambda_fc         : 2.5  (era 0.3)
      lambda_nv         : 3.0  (era 0.5)
      lambda_static_cat : 5.0  (era mancante)
      lambda_sc_var     : 1.0  (era mancante)
      lambda_gp_t       : 4.0  (era 8.0 — causava D_t dominante)
      alpha_irr         : 0.25 (era 0.4)

  [v6.4] _get() helper con null-safety:
    cfg.get(key, default) or default → gestisce JSON con valori null espliciti.

  [v6.4] dgan.py — allineamento getattr:
    I getattr in dgan.py ora trovano l'attributo in ModelConfig (non più mancante),
    quindi leggono il valore corretto dal JSON o il default allineato.
    I getattr restano come doppia sicurezza ma non sono più la fonte primaria.

  INVARIATO da v6:
    - GeneratorConfig: arch, n_layers, bidirectional, transformer params
    - TempDiscriminatorConfig: CNN/GRU, tutti i parametri
    - critic_steps_temporal: __post_init__ gestisce -1 → critic_steps
    - Retrocompatibilità completa con JSON v5/v6

  NOTA — visit_time vs visit_times (NON un bug):
    preprocessor.to_tensors() → "visit_time"  (tempi reali, per il discriminatore)
    Generator.forward()       → "visit_times" (tempi sintetici, per fc_loss)
    Le due chiavi sono diverse intenzionalmente. Non vanno unificate.
================================================================================
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import json


# ======================================================================
# SUB-CONFIGS
# ======================================================================

@dataclass
class TimeConfig:
    max_visits:   int
    visit_column: str
    patient_id:   str


@dataclass
class VariableConfig:
    name:         str
    kind:         str
    static:       bool
    dtype:        str
    mapping:      Optional[Dict] = None
    irreversible: bool = False

    @property
    def n_categories(self) -> Optional[int]:
        if self.mapping is None:
            return None
        return len(self.mapping)


@dataclass
class GeneratorConfig:
    """
    arch: "gru" | "lstm" | "transformer"
    """
    hidden_dim:     int
    z_static_dim:   int
    z_temporal_dim: int
    dropout:        float

    arch:          str  = "gru"
    n_layers:      int  = 2
    bidirectional: bool = True
    gru_layers:    int  = 2   # retrocompatibilità v5

    n_transformer_layers: int = 2
    n_heads:              int = 4
    pe_frequencies:       int = 16


@dataclass
class DiscriminatorConfig:
    static_layers:  int
    mlp_hidden_dim: int
    dropout:        float


@dataclass
class TempDiscriminatorConfig:
    """
    arch: "cnn" | "gru"
    """
    dropout:   float = 0.1
    arch:      str   = "cnn"
    mlp_layers: int  = 2

    # CNN
    hidden_dim:    int = 64
    kernel_size:   int = 3
    n_layers:      int = 3
    dilation_base: int = 2

    # GRU (retrocompatibilità)
    mlp_hidden_dim: int = 128
    gru_hidden_dim: int = 64
    gru_layers:     int = 2


# ======================================================================
# MODEL CONFIG
# ======================================================================

@dataclass
class ModelConfig:
    """
    [v6.4] Tutti i parametri lambda dichiarati con default allineati a dgan.py.
    """
    # Spazio latente
    z_static_dim:   int
    z_temporal_dim: int
    hidden:         int

    # Training
    epochs:     int
    batch_size: int
    lr_g:       float
    lr_d_s:     float
    lr_d_t:     float

    # Architetture
    generator:              GeneratorConfig
    static_discriminator:   DiscriminatorConfig
    temporal_discriminator: TempDiscriminatorConfig

    # Misc
    noise_std:                float
    critic_steps:             int
    grad_clip:                float
    patience:                 int
    use_dp:                   bool
    force_full_mask:          bool
    regular:                  bool
    gumbel_temperature_start: float
    fixed_visits:             Optional[int]

    # Critic steps asimmetrici (-1 → __post_init__ = critic_steps)
    critic_steps_temporal: int = -1

    # Gradient penalty
    lambda_gp:   float = 4.0
    lambda_gp_s: float = 4.0
    lambda_gp_t: float = 4.0   # [v6.4] abbassato da 8.0 → riduce D_t dominance

    # Irreversibilità e auxiliary
    alpha_irr:  float = 0.25   # [v6.4] abbassato da 0.4
    lambda_aux: float = 0.2

    # Categorical frequency
    lambda_freq_gen:   float = 0.30
    lambda_freq_disc:  float = 0.10
    freq_weight_power: float = 1.5

    # Gumbel temperature
    temperature_min: float = 0.5

    # Visit mask
    n_visits_sharpness: float = 10.0

    # ── Loss generatore [v6.4] ─────────────────────────────────────────
    # Feature matching
    lambda_fm: float = 1.5

    # Follow-up norm distribution (media + varianza + quantili)
    lambda_fup: float = 3.0   # [v6.4] era 0.5

    # Follow-up constraint (ultima visita attiva = followup_scale)
    lambda_fc: float = 2.5   # [v6.4] era 0.3

    # n_visits supervision (distribuzione media+var+quantili)
    lambda_nv: float = 3.0   # [v6.4] era 0.5

    # Static categorical marginal [v6.4] NUOVO — era mancante
    lambda_static_cat: float = 5.0

    # Static continuous distribution [v6.4] NUOVO
    lambda_sc_var: float = 1.0

    # Inter-visit interval distribution [v6.5] NUOVO
    # Supervisiona la distribuzione degli intervalli tra visite consecutive.
    lambda_ivi: float = 2.0

    lambda_coverage: float = 25

    lambda_uniformity: float = 5

    

    def __post_init__(self):
        if self.critic_steps_temporal < 0:
            self.critic_steps_temporal = self.critic_steps


# ======================================================================
# DATA CONFIG
# ======================================================================

@dataclass
class DataConfig:
    max_len:        int
    patient_id_col: str
    time_col:       str

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
        max_len        = time_cfg.max_visits,
        patient_id_col = time_cfg.patient_id,
        time_col       = time_cfg.visit_column,
        static_cont    = static_cont,
        static_cat     = static_cat,
        temporal_cont  = temporal_cont,
        temporal_cat   = temporal_cat,
        n_static_cont  = len(static_cont),
        n_static_cat   = [len(v.mapping) for v in static_cat],
        n_temp_cont    = len(temporal_cont),
        n_temp_cat     = [len(v.mapping) for v in temporal_cat],
        irreversible_idx = irreversible_idx,
    )


def _build_generator_config(gen_raw: dict) -> GeneratorConfig:
    arch       = gen_raw.get("arch", "gru").lower()
    gru_layers = gen_raw.get("gru_layers", 2)
    n_layers   = gen_raw.get("n_layers", gru_layers)

    return GeneratorConfig(
        arch           = arch,
        hidden_dim     = gen_raw["hidden_dim"],
        z_static_dim   = gen_raw["z_static_dim"],
        z_temporal_dim = gen_raw["z_temporal_dim"],
        dropout        = gen_raw["dropout"],
        n_layers       = n_layers,
        bidirectional  = gen_raw.get("bidirectional", True),
        gru_layers     = gru_layers,
        n_transformer_layers = gen_raw.get("n_transformer_layers", n_layers),
        n_heads              = gen_raw.get("n_heads", 4),
        pe_frequencies       = gen_raw.get("pe_frequencies", 16),
    )


def _build_temp_disc_config(td_raw: dict) -> TempDiscriminatorConfig:
    return TempDiscriminatorConfig(
        arch          = td_raw.get("arch", "cnn").lower(),
        dropout       = td_raw.get("dropout", 0.1),
        mlp_layers    = td_raw.get("mlp_layers", 2),
        hidden_dim    = td_raw.get("hidden_dim",    64),
        kernel_size   = td_raw.get("kernel_size",   3),
        n_layers      = td_raw.get("n_layers",      3),
        dilation_base = td_raw.get("dilation_base", 2),
        mlp_hidden_dim = td_raw.get("mlp_hidden_dim", 128),
        gru_hidden_dim = td_raw.get("gru_hidden_dim",  64),
        gru_layers     = td_raw.get("gru_layers",       2),
    )


def _get(cfg: dict, key: str, default):
    """
    Legge cfg[key] con null-safety.
    Se il valore è None/null nel JSON, restituisce default.
    Equivalente a: cfg.get(key, default) or default, ma più esplicito.
    """
    val = cfg.get(key, default)
    return val if val is not None else default


def build_model_config(cfg: dict) -> ModelConfig:
    """
    [v6.4] Costruisce ModelConfig da dict JSON.
    Tutti i parametri lambda usano _get() per null-safety.
    """
    gen_cfg      = _build_generator_config(cfg["generator"])
    disc_cfg     = DiscriminatorConfig(**cfg["static_discriminator"])
    td_cfg       = _build_temp_disc_config(cfg["temporal_discriminator"])
    lambda_gp_d  = _get(cfg, "lambda_gp", 4.0)
    critic_steps = cfg["critic_steps"]

    return ModelConfig(
        z_static_dim   = cfg["z_static_dim"],
        z_temporal_dim = cfg["z_temporal_dim"],
        hidden         = cfg["hidden"],

        epochs     = cfg["epochs"],
        batch_size = cfg["batch_size"],
        lr_g       = cfg["lr_g"],
        lr_d_s     = cfg["lr_d_s"],
        lr_d_t     = _get(cfg, "lr_d_t", cfg["lr_d_s"] / 3.0),

        generator              = gen_cfg,
        static_discriminator   = disc_cfg,
        temporal_discriminator = td_cfg,

        noise_std                = cfg["noise_std"],
        critic_steps             = critic_steps,
        critic_steps_temporal    = _get(cfg, "critic_steps_temporal", -1),
        grad_clip                = cfg["grad_clip"],
        patience                 = cfg["patience"],
        use_dp                   = cfg["use_dp"],
        force_full_mask          = _get(cfg, "force_full_mask",          False),
        regular                  = _get(cfg, "regular",                  True),
        gumbel_temperature_start = _get(cfg, "gumbel_temperature_start", 1.0),
        fixed_visits             = _get(cfg, "fixed_visits",             None),

        lambda_gp   = lambda_gp_d,
        lambda_gp_s = _get(cfg, "lambda_gp_s", lambda_gp_d),
        lambda_gp_t = _get(cfg, "lambda_gp_t", lambda_gp_d),   # [v6.4] default = gp_s

        alpha_irr  = _get(cfg, "alpha_irr",  0.25),
        lambda_aux = _get(cfg, "lambda_aux", 0.2),

        lambda_freq_gen    = _get(cfg, "lambda_freq_gen",    0.30),
        lambda_freq_disc   = _get(cfg, "lambda_freq_disc",   0.10),
        freq_weight_power  = _get(cfg, "freq_weight_power",  1.5),

        temperature_min    = _get(cfg, "temperature_min",    0.5),
        n_visits_sharpness = _get(cfg, "n_visits_sharpness", 10.0),

        # ── Loss generatore [v6.4] ──────────────────────────────────
        lambda_fm         = _get(cfg, "lambda_fm",          1.5),
        lambda_fup        = _get(cfg, "lambda_fup",         3.0),
        lambda_fc         = _get(cfg, "lambda_fc",          2.5),
        lambda_nv         = _get(cfg, "lambda_nv",          3.0),
        lambda_static_cat = _get(cfg, "lambda_static_cat",  5.0),   # [v6.4] NUOVO
        lambda_sc_var     = _get(cfg, "lambda_sc_var",      1.0),   # [v6.4] NUOVO
        lambda_ivi        = _get(cfg, "lambda_ivi",         2.0),   # [v6.5] NUOVO
        lambda_coverage   = _get(cfg, "lambda_coverage",     25),
        lambda_uniformity = _get(cfg, "lambda_uniformity",    5),
    )


def load_config(path: str) -> Tuple[TimeConfig, List[VariableConfig], ModelConfig]:
    with open(path, "r") as f:
        cfg = json.load(f)

    time_cfg  = TimeConfig(
        max_visits   = cfg["time"]["max_visits"],
        visit_column = cfg["time"]["visit_column"],
        patient_id   = cfg["time"]["patient_id"],
    )
    model_cfg = build_model_config(cfg["model"])
    variables: List[VariableConfig] = []

    for spec in cfg.get("baseline", {}).get("continuous", []):
        variables.append(VariableConfig(
            name=spec["name"], kind="continuous", static=True,
            dtype=spec.get("type", "float"),
        ))

    for name, spec in cfg.get("baseline", {}).get("categorical", {}).items():
        mapping = {k: int(v) for k, v in spec["mapping"].items()}
        variables.append(VariableConfig(
            name=name, kind="categorical", static=True,
            dtype=spec.get("type", "int"), mapping=mapping,
        ))

    for spec in cfg.get("followup", {}).get("continuous", []):
        variables.append(VariableConfig(
            name=spec["name"], kind="continuous", static=False,
            dtype=spec.get("type", "float"),
        ))

    for name, spec in cfg.get("followup", {}).get("categorical", {}).items():
        mapping = {k: int(v) for k, v in spec["mapping"].items()}
        variables.append(VariableConfig(
            name=name, kind="categorical", static=False,
            dtype=spec.get("type", "int"), mapping=mapping,
            irreversible=spec.get("irreversible", False),
        ))

    return time_cfg, variables, model_cfg