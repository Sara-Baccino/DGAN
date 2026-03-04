"""
config/config_loader.py  [v6 — architettura configurabile GRU/LSTM/Transformer + CNN disc]
================================================================================
Modifiche rispetto alla versione precedente:

  [v6] GeneratorConfig:
    - Nuovo campo obbligatorio: arch ("gru" | "lstm" | "transformer")
    - Nuovi campi: n_layers, bidirectional
    - gru_layers mantenuto per retrocompatibilità (alias di n_layers se
      arch non è presente nel JSON vecchio)
    - n_transformer_layers, n_heads, pe_frequencies: ora opzionali con default,
      usati solo quando arch="transformer"

  [v6] TempDiscriminatorConfig:
    - Nuovo campo: arch ("cnn" | "gru"), default "cnn"
    - Campi CNN: hidden_dim, kernel_size, n_layers, dilation_base
    - Campi GRU: mlp_hidden_dim, gru_hidden_dim, gru_layers, mlp_layers
    - Tutti con default → retrocompatibile con JSON che aveva solo campi GRU
    - Con arch="cnn" i parametri GRU vengono ignorati dalla factory

  [v6] ModelConfig:
    - Aggiunto critic_steps_temporal (default = critic_steps)
    - Aggiunti lambda_fm, lambda_nv, lambda_fc, lambda_fup
      (prima usati in dgan.py via getattr, ora dichiarati esplicitamente)

  [v6] build_model_config:
    - Tutti i nuovi campi con .get() e default → compatibile con JSON v5

  RETROCOMPATIBILITÀ:
    Un JSON v5 senza arch, bidirectional, lambda_fm ecc. funziona ancora:
    ottiene arch="gru", bidirectional=True, e tutti i default sensati.
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
    kind:         str          # "continuous" | "categorical"
    static:       bool
    dtype:        str          # "int" | "float" | "string"
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
    Configurazione generatore.

    arch:          "gru" (default) | "lstm" | "transformer"
    hidden_dim:    dimensione hidden
    n_layers:      layer RNN o Transformer (alias: gru_layers per retrocompat.)
    bidirectional: solo gru/lstm — ogni step vede passato e futuro
    z_static_dim, z_temporal_dim: dimensioni spazio latente
    dropout:       dropout tra layer

    Solo arch="transformer":
      n_transformer_layers: sovrascrive n_layers per il Transformer
      n_heads:              attention heads (hidden_dim % n_heads == 0)
      pe_frequencies:       frequenze positional encoding continuo
    """
    # Obbligatori (nel JSON o con default nel build)
    hidden_dim:     int
    z_static_dim:   int
    z_temporal_dim: int
    dropout:        float

    # Architettura — tutti con default per retrocompatibilità
    arch:          str   = "gru"   # "gru" | "lstm" | "transformer"
    n_layers:      int   = 2       # layer RNN o Transformer
    bidirectional: bool  = True    # solo gru/lstm

    # Retrocompatibilità: gru_layers → alias di n_layers se arch assente
    gru_layers:    int   = 2

    # Transformer-only (ignorati da gru/lstm)
    n_transformer_layers: int = 2
    n_heads:              int = 4
    pe_frequencies:       int = 16


@dataclass
class DiscriminatorConfig:
    """Discriminatore statico (Residual MLP con Spectral Norm)."""
    static_layers:  int
    mlp_hidden_dim: int
    dropout:        float


@dataclass
class TempDiscriminatorConfig:
    """
    Discriminatore temporale configurabile.

    arch: "cnn" (default) | "gru"

    Parametri CNN (arch="cnn"):
      hidden_dim:    canali Conv1d
      kernel_size:   dimensione kernel (default 3)
      n_layers:      blocchi dilatati (default 3, receptive field = 7 con k=3)
      dilation_base: base esponenziale dilatazioni (default 2 → d=1,2,4)
      mlp_layers:    layer MLP finale

    Parametri GRU (arch="gru"):
      gru_hidden_dim, gru_layers: architettura GRU
      mlp_hidden_dim, mlp_layers: MLP finale

    dropout: condiviso tra arch
    """
    # Comune
    dropout:   float = 0.1
    arch:      str   = "cnn"     # "cnn" | "gru"
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
    Configurazione completa del modello.

    Tutti i parametri con default → retrocompatibile con JSON v5.
    I parametri senza default sono obbligatori nel JSON.
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

    # Misc training
    noise_std:                float
    critic_steps:             int
    grad_clip:                float
    patience:                 int
    use_dp:                   bool
    force_full_mask:          bool
    regular:                  bool
    gumbel_temperature_start: float
    fixed_visits:             Optional[int]

    # [v6] Critic steps asimmetrici: D_t si aggiorna meno di D_s
    # Se D_t >> D_s (es. -20 vs -0.3), abbassa critic_steps_temporal a 2.
    # Default = critic_steps (comportamento identico a versione precedente).
    critic_steps_temporal: int = -1   # -1 = usa critic_steps (impostato in __post_init__)

    # Gradient penalty separati per i due discriminatori
    lambda_gp:   float = 4.0
    lambda_gp_s: float = 4.0
    lambda_gp_t: float = 8.0

    # Irreversibilità e auxiliary
    alpha_irr:  float = 0.4
    lambda_aux: float = 0.2

    # Categorical frequency regularization
    lambda_freq_gen:   float = 0.20
    lambda_freq_disc:  float = 0.05
    freq_weight_power: float = 1.5

    # Gumbel temperature
    temperature_min: float = 0.5

    # Visit mask sharpness
    n_visits_sharpness: float = 10.0

    # [v6] Losses ausiliarie generatore
    # lambda_fm:  feature matching disc_static (0=disabilitato)
    # lambda_nv:  n_visits supervision (forza n_visits_head verso reale)
    # lambda_fc:  followup constraint (t_norm_last ≈ 1.0)
    # lambda_fup: followup_norm MSE (followup_head verso reale)
    lambda_fm:  float = 3.0
    lambda_nv:  float = 0.5
    lambda_fc:  float = 0.3
    lambda_fup: float = 0.5

    def __post_init__(self):
        # critic_steps_temporal = critic_steps se non specificato
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
    """
    Costruisce GeneratorConfig da dict JSON.

    Retrocompatibilità con JSON v5 (senza arch/n_layers/bidirectional):
      - arch: default "gru"
      - n_layers: usa n_layers se presente, poi gru_layers, poi 2
      - bidirectional: default True
      - n_transformer_layers: mantiene v5 se arch="transformer"
    """
    arch       = gen_raw.get("arch", "gru").lower()
    gru_layers = gen_raw.get("gru_layers", 2)
    n_layers   = gen_raw.get("n_layers", gru_layers)  # n_layers > gru_layers > 2

    return GeneratorConfig(
        arch          = arch,
        hidden_dim    = gen_raw["hidden_dim"],
        z_static_dim  = gen_raw["z_static_dim"],
        z_temporal_dim = gen_raw["z_temporal_dim"],
        dropout       = gen_raw["dropout"],
        n_layers      = n_layers,
        bidirectional = gen_raw.get("bidirectional", True),
        gru_layers    = gru_layers,
        # Transformer-only
        n_transformer_layers = gen_raw.get("n_transformer_layers", n_layers),
        n_heads              = gen_raw.get("n_heads", 4),
        pe_frequencies       = gen_raw.get("pe_frequencies", 16),
    )


def _build_temp_disc_config(td_raw: dict) -> TempDiscriminatorConfig:
    """
    Costruisce TempDiscriminatorConfig da dict JSON.

    Retrocompatibilità con JSON v5 (solo parametri GRU):
      - arch: default "cnn" se non specificato
      - parametri CNN hanno default → nessun errore su JSON vecchi
    """
    return TempDiscriminatorConfig(
        arch         = td_raw.get("arch", "cnn").lower(),
        dropout      = td_raw.get("dropout", 0.1),
        mlp_layers   = td_raw.get("mlp_layers", 2),
        # CNN
        hidden_dim   = td_raw.get("hidden_dim",   64),
        kernel_size  = td_raw.get("kernel_size",   3),
        n_layers     = td_raw.get("n_layers",      3),
        dilation_base = td_raw.get("dilation_base", 2),
        # GRU (retrocompatibilità)
        mlp_hidden_dim = td_raw.get("mlp_hidden_dim", 128),
        gru_hidden_dim = td_raw.get("gru_hidden_dim",  64),
        gru_layers     = td_raw.get("gru_layers",       2),
    )


def build_model_config(cfg: dict) -> ModelConfig:
    """
    Costruisce ModelConfig da dict JSON (sezione "model").
    Usa .get() con default per tutti i campi nuovi → retrocompatibile.
    """
    gen_cfg      = _build_generator_config(cfg["generator"])
    disc_cfg     = DiscriminatorConfig(**cfg["static_discriminator"])
    td_cfg       = _build_temp_disc_config(cfg["temporal_discriminator"])
    lambda_gp_d  = cfg.get("lambda_gp", 4.0)
    critic_steps = cfg["critic_steps"]

    return ModelConfig(
        # Spazio latente
        z_static_dim   = cfg["z_static_dim"],
        z_temporal_dim = cfg["z_temporal_dim"],
        hidden         = cfg["hidden"],
        # Training
        epochs     = cfg["epochs"],
        batch_size = cfg["batch_size"],
        lr_g       = cfg["lr_g"],
        lr_d_s     = cfg["lr_d_s"],
        lr_d_t     = cfg.get("lr_d_t", cfg["lr_d_s"] / 3.0),
        # Architetture
        generator              = gen_cfg,
        static_discriminator   = disc_cfg,
        temporal_discriminator = td_cfg,
        # Misc
        noise_std                = cfg["noise_std"],
        critic_steps             = critic_steps,
        critic_steps_temporal    = cfg.get("critic_steps_temporal", -1),  # -1 → __post_init__
        grad_clip                = cfg["grad_clip"],
        patience                 = cfg["patience"],
        use_dp                   = cfg["use_dp"],
        force_full_mask          = cfg.get("force_full_mask", False),
        regular                  = cfg.get("regular", True),
        gumbel_temperature_start = cfg.get("gumbel_temperature_start", 1.0),
        fixed_visits             = cfg.get("fixed_visits", None),
        # Gradient penalty
        lambda_gp   = lambda_gp_d,
        lambda_gp_s = cfg.get("lambda_gp_s", lambda_gp_d),
        lambda_gp_t = cfg.get("lambda_gp_t", lambda_gp_d * 2.0),
        # Losses
        alpha_irr          = cfg.get("alpha_irr",   0.4),
        lambda_aux         = cfg.get("lambda_aux",  0.2),
        lambda_freq_gen    = cfg.get("lambda_freq_gen",   0.20),
        lambda_freq_disc   = cfg.get("lambda_freq_disc",  0.05),
        freq_weight_power  = cfg.get("freq_weight_power", 1.5),
        temperature_min    = cfg.get("temperature_min",   0.5),
        n_visits_sharpness = cfg.get("n_visits_sharpness", 10.0),
        # [v6] nuove losses
        lambda_fm  = cfg.get("lambda_fm",  3.0),
        lambda_nv  = cfg.get("lambda_nv",  0.5),
        lambda_fc  = cfg.get("lambda_fc",  0.3),
        lambda_fup = cfg.get("lambda_fup", 0.5),
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