"""
config/config_loader.py
================================================================================
Supporta due modalità di caricamento:

  A) Due file separati (consigliato):
       time_cfg, variables, model_cfg, prep_cfg = load_config(
           data_path  = "config_data.json",
           model_path = "config_model.json",
       )

  B) File unico legacy (retrocompatibile):
       time_cfg, variables, model_cfg, prep_cfg = load_config(
           data_path="config.json"
       )

Struttura config_data.json:
  { "time": {...}, "preprocessing": {...}, "baseline": {...}, "followup": {...} }

Struttura config_model.json:
  { "latent": {...}, "generator": {...}, "static_discriminator": {...},
    "temporal_discriminator": {...}, "training": {...}, "loss": {...} }

NOTA SU t_FUP IN BASELINE
──────────────────────────
Se fup_column è dichiarata anche in baseline.continuous viene
automaticamente esclusa dalle variabili statiche (warning emesso).
================================================================================
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ======================================================================
# HELPER
# ======================================================================

def _get(cfg, key: str, default):
    """Legge cfg[key]; restituisce default se assente o None."""
    if not isinstance(cfg, dict):
        return default
    val = cfg.get(key, default)
    return val if val is not None else default


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    mice_max_iter: int            = 10
    knn_neighbors: int            = 5
    log_vars:      List[str]      = field(default_factory=list)
    emb_vars:      Optional[Dict] = None
    clip_z:        float          = 4.0


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
class StaticBranchConfig:
    hidden_dim: int   = 128
    n_layers:   int   = 2
    activation: str   = "tanh"
    proj_dim:   int   = 16     # proiezione s_h → input GRU ogni step


@dataclass
class TemporalBranchConfig:
    hidden_dim:  int   = 64
    n_layers:    int   = 2
    activation:  str   = "leaky_relu"
    attn_heads:  int   = 2     # 0 = skip self-attention
    dropout:     float = 0.2


@dataclass
class HeadsConfig:
    """Parametri degli head di output del generatore."""
    followup_hidden_factor: int   = 2        # hidden followup_head = hidden_dim // factor
    n_visits_init_bias:     float = 4.0      # bias iniziale n_visits_head
    interval_activation:    str   = "softplus"


@dataclass
class GeneratorConfig:
    arch:     str = "gru"
    z_static_dim:   int = 64
    z_temporal_dim: int = 32
    static:   StaticBranchConfig  = field(default_factory=StaticBranchConfig)
    temporal: TemporalBranchConfig = field(default_factory=TemporalBranchConfig)
    heads:    HeadsConfig          = field(default_factory=HeadsConfig)

    # Shortcut properties per evitare refactoring immediato nel generatore
    @property
    def hidden_dim(self) -> int:
        return self.temporal.hidden_dim

    @property
    def static_hidden_dim(self) -> int:
        return self.static.hidden_dim

    @property
    def n_layers(self) -> int:
        return self.temporal.n_layers

    @property
    def static_n_layers(self) -> int:
        return self.static.n_layers

    @property
    def dropout(self) -> float:
        return self.temporal.dropout

    @property
    def attn_heads(self) -> int:
        return self.temporal.attn_heads

    @property
    def static_proj_dim(self) -> int:
        return self.static.proj_dim

    @property
    def noise_ar_rho(self) -> float:
        # Letto da TrainingConfig; qui è un placeholder per compatibilità
        return 0.0


# ======================================================================
# DISCRIMINATOR CONFIGS
# ======================================================================

@dataclass
class DiscriminatorConfig:
    """Discriminatore statico (MLP residuale)."""
    hidden_dim:   int   = 96
    n_layers:     int   = 4
    dropout:      float = 0.1

    # Alias per codice esistente che usa mlp_hidden_dim / static_layers
    @property
    def mlp_hidden_dim(self) -> int:
        return self.hidden_dim

    @property
    def static_layers(self) -> int:
        return self.n_layers


@dataclass
class CNNDiscConfig:
    hidden_dim:    int = 96
    kernel_size:   int = 3
    dilation_base: int = 2
    n_layers:      int = 3


@dataclass
class GRUDiscConfig:
    hidden_dim: int = 64
    n_layers:   int = 2


@dataclass
class MLPHeadDiscConfig:
    n_layers:   int = 3
    hidden_dim: int = 64

    # Alias
    @property
    def mlp_layers(self) -> int:
        return self.n_layers

    @property
    def mlp_hidden_dim(self) -> int:
        return self.hidden_dim


@dataclass
class TempDiscriminatorConfig:
    arch:    str              = "cnn"
    cnn:     CNNDiscConfig    = field(default_factory=CNNDiscConfig)
    gru:     GRUDiscConfig    = field(default_factory=GRUDiscConfig)
    mlp:     MLPHeadDiscConfig = field(default_factory=MLPHeadDiscConfig)
    dropout: float            = 0.1

    # Shortcut properties per compatibilità col codice esistente
    @property
    def hidden_dim(self) -> int:
        return self.cnn.hidden_dim if self.arch == "cnn" else self.gru.hidden_dim

    @property
    def kernel_size(self) -> int:
        return self.cnn.kernel_size

    @property
    def dilation_base(self) -> int:
        return self.cnn.dilation_base

    @property
    def n_layers(self) -> int:
        return self.cnn.n_layers if self.arch == "cnn" else self.gru.n_layers

    @property
    def mlp_layers(self) -> int:
        return self.mlp.n_layers

    @property
    def mlp_hidden_dim(self) -> int:
        return self.mlp.hidden_dim

    @property
    def gru_hidden_dim(self) -> int:
        return self.gru.hidden_dim

    @property
    def gru_layers(self) -> int:
        return self.gru.n_layers


# ======================================================================
# LOSS CONFIG
# ======================================================================

@dataclass
class GradientPenaltyConfig:
    lambda_static:   float = 10.0
    lambda_temporal: float = 10.0
    warmup_epochs:   int   = 0

    # Alias
    @property
    def lambda_gp_s(self) -> float:
        return self.lambda_static

    @property
    def lambda_gp_t(self) -> float:
        return self.lambda_temporal

    @property
    def gp_warmup_epochs(self) -> int:
        return self.warmup_epochs


@dataclass
class AuxLossConfig:
    lambda_aux:        float = 0.2
    alpha_irr:         float = 0.1
    lambda_fup:        float = 1.0
    lambda_nv:         float = 1.0
    lambda_static_cat: float = 2.0
    lambda_fm:         float = 0.0
    lambda_var:        float = 0.5
    lambda_interval:   float = 2.0
    lambda_delta:      float = 2.0
    lambda_autocorr:   float = 0.0
    autocorr_max_lag:  int   = 2


@dataclass
class WarmupConfig:
    lambda_scat_epochs: int = 0


@dataclass
class LossConfig:
    gradient_penalty: GradientPenaltyConfig = field(default_factory=GradientPenaltyConfig)
    auxiliary:        AuxLossConfig         = field(default_factory=AuxLossConfig)
    warmup:           WarmupConfig          = field(default_factory=WarmupConfig)

    # Shortcut properties per compatibilità con dgan.py
    @property
    def lambda_gp_s(self) -> float:
        return self.gradient_penalty.lambda_static

    @property
    def lambda_gp_t(self) -> float:
        return self.gradient_penalty.lambda_temporal

    @property
    def lambda_aux(self) -> float:
        return self.auxiliary.lambda_aux

    @property
    def alpha_irr(self) -> float:
        return self.auxiliary.alpha_irr

    @property
    def lambda_fup(self) -> float:
        return self.auxiliary.lambda_fup

    @property
    def lambda_nv(self) -> float:
        return self.auxiliary.lambda_nv

    @property
    def lambda_static_cat(self) -> float:
        return self.auxiliary.lambda_static_cat

    @property
    def lambda_fm(self) -> float:
        return self.auxiliary.lambda_fm

    @property
    def lambda_var(self) -> float:
        return self.auxiliary.lambda_var

    @property
    def lambda_interval(self) -> float:
        return self.auxiliary.lambda_interval

    @property
    def lambda_delta(self) -> float:
        return self.auxiliary.lambda_delta

    @property
    def lambda_autocorr(self) -> float:
        return self.auxiliary.lambda_autocorr

    @property
    def autocorr_max_lag(self) -> int:
        return self.auxiliary.autocorr_max_lag

    @property
    def lambda_scat_warmup_epochs(self) -> int:
        return self.warmup.lambda_scat_epochs

    @property
    def gp_warmup_epochs(self) -> int:
        return self.gradient_penalty.warmup_epochs


# ======================================================================
# TRAINING CONFIG
# ======================================================================

@dataclass
class LRConfig:
    generator:     float = 1e-4
    disc_static:   float = 1e-4
    disc_temporal: float = 5e-5

    # Alias
    @property
    def lr_g(self) -> float:
        return self.generator

    @property
    def lr_d_s(self) -> float:
        return self.disc_static

    @property
    def lr_d_t(self) -> float:
        return self.disc_temporal


@dataclass
class OptimizerConfig:
    betas:     List[float] = field(default_factory=lambda: [0.5, 0.9])
    drop_last: bool        = True

    @property
    def optimizer_beta1(self) -> float:
        return float(self.betas[0])

    @property
    def optimizer_beta2(self) -> float:
        return float(self.betas[1])


@dataclass
class CriticStepsConfig:
    static:   int = 5
    temporal: int = -1    # -1 = uguale a static

    def __post_init__(self):
        if self.temporal < 0:
            self.temporal = self.static


@dataclass
class NoiseConfig:
    std:    float = 0.1
    ar_rho: float = 0.0


@dataclass
class InstanceNoiseConfig:
    start: float = 0.05
    end:   float = 0.0


@dataclass
class GumbelConfig:
    temperature_start: float = 1.0
    temperature_min:   float = 0.5


@dataclass
class TrainingConfig:
    epochs:     int   = 200
    patience:   int   = 20
    batch_size: int   = 64
    grad_clip:  float = 1.0
    use_dp:     bool  = False
    ema_decay:  float = 0.0

    lr:             LRConfig            = field(default_factory=LRConfig)
    optimizer:      OptimizerConfig     = field(default_factory=OptimizerConfig)
    critic_steps:   CriticStepsConfig   = field(default_factory=CriticStepsConfig)
    noise:          NoiseConfig         = field(default_factory=NoiseConfig)
    instance_noise: InstanceNoiseConfig = field(default_factory=InstanceNoiseConfig)
    gumbel:         GumbelConfig        = field(default_factory=GumbelConfig)

    # Shortcut properties per compatibilità con dgan.py
    @property
    def lr_g(self) -> float:
        return self.lr.generator

    @property
    def lr_d_s(self) -> float:
        return self.lr.disc_static

    @property
    def lr_d_t(self) -> float:
        return self.lr.disc_temporal

    @property
    def optimizer_betas(self) -> List[float]:
        return self.optimizer.betas

    @property
    def optimizer_beta1(self) -> float:
        return self.optimizer.optimizer_beta1

    @property
    def optimizer_beta2(self) -> float:
        return self.optimizer.optimizer_beta2

    @property
    def dataloader_drop_last(self) -> bool:
        return self.optimizer.drop_last

    @property
    def critic_steps_static(self) -> int:
        return self.critic_steps.static

    @property
    def critic_steps_temporal(self) -> int:
        return self.critic_steps.temporal

    @property
    def noise_std(self) -> float:
        return self.noise.std

    @property
    def noise_ar_rho(self) -> float:
        return self.noise.ar_rho

    @property
    def instance_noise_start(self) -> float:
        return self.instance_noise.start

    @property
    def instance_noise_end(self) -> float:
        return self.instance_noise.end

    @property
    def gumbel_temperature_start(self) -> float:
        return self.gumbel.temperature_start

    @property
    def temperature_min(self) -> float:
        return self.gumbel.temperature_min


# ======================================================================
# MODEL CONFIG  (aggregato — letto da dgan.py)
# ======================================================================

@dataclass
class ModelConfig:
    """
    Config aggregata passata a DGAN. Raggruppa generator, discriminatori,
    training e loss. I parametri di training e loss sono accessibili
    direttamente tramite property per retrocompatibilità con dgan.py.
    """
    generator:              GeneratorConfig
    static_discriminator:   DiscriminatorConfig
    temporal_discriminator: TempDiscriminatorConfig
    training:               TrainingConfig
    loss:                   LossConfig

    # ── Shortcut diretti per dgan.py (evitano model_cfg.training.epochs ecc.) ──

    @property
    def z_static_dim(self) -> int:
        return self.generator.z_static_dim

    @property
    def z_temporal_dim(self) -> int:
        return self.generator.z_temporal_dim

    @property
    def epochs(self) -> int:
        return self.training.epochs

    @property
    def patience(self) -> int:
        return self.training.patience

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @property
    def grad_clip(self) -> float:
        return self.training.grad_clip

    @property
    def use_dp(self) -> bool:
        return self.training.use_dp

    @property
    def ema_decay(self) -> float:
        return self.training.ema_decay

    @property
    def lr_g(self) -> float:
        return self.training.lr.generator

    @property
    def lr_d_s(self) -> float:
        return self.training.lr.disc_static

    @property
    def lr_d_t(self) -> float:
        return self.training.lr.disc_temporal

    @property
    def optimizer_betas(self) -> List[float]:
        return self.training.optimizer.betas

    @property
    def optimizer_beta1(self) -> float:
        return self.training.optimizer_beta1

    @property
    def optimizer_beta2(self) -> float:
        return self.training.optimizer_beta2

    @property
    def dataloader_drop_last(self) -> bool:
        return self.training.dataloader_drop_last

    @property
    def critic_steps(self) -> int:
        return self.training.critic_steps.static

    @property
    def critic_steps_temporal(self) -> int:
        return self.training.critic_steps.temporal

    @property
    def noise_std(self) -> float:
        return self.training.noise.std

    @property
    def noise_ar_rho(self) -> float:
        return self.training.noise.ar_rho

    @property
    def instance_noise_start(self) -> float:
        return self.training.instance_noise.start

    @property
    def instance_noise_end(self) -> float:
        return self.training.instance_noise.end

    @property
    def gumbel_temperature_start(self) -> float:
        return self.training.gumbel.temperature_start

    @property
    def temperature_min(self) -> float:
        return self.training.gumbel.temperature_min

    # Loss shortcuts
    @property
    def lambda_gp_s(self) -> float:
        return self.loss.lambda_gp_s

    @property
    def lambda_gp_t(self) -> float:
        return self.loss.lambda_gp_t

    @property
    def lambda_aux(self) -> float:
        return self.loss.lambda_aux

    @property
    def alpha_irr(self) -> float:
        return self.loss.alpha_irr

    @property
    def lambda_fup(self) -> float:
        return self.loss.lambda_fup

    @property
    def lambda_nv(self) -> float:
        return self.loss.lambda_nv

    @property
    def lambda_static_cat(self) -> float:
        return self.loss.lambda_static_cat

    @property
    def lambda_fm(self) -> float:
        return self.loss.lambda_fm

    @property
    def lambda_var(self) -> float:
        return self.loss.lambda_var

    @property
    def lambda_interval(self) -> float:
        return self.loss.lambda_interval

    @property
    def lambda_delta(self) -> float:
        return self.loss.lambda_delta

    @property
    def lambda_autocorr(self) -> float:
        return self.loss.lambda_autocorr

    @property
    def autocorr_max_lag(self) -> int:
        return self.loss.autocorr_max_lag

    @property
    def lambda_scat_warmup_epochs(self) -> int:
        return self.loss.lambda_scat_warmup_epochs

    @property
    def gp_warmup_epochs(self) -> int:
        return self.loss.gp_warmup_epochs


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
# BUILD FUNCTIONS — Data
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


# ======================================================================
# BUILD FUNCTIONS — Model (nuovo formato a 2 file)
# ======================================================================

def _build_generator_config(raw: dict, latent: dict) -> GeneratorConfig:
    sb  = _get(raw, "static_branch",   {})
    tb  = _get(raw, "temporal_branch", {})
    hd  = _get(raw, "heads",           {})

    return GeneratorConfig(
        arch           = _get(raw, "arch",          "gru"),
        z_static_dim   = _get(latent, "z_static_dim",   64),
        z_temporal_dim = _get(latent, "z_temporal_dim", 32),
        static  = StaticBranchConfig(
            hidden_dim = _get(sb, "hidden_dim", 128),
            n_layers   = _get(sb, "n_layers",     2),
            activation = _get(sb, "activation", "tanh"),
            proj_dim   = _get(sb, "proj_dim",    16),
        ),
        temporal = TemporalBranchConfig(
            hidden_dim  = _get(tb, "hidden_dim",   64),
            n_layers    = _get(tb, "n_layers",      2),
            activation  = _get(tb, "activation", "leaky_relu"),
            attn_heads  = _get(tb, "attn_heads",   2),
            dropout     = _get(tb, "dropout",     0.2),
        ),
        heads = HeadsConfig(
            followup_hidden_factor = _get(hd, "followup_hidden_factor", 2),
            n_visits_init_bias     = _get(hd, "n_visits_init_bias",   4.0),
            interval_activation    = _get(hd, "interval_activation", "softplus"),
        ),
    )


def _build_static_disc_config(raw: dict) -> DiscriminatorConfig:
    return DiscriminatorConfig(
        hidden_dim = _get(raw, "hidden_dim", 96),
        n_layers   = _get(raw, "n_layers",    4),
        dropout    = _get(raw, "dropout",   0.1),
    )


def _build_temp_disc_config(raw: dict) -> TempDiscriminatorConfig:
    cnn_raw = _get(raw, "cnn",      {})
    gru_raw = _get(raw, "gru",      {})
    mlp_raw = _get(raw, "mlp_head", {})
    return TempDiscriminatorConfig(
        arch    = _get(raw, "arch",    "cnn"),
        dropout = _get(raw, "dropout", 0.1),
        cnn = CNNDiscConfig(
            hidden_dim    = _get(cnn_raw, "hidden_dim",   96),
            kernel_size   = _get(cnn_raw, "kernel_size",   3),
            dilation_base = _get(cnn_raw, "dilation_base", 2),
            n_layers      = _get(cnn_raw, "n_layers",      3),
        ),
        gru = GRUDiscConfig(
            hidden_dim = _get(gru_raw, "hidden_dim", 64),
            n_layers   = _get(gru_raw, "n_layers",    2),
        ),
        mlp = MLPHeadDiscConfig(
            n_layers   = _get(mlp_raw, "n_layers",   3),
            hidden_dim = _get(mlp_raw, "hidden_dim", 64),
        ),
    )


def _build_training_config(raw: dict) -> TrainingConfig:
    lr_raw  = _get(raw, "lr",             {})
    opt_raw = _get(raw, "optimizer",      {})
    cs_raw  = _get(raw, "critic_steps",   {})
    ns_raw  = _get(raw, "noise",          {})
    in_raw  = _get(raw, "instance_noise", {})
    gm_raw  = _get(raw, "gumbel",         {})

    cs_static   = _get(cs_raw, "static",   5)
    cs_temporal = _get(cs_raw, "temporal", -1)

    return TrainingConfig(
        epochs     = _get(raw, "epochs",     200),
        patience   = _get(raw, "patience",    20),
        batch_size = _get(raw, "batch_size",  64),
        grad_clip  = _get(raw, "grad_clip",  1.0),
        use_dp     = _get(raw, "use_dp",    False),
        ema_decay  = _get(raw, "ema_decay",  0.0),
        lr = LRConfig(
            generator     = _get(lr_raw, "generator",     1e-4),
            disc_static   = _get(lr_raw, "disc_static",   1e-4),
            disc_temporal = _get(lr_raw, "disc_temporal", 5e-5),
        ),
        optimizer = OptimizerConfig(
            betas     = _get(opt_raw, "betas",     [0.5, 0.9]),
            drop_last = _get(opt_raw, "drop_last", True),
        ),
        critic_steps = CriticStepsConfig(
            static   = cs_static,
            temporal = cs_temporal,
        ),
        noise = NoiseConfig(
            std    = _get(ns_raw, "std",    0.1),
            ar_rho = _get(ns_raw, "ar_rho", 0.0),
        ),
        instance_noise = InstanceNoiseConfig(
            start = _get(in_raw, "start", 0.05),
            end   = _get(in_raw, "end",   0.0),
        ),
        gumbel = GumbelConfig(
            temperature_start = _get(gm_raw, "temperature_start", 1.0),
            temperature_min   = _get(gm_raw, "temperature_min",   0.5),
        ),
    )


def _build_loss_config(raw: dict) -> LossConfig:
    gp_raw  = _get(raw, "gradient_penalty", {})
    aux_raw = _get(raw, "auxiliary",        {})
    wu_raw  = _get(raw, "warmup",           {})

    # Alias legacy: lambda_gp come fallback per gp_s/gp_t
    lambda_gp_fallback = _get(raw, "lambda_gp", 10.0)

    return LossConfig(
        gradient_penalty = GradientPenaltyConfig(
            lambda_static   = _get(gp_raw, "lambda_static",   lambda_gp_fallback),
            lambda_temporal = _get(gp_raw, "lambda_temporal", lambda_gp_fallback),
            warmup_epochs   = _get(gp_raw, "warmup_epochs",   0),
        ),
        auxiliary = AuxLossConfig(
            lambda_aux        = _get(aux_raw, "lambda_aux",        0.2),
            alpha_irr         = _get(aux_raw, "alpha_irr",         0.1),
            lambda_fup        = _get(aux_raw, "lambda_fup",        1.0),
            lambda_nv         = _get(aux_raw, "lambda_nv",         1.0),
            lambda_static_cat = _get(aux_raw, "lambda_static_cat", 2.0),
            lambda_fm         = _get(aux_raw, "lambda_fm",         0.0),
            lambda_var        = _get(aux_raw, "lambda_var",        0.5),
            lambda_interval   = _get(aux_raw, "lambda_interval",   2.0),
            lambda_delta      = _get(aux_raw, "lambda_delta",      2.0),
            lambda_autocorr   = _get(aux_raw, "lambda_autocorr",   0.0),
            autocorr_max_lag  = _get(aux_raw, "autocorr_max_lag",  2),
        ),
        warmup = WarmupConfig(
            lambda_scat_epochs = _get(wu_raw, "lambda_scat_epochs", 0),
        ),
    )


def build_model_config(cfg: dict) -> ModelConfig:
    """
    Costruisce ModelConfig dal blocco "model" del JSON legacy (file unico)
    o dall'intero JSON del config_model.json (due file).
    """
    latent  = _get(cfg, "latent",                  {})
    gen_raw = _get(cfg, "generator",               {})
    sd_raw  = _get(cfg, "static_discriminator",    {})
    td_raw  = _get(cfg, "temporal_discriminator",  {})
    tr_raw  = _get(cfg, "training",                {})
    ls_raw  = _get(cfg, "loss",                    {})

    # ── Retrocompatibilità con il vecchio formato flat ─────────────────
    # Se "training" è assente, tentiamo di leggere i campi dal livello root.
    if not tr_raw:
        tr_raw = cfg
    if not ls_raw:
        ls_raw = cfg
    if not latent:
        latent = cfg

    # Retrocompatibilità: z_static_dim / z_temporal_dim a livello root del gen
    if not _get(gen_raw, "static_branch", {}):
        # Vecchio formato flat: parametri del generator al primo livello
        gen_raw_compat = {
            "arch":          _get(gen_raw, "arch",          "gru"),
            "static_branch": {
                "hidden_dim": _get(gen_raw, "static_hidden_dim", 128),
                "n_layers":   _get(gen_raw, "static_n_layers",   2),
                "proj_dim":   _get(gen_raw, "static_proj_dim",   16),
            },
            "temporal_branch": {
                "hidden_dim": _get(gen_raw, "temp_hidden_dim", 64),
                "n_layers":   _get(gen_raw, "temp_n_layers",   2),
                "attn_heads": _get(gen_raw, "attn_heads",       2),
                "dropout":    _get(gen_raw, "dropout",         0.2),
            },
        }
        gen_raw = gen_raw_compat

    # Retrocompatibilità discriminatore statico flat
    if not _get(sd_raw, "hidden_dim", None):
        sd_raw = {
            "hidden_dim": _get(sd_raw, "mlp_hidden_dim", 96),
            "n_layers":   _get(sd_raw, "static_layers",   4),
            "dropout":    _get(sd_raw, "dropout",         0.1),
        }

    # Retrocompatibilità discriminatore temporale flat (senza sottoblocchi cnn/gru/mlp)
    if not _get(td_raw, "cnn", None) and not _get(td_raw, "gru", None):
        td_raw = {
            "arch":    _get(td_raw, "arch",    "cnn"),
            "dropout": _get(td_raw, "dropout", 0.1),
            "cnn": {
                "hidden_dim":    _get(td_raw, "hidden_dim",    96),
                "kernel_size":   _get(td_raw, "kernel_size",    3),
                "dilation_base": _get(td_raw, "dilation_base",  2),
                "n_layers":      _get(td_raw, "n_layers",        3),
            },
            "gru": {
                "hidden_dim": _get(td_raw, "gru_hidden_dim", 64),
                "n_layers":   _get(td_raw, "gru_layers",      2),
            },
            "mlp_head": {
                "n_layers":   _get(td_raw, "mlp_layers",    3),
                "hidden_dim": _get(td_raw, "mlp_hidden_dim", 64),
            },
        }

    # Retrocompatibilità loss flat
    if not _get(ls_raw, "gradient_penalty", None):
        gp_fallback = _get(ls_raw, "lambda_gp", 10.0)
        ls_raw = {
            "gradient_penalty": {
                "lambda_static":   _get(ls_raw, "lambda_gp_s",       gp_fallback),
                "lambda_temporal": _get(ls_raw, "lambda_gp_t",       gp_fallback),
                "warmup_epochs":   _get(ls_raw, "gp_warmup_epochs",   0),
            },
            "auxiliary": {k: _get(ls_raw, k, None) for k in [
                "lambda_aux", "alpha_irr", "lambda_fup", "lambda_nv",
                "lambda_static_cat", "lambda_fm", "lambda_var",
                "lambda_interval", "lambda_delta", "lambda_autocorr", "autocorr_max_lag",
            ]},
            "warmup": {
                "lambda_scat_epochs": _get(ls_raw, "lambda_scat_warmup_epochs", 0),
            },
        }

    # Retrocompatibilità training flat
    if not _get(tr_raw, "lr", None):
        lr_d_s = _get(tr_raw, "lr_d_s", 1e-4)
        tr_raw = {
            "epochs":     _get(tr_raw, "epochs",     200),
            "patience":   _get(tr_raw, "patience",    20),
            "batch_size": _get(tr_raw, "batch_size",  64),
            "grad_clip":  _get(tr_raw, "grad_clip",  1.0),
            "use_dp":     _get(tr_raw, "use_dp",    False),
            "ema_decay":  _get(tr_raw, "ema_decay",  0.0),
            "lr": {
                "generator":     _get(tr_raw, "lr_g",   1e-4),
                "disc_static":   lr_d_s,
                "disc_temporal": _get(tr_raw, "lr_d_t", lr_d_s),
            },
            "optimizer": {
                "betas":     _get(tr_raw, "optimizer_betas",      [0.5, 0.9]),
                "drop_last": _get(tr_raw, "dataloader_drop_last", True),
            },
            "critic_steps": {
                "static":   _get(tr_raw, "critic_steps",          5),
                "temporal": _get(tr_raw, "critic_steps_temporal", -1),
            },
            "noise": {
                "std":    _get(tr_raw, "noise_std",    0.1),
                "ar_rho": _get(tr_raw, "noise_ar_rho", 0.0),
            },
            "instance_noise": {
                "start": _get(tr_raw, "instance_noise_start", 0.05),
                "end":   _get(tr_raw, "instance_noise_end",   0.0),
            },
            "gumbel": {
                "temperature_start": _get(tr_raw, "gumbel_temperature_start", 1.0),
                "temperature_min":   _get(tr_raw, "temperature_min",          0.5),
            },
        }

    return ModelConfig(
        generator              = _build_generator_config(gen_raw, latent),
        static_discriminator   = _build_static_disc_config(sd_raw),
        temporal_discriminator = _build_temp_disc_config(td_raw),
        training               = _build_training_config(tr_raw),
        loss                   = _build_loss_config(ls_raw),
    )


# ======================================================================
# PREPROCESSING CONFIG
# ======================================================================

def build_preprocessing_config(cfg: dict) -> PreprocessingConfig:
    raw = _get(cfg, "preprocessing", {})
    if not isinstance(raw, dict):
        raw = {}
    return PreprocessingConfig(
        mice_max_iter = _get(raw, "mice_max_iter", 10),
        knn_neighbors = _get(raw, "knn_neighbors",  5),
        log_vars      = _get(raw, "log_vars",       []),
        emb_vars      = _get(raw, "emb_vars",      None),
        clip_z        = _get(raw, "clip_z",         4.0),
    )


# ======================================================================
# PARSE VARIABLES
# ======================================================================

def _parse_variables(cfg: dict, fup_column: str) -> List[VariableConfig]:
    """Estrae VariableConfig dal blocco baseline + followup del JSON dati."""
    fup_col_lower = fup_column.lower()
    variables: List[VariableConfig] = []

    baseline_cont = cfg.get("baseline", {}).get("continuous", [])
    fup_in_baseline = [s["name"] for s in baseline_cont
                       if s["name"].lower() == fup_col_lower]
    if fup_in_baseline:
        warnings.warn(
            f"La colonna '{fup_in_baseline[0]}' è dichiarata sia in 'time.fup_column' "
            f"sia in 'baseline.continuous'. Verrà esclusa dalle variabili statiche "
            f"continue — già gestita come target di follow-up.",
            UserWarning, stacklevel=3,
        )

    for spec in baseline_cont:
        if spec["name"].lower() == fup_col_lower:
            continue
        variables.append(VariableConfig(
            name=spec["name"], kind="continuous", static=True,
            dtype=spec.get("type", "float"),
        ))

    for name, spec in cfg.get("baseline", {}).get("categorical", {}).items():
        if not isinstance(spec.get("mapping"), dict) or not spec["mapping"]:
            warnings.warn(
                f"Variabile categorica baseline '{name}' ha mapping vuoto. Saltata.",
                UserWarning, stacklevel=3,
            )
            continue
        variables.append(VariableConfig(
            name=name, kind="categorical", static=True,
            dtype=spec.get("type", "int"),
            mapping={k: int(v) for k, v in spec["mapping"].items()},
        ))

    for spec in cfg.get("followup", {}).get("continuous", []):
        variables.append(VariableConfig(
            name=spec["name"], kind="continuous", static=False,
            dtype=spec.get("type", "float"),
        ))

    for name, spec in cfg.get("followup", {}).get("categorical", {}).items():
        if not isinstance(spec.get("mapping"), dict) or not spec["mapping"]:
            warnings.warn(
                f"Variabile categorica followup '{name}' ha mapping vuoto. Saltata.",
                UserWarning, stacklevel=3,
            )
            continue
        variables.append(VariableConfig(
            name=name, kind="categorical", static=False,
            dtype=spec.get("type", "int"),
            mapping={k: int(v) for k, v in spec["mapping"].items()},
            irreversible=bool(spec.get("irreversible", False)),
        ))

    return variables


# ======================================================================
# LOAD CONFIG — entry point
# ======================================================================

def load_config(
    data_path:  str,
    model_path: Optional[str] = None,
) -> Tuple[TimeConfig, List[VariableConfig], ModelConfig, PreprocessingConfig]:
    """
    Carica la configurazione.

    Modalità A — due file separati (consigliato):
        time_cfg, variables, model_cfg, prep_cfg = load_config(
            data_path  = "config_data.json",
            model_path = "config_model.json",
        )

    Modalità B — file unico legacy:
        time_cfg, variables, model_cfg, prep_cfg = load_config(
            data_path = "config.json",
        )
    """
    data_cfg_raw = _load_json(data_path)

    if model_path is not None:
        model_cfg_raw = _load_json(model_path)
    else:
        # Legacy: tutto in un file, blocco "model" contiene i parametri
        if "model" not in data_cfg_raw:
            raise ValueError(
                "File unico: deve contenere il blocco 'model'. "
                "Oppure passa model_path= separatamente."
            )
        model_cfg_raw = data_cfg_raw["model"]

    # ── Validazione blocco time ────────────────────────────────────────
    t = data_cfg_raw.get("time", {})
    if not t:
        raise ValueError("Il config dati deve contenere il blocco 'time'.")
    for required in ("max_visits", "visit_column", "patient_id"):
        if required not in t:
            raise ValueError(f"Il blocco 'time' deve contenere '{required}'.")

    fup_column = _get(t, "fup_column", "t_FUP")
    time_cfg = TimeConfig(
        max_visits   = int(t["max_visits"]),
        visit_column = t["visit_column"],
        patient_id   = t["patient_id"],
        fup_column   = fup_column,
        min_visits   = int(_get(t, "min_visits", 1)),
    )

    variables = _parse_variables(data_cfg_raw, fup_column)
    model_cfg = build_model_config(model_cfg_raw)
    prep_cfg  = build_preprocessing_config(data_cfg_raw)

    return time_cfg, variables, model_cfg, prep_cfg