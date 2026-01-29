from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import json


@dataclass
class TimeConfig:
    max_visits: int
    visit_column: str
    patient_id: str

@dataclass
class VariableConfig:
    name: str
    kind: str                 # "continuous" | "categorical"
    static: bool
    dtype: str                # "int" | "float" | "string"
    mapping: Optional[Dict] = None
    irreversible: bool = False

    @property
    def n_categories(self):
        if self.mapping is None:
            return None
        return len(self.mapping)

@dataclass
class GeneratorConfig:
    gru_layers: int
    hidden_dim: int
    z_static_dim: int
    z_temporal_dim: int
    dropout: float

@dataclass
class DiscriminatorConfig:
    static_layers: int
    temporal_layers: int
    hidden_dim: int
    gru_layers: int           # numero di layer GRU temporale
    mlp_units: int            # numero di unità per layer MLP temporale
    dropout: float

@dataclass
class ModelConfig:
    z_static_dim: int
    z_temporal_dim: int
    hidden: int
    epochs: int
    batch_size: int
    lr: float
    generator: GeneratorConfig
    discriminator: DiscriminatorConfig
    noise_std: float
    critic_steps: int
    grad_clip: float
    patience: int
    use_dp: bool
    force_full_mask: bool
    regular: bool
    gumbel_temperature_start: float

@dataclass
class DataConfig:
    max_len: int
    patient_id_col: str
    time_col: str

    static_cont: List[VariableConfig]
    static_cat: List[VariableConfig]
    temporal_cont: List[VariableConfig]
    temporal_cat: List[VariableConfig]

    n_static_cont: int
    n_static_cat: List[int]
    n_temp_cont: int
    n_temp_cat: List[int]
    irreversible_idx: List[int]


def build_data_config(
    time_cfg: TimeConfig,
    variables: List[VariableConfig]
) -> DataConfig:

    static_cont = [v for v in variables if v.static and v.kind == "continuous"]
    static_cat  = [v for v in variables if v.static and v.kind == "categorical"]

    temporal_cont = [v for v in variables if not v.static and v.kind == "continuous"]
    temporal_cat  = [v for v in variables if not v.static and v.kind == "categorical"]

    irreversible_idx = [
        i for i, v in enumerate(temporal_cat) if v.irreversible
    ]

    return DataConfig(
        max_len=time_cfg.max_visits,
        patient_id_col=time_cfg.patient_id,
        time_col=time_cfg.visit_column,

        static_cont=static_cont,
        static_cat=static_cat,
        temporal_cont=temporal_cont,
        temporal_cat=temporal_cat,

        n_static_cont=len(static_cont),
        n_static_cat=[len(v.mapping) for v in static_cat],
        n_temp_cont=len(temporal_cont),
        n_temp_cat=[len(v.mapping) for v in temporal_cat],
        irreversible_idx=irreversible_idx
    )

def build_model_config(cfg: dict) -> ModelConfig:
    gen_cfg = GeneratorConfig(**cfg["generator"])
    disc_cfg = DiscriminatorConfig(**cfg["discriminator"])

    return ModelConfig(
        z_static_dim=cfg["z_static_dim"],
        z_temporal_dim=cfg["z_temporal_dim"],
        hidden=cfg["hidden"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        generator=gen_cfg,
        discriminator=disc_cfg,
        noise_std=cfg["noise_std"],
        critic_steps=cfg["critic_steps"],
        grad_clip=cfg["grad_clip"],
        patience=cfg["patience"],
        use_dp=cfg["use_dp"],
        force_full_mask=cfg["force_full_mask"],
        regular=cfg["regular"],
        gumbel_temperature_start=cfg["gumbel_temperature_start"]
    )


def load_config(path: str) -> Tuple[TimeConfig, List[VariableConfig], ModelConfig]:
    """
    Carica il file di configurazione JSON e restituisce:
    - TimeConfig
    - lista di VariableConfig
    - ModelConfig
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    # ======================================================
    # TIME CONFIG
    # ======================================================
    time_cfg = TimeConfig(
        max_visits=cfg["time"]["max_visits"],
        visit_column=cfg["time"]["visit_column"],
        patient_id=cfg["time"]["patient_id"]
    )

    # ======================================================
    # MODEL CONFIG
    # ======================================================
    model_cfg = build_model_config(cfg["model"])

    # ======================================================
    # VARIABLES
    # ======================================================
    variables: List[VariableConfig] = []

    # ---------- BASELINE CONTINUOUS ----------
    for spec in cfg.get("baseline", {}).get("continuous", []):
        variables.append(
            VariableConfig(
                name=spec["name"],
                kind="continuous",
                static=True,
                dtype=spec.get("type", "float")
            )
        )

    # ---------- BASELINE CATEGORICAL ----------
    for name, spec in cfg.get("baseline", {}).get("categorical", {}).items():
        mapping = {k: int(v) for k, v in spec["mapping"].items()}

        variables.append(
            VariableConfig(
                name=name,
                kind="categorical",
                static=True,
                dtype="int",
                mapping=mapping
            )
        )

    # ---------- FOLLOW-UP CONTINUOUS ----------
    for spec in cfg.get("followup", {}).get("continuous", []):
        variables.append(
            VariableConfig(
                name=spec["name"],
                kind="continuous",
                static=False,
                dtype=spec.get("type", "float")
            )
        )

    # ---------- FOLLOW-UP CATEGORICAL ----------
    for name, spec in cfg.get("followup", {}).get("categorical", {}).items():
        mapping = {k: int(v) for k, v in spec["mapping"].items()}

        variables.append(
            VariableConfig(
                name=name,
                kind="categorical",
                static=False,
                dtype="int",
                mapping=mapping,
                irreversible=spec.get("irreversible", False)
            )
        )

    return time_cfg, variables, model_cfg