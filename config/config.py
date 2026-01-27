# config.py
from dataclasses import dataclass
from typing import Optional, List, Any
import json
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
import json
import numpy as np


@dataclass
class VariableConfig:
    """Configurazione per una singola variabile."""
    name: str
    type: str  # 'continuous' o 'categorical'
    is_static: bool
    categories: Optional[List[Any]] = None
    is_irreversible: bool = False  # Eventi 0->1
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass
class DataConfig:
    """Configurazione dataset con gestione tempi e time-to-event."""
    variables: List[VariableConfig]
    max_sequence_len: int
    
    # NUOVO: Gestione tempi visite
    visit_times_variable: Optional[str] = None  # Nome variabile con tempi (es: "months_from_baseline")
    max_visit_time: Optional[float] = None  # Tempo massimo (es: 60 mesi)
    
    # NUOVO: Time-to-event opzionale
    generate_time_to_event: bool = False
    time_to_event_max: Optional[float] = None
    
    def __post_init__(self):
        self.static_continuous = [v for v in self.variables if v.is_static and v.type == 'continuous']
        self.static_categorical = [v for v in self.variables if v.is_static and v.type == 'categorical']
        self.temporal_continuous = [v for v in self.variables if not v.is_static and v.type == 'continuous']
        self.temporal_categorical = [v for v in self.variables if not v.is_static and v.type == 'categorical']
        self.irreversible_vars = [v for v in self.temporal_categorical if v.is_irreversible]
    
    @classmethod
    def from_json(cls, filepath: str) -> 'DataConfig':
        with open(filepath, 'r') as f:
            data = json.load(f)
        variables = [VariableConfig(**v) for v in data['variables']]
        return cls(
            variables=variables,
            max_sequence_len=data['max_sequence_len'],
            visit_times_variable=data.get('visit_times_variable'),
            max_visit_time=data.get('max_visit_time'),
            generate_time_to_event=data.get('generate_time_to_event', False),
            time_to_event_max=data.get('time_to_event_max')
        )
    
    def to_json(self, filepath: str):
        data = {
            'variables': [vars(v) for v in self.variables],
            'max_sequence_len': self.max_sequence_len,
            'visit_times_variable': self.visit_times_variable,
            'max_visit_time': self.max_visit_time,
            'generate_time_to_event': self.generate_time_to_event,
            'time_to_event_max': self.time_to_event_max
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class DGANConfig:
    """Configurazione modello DGAN."""
    z_static_dim: int = 32
    z_temporal_dim: int = 16
    hidden_dim: int = 128
    gru_layers: int = 2
    discriminator_layers: int = 5
    discriminator_units: int = 200
    
    epochs: int = 100
    batch_size: int = 128
    lr_generator: float = 1e-4
    lr_discriminator: float = 1e-4
    beta1: float = 0.5
    
    discriminator_rounds: int = 5
    generator_rounds: int = 1
    gradient_penalty_coef: float = 10.0
    
    # Gumbel-Softmax
    gumbel_temperature_start: float = 1.0
    gumbel_temperature_end: float = 0.5
    gumbel_temperature_decay: float = 0.99
    
    # Differential Privacy con Opacus
    use_dp: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-4
    dp_max_grad_norm: float = 1.0
    dp_noise_multiplier: float = 1.0
    
    cuda: bool = True
    mixed_precision: bool = False  # Disabilitato con DP
