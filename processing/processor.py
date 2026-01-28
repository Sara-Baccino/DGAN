"""
================================================================================
MODULO 4: PREPROCESSING.PY
Preprocessing completo con gestione missing values
================================================================================
"""
import numpy as np
import torch
from config.config import DataConfig
import pandas as pd
from typing import Dict, List, Optional

class LongToWideConverter:
    """
    Converte dataset da formato LONG a formato WIDE per DoppelGANger.
    
    Formato LONG:
        patient_id | visit | age | sex | glucose | diabetes | months_from_baseline
        1          | 1     | 45  | M   | 120     | 0        | 0
        1          | 2     | 45  | M   | 125     | 0        | 6
        2          | 1     | 52  | F   | 140     | 1        | 0
    
    Formato WIDE (output):
        Dict con chiavi = nomi variabili
        Static: [n_patients]
        Temporal: [n_patients, max_visits]
    """
    
    def __init__(
        self,
        patient_id_col: str = 'patient_id',
        visit_col: str = 'visit',
        static_vars: List[str] = None,
        temporal_vars: List[str] = None,
        visit_time_var: Optional[str] = None
    ):
        self.patient_id_col = patient_id_col
        self.visit_col = visit_col
        self.static_vars = static_vars or []
        self.temporal_vars = temporal_vars or []
        self.visit_time_var = visit_time_var
    
    def convert(self, df_long: pd.DataFrame, max_visits: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Converte da long a wide.
        
        Args:
            df_long: DataFrame in formato long
            max_visits: numero massimo di visite (se None, usa max dal dataset)
        
        Returns:
            Dict con arrays in formato wide
        """
        
        # Identifica pazienti unici
        patient_ids = df_long[self.patient_id_col].unique()
        n_patients = len(patient_ids)
        
        # Determina max_visits
        if max_visits is None:
            max_visits = df_long.groupby(self.patient_id_col).size().max()
        
        wide_data = {}
        
        # === STATIC VARIABLES ===
        for var in self.static_vars:
            # Prendi primo valore per ogni paziente
            static_values = df_long.groupby(self.patient_id_col)[var].first().values
            wide_data[var] = static_values
        
        # === TEMPORAL VARIABLES ===
        for var in self.temporal_vars:
            temporal_array = np.full((n_patients, max_visits), np.nan)
            
            for i, patient_id in enumerate(patient_ids):
                patient_data = df_long[df_long[self.patient_id_col] == patient_id].sort_values(self.visit_col)
                values = patient_data[var].values[:max_visits]
                temporal_array[i, :len(values)] = values
            
            wide_data[var] = temporal_array
        
        # === VISIT TIMES ===
        if self.visit_time_var:
            visit_times_array = np.full((n_patients, max_visits), np.nan)
            
            for i, patient_id in enumerate(patient_ids):
                patient_data = df_long[df_long[self.patient_id_col] == patient_id].sort_values(self.visit_col)
                times = patient_data[self.visit_time_var].values[:max_visits]
                visit_times_array[i, :len(times)] = times
            
            wide_data[self.visit_time_var] = visit_times_array
        
        return wide_data
    
    def convert_back_to_long(
        self,
        wide_data: Dict[str, np.ndarray],
        patient_ids: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Converte da wide a long.
        
        Args:
            wide_data: dict con arrays wide
            patient_ids: array di patient IDs (se None, genera sequenziali)
        
        Returns:
            DataFrame in formato long
        """
        
        # Ottieni dimensioni
        first_temporal = next((v for k, v in wide_data.items() if k in self.temporal_vars), None)
        if first_temporal is not None:
            n_patients, max_visits = first_temporal.shape
        else:
            n_patients = len(next(iter(wide_data.values())))
            max_visits = 1
        
        if patient_ids is None:
            patient_ids = np.arange(n_patients)
        
        records = []
        
        for i in range(n_patients):
            for visit in range(max_visits):
                # Controlla se visita esiste (almeno una variabile non-NaN)
                visit_exists = False
                for var in self.temporal_vars:
                    if var in wide_data and not np.isnan(wide_data[var][i, visit]):
                        visit_exists = True
                        break
                
                if not visit_exists:
                    continue
                
                record = {
                    self.patient_id_col: patient_ids[i],
                    self.visit_col: visit + 1
                }
                
                # Static vars
                for var in self.static_vars:
                    if var in wide_data:
                        record[var] = wide_data[var][i]
                
                # Temporal vars
                for var in self.temporal_vars:
                    if var in wide_data:
                        record[var] = wide_data[var][i, visit]
                
                # Visit times
                if self.visit_time_var and self.visit_time_var in wide_data:
                    record[self.visit_time_var] = wide_data[self.visit_time_var][i, visit]
                
                records.append(record)
        
        return pd.DataFrame(records)


def compute_time_to_event(
    temporal_outcome: np.ndarray,
    visit_times: np.ndarray,
    max_time: float
) -> np.ndarray:
    """
    Calcola time-to-event per outcome binario irreversibile.
    
    Args:
        temporal_outcome: [n_samples, T] con valori 0/1
        visit_times: [n_samples, T] tempi visite
        max_time: tempo massimo (per censurati)
    
    Returns:
        [n_samples] time-to-event (max_time se non avviene)
    """
    n_samples, T = temporal_outcome.shape
    tte = np.full(n_samples, max_time)
    
    for i in range(n_samples):
        # Trova prima transizione a 1
        valid_indices = np.where(~np.isnan(temporal_outcome[i]))[0]
        
        for t in valid_indices:
            if temporal_outcome[i, t] == 1:
                # Trova tempo associato
                if not np.isnan(visit_times[i, t]):
                    tte[i] = visit_times[i, t]
                else:
                    # Se tempo mancante, usa posizione relativa
                    tte[i] = (t / T) * max_time
                break
    
    return tte


class LongitudinalDataPreprocessor:
    """
    Preprocessing robusto con:
    1. Padding per visite incomplete (missing data in visit)
    2. Padding per visite mancanti (missing visit)
    3. Gestione corretta visit times
    4. Stati iniziali per irreversibili
    5. Normalizzazione [0,1] per compatibilità Wasserstein

    - Visita VALIDA se almeno UNA variabile presente
    - Visita MANCANTE se TUTTE le variabili assenti
    """
    
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.continuous_min = {}
        self.continuous_max = {}
        self.categorical_mappings = {}
        self.visit_times_min = None
        self.visit_times_max = None
        self.is_fitted = False
    
    def fit(self, data: Dict[str, np.ndarray]):
        """Fit preprocessor."""
        
        # Continuous: min/max
        for var in (self.data_config.static_continuous + self.data_config.temporal_continuous):
            values = data[var.name]
            valid = values[~np.isnan(values)]
            
            if len(valid) > 0:
                if var.min_val is not None and var.max_val is not None:
                    self.continuous_min[var.name] = var.min_val
                    self.continuous_max[var.name] = var.max_val
                else:
                    self.continuous_min[var.name] = float(np.min(valid))
                    self.continuous_max[var.name] = float(np.max(valid))
            else:
                self.continuous_min[var.name] = 0.0
                self.continuous_max[var.name] = 1.0
        
        # Categorical: mappings
        for var in (self.data_config.static_categorical + self.data_config.temporal_categorical):
            if var.categories is not None:
                self.categorical_mappings[var.name] = {
                    cat: idx for idx, cat in enumerate(var.categories)
                }
            else:
                values = data[var.name]
                unique = np.unique(values[~np.isnan(values)])
                self.categorical_mappings[var.name] = {
                    val: idx for idx, val in enumerate(unique)
                }
                var.categories = unique.tolist()
        
        # Visit times
        if self.data_config.visit_times_variable:
            visit_times_data = data[self.data_config.visit_times_variable]
            valid_times = visit_times_data[~np.isnan(visit_times_data)]
            
            if self.data_config.max_visit_time is not None:
                self.visit_times_max = self.data_config.max_visit_time
            else:
                self.visit_times_max = float(np.max(valid_times)) if len(valid_times) > 0 else 1.0
            
            self.visit_times_min = 0.0
        
        self.is_fitted = True
    
    def transform(self, data: Dict[str, np.ndarray]) -> tuple:
        """
        Transform con logica CORRETTA per visite mancanti:
        Visita mancante = TUTTE le variabili temporali sono NaN
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit before transform")
        
        n_samples = next(iter(data.values())).shape[0]
        T = self.data_config.max_sequence_len
        
        # === STATIC CONTINUOUS ===
        static_cont = None
        if len(self.data_config.static_continuous) > 0:
            static_cont_list = []
            for var in self.data_config.static_continuous:
                values = data[var.name]
                min_v = self.continuous_min[var.name]
                max_v = self.continuous_max[var.name]
                
                normalized = (values - min_v) / (max_v - min_v + 1e-8)
                normalized = np.clip(normalized, 0, 1)
                normalized[np.isnan(values)] = 0
                
                static_cont_list.append(normalized[:, np.newaxis])
            static_cont = torch.FloatTensor(np.concatenate(static_cont_list, axis=1))
        
        # === STATIC CATEGORICAL ===
        static_cat = None
        if len(self.data_config.static_categorical) > 0:
            static_cat_list = []
            for var in self.data_config.static_categorical:
                values = data[var.name]
                n_cats = len(var.categories)
                one_hot = np.zeros((n_samples, n_cats))
                
                for i in range(n_samples):
                    if not np.isnan(values[i]):
                        idx = self.categorical_mappings[var.name][values[i]]
                        one_hot[i, idx] = 1
                    else:
                        one_hot[i, :] = 1.0 / n_cats
                
                static_cat_list.append(one_hot)
            static_cat = torch.FloatTensor(np.concatenate(static_cat_list, axis=1))
        
        # === TEMPORAL CONTINUOUS ===
        temporal_cont = None
        if len(self.data_config.temporal_continuous) > 0:
            temporal_cont_list = []
            for var in self.data_config.temporal_continuous:
                values = data[var.name]
                min_v = self.continuous_min[var.name]
                max_v = self.continuous_max[var.name]
                
                normalized = (values - min_v) / (max_v - min_v + 1e-8)
                normalized = np.clip(normalized, 0, 1)
                normalized[np.isnan(values)] = 0
                
                temporal_cont_list.append(normalized[:, :, np.newaxis])
            
            temporal_cont = torch.FloatTensor(np.concatenate(temporal_cont_list, axis=2))
        
        # === TEMPORAL CATEGORICAL + INITIAL STATES ===
        temporal_cat = None
        initial_states = None
        
        if len(self.data_config.temporal_categorical) > 0:
            temporal_cat_list = []
            initial_states_list = []
            
            for var in self.data_config.temporal_categorical:
                values = data[var.name]
                n_cats = len(var.categories)
                one_hot = np.zeros((n_samples, T, n_cats))
                
                # Initial states per irreversibili
                if var.is_irreversible:
                    initial_state = np.zeros((n_samples, 2))
                    for i in range(n_samples):
                        valid_indices = np.where(~np.isnan(values[i]))[0]
                        if len(valid_indices) > 0:
                            first_val = values[i, valid_indices[0]]
                            cat_idx = self.categorical_mappings[var.name][first_val]
                            initial_state[i, cat_idx] = 1
                        else:
                            initial_state[i, 0] = 1
                    initial_states_list.append(initial_state)
                
                # One-hot
                for i in range(n_samples):
                    for t in range(T):
                        if not np.isnan(values[i, t]):
                            idx = self.categorical_mappings[var.name][values[i, t]]
                            one_hot[i, t, idx] = 1
                        else:
                            one_hot[i, t, :] = 1.0 / n_cats
                
                temporal_cat_list.append(one_hot)
            
            temporal_cat = torch.FloatTensor(np.concatenate(temporal_cat_list, axis=2))
            
            if initial_states_list:
                initial_states = torch.FloatTensor(np.stack(initial_states_list, axis=1))
        
        # === MASK: Visita VALIDA se ALMENO UNA variabile presente ===
        temporal_mask = np.zeros((n_samples, T), dtype=bool)
        
        # Colleziona tutte le variabili temporali
        all_temporal_vars = []
        for var in self.data_config.temporal_continuous + self.data_config.temporal_categorical:
            all_temporal_vars.append(data[var.name])
        
        # Visita valida = almeno una variabile NON-NaN
        for i in range(n_samples):
            for t in range(T):
                has_any_value = False
                for var_data in all_temporal_vars:
                    if not np.isnan(var_data[i, t]):
                        has_any_value = True
                        break
                temporal_mask[i, t] = has_any_value
        
        temporal_mask = torch.FloatTensor(temporal_mask[:, :, np.newaxis])
        
        # === VISIT TIMES ===
        visit_times = None
        if self.data_config.visit_times_variable:
            times = data[self.data_config.visit_times_variable]
            normalized_times = times / (self.visit_times_max + 1e-8)
            normalized_times = np.clip(normalized_times, 0, 1)
            
            # Interpolazione per missing
            for i in range(n_samples):
                valid_indices = np.where(~np.isnan(times[i]))[0]
                if len(valid_indices) > 0:
                    for t in range(T):
                        if np.isnan(times[i, t]):
                            before = valid_indices[valid_indices < t]
                            after = valid_indices[valid_indices > t]
                            
                            if len(before) > 0 and len(after) > 0:
                                t_before = before[-1]
                                t_after = after[0]
                                weight = (t - t_before) / (t_after - t_before)
                                normalized_times[i, t] = (
                                    (1 - weight) * normalized_times[i, t_before] +
                                    weight * normalized_times[i, t_after]
                                )
                            elif len(before) > 0:
                                normalized_times[i, t] = normalized_times[i, before[-1]]
                            elif len(after) > 0:
                                normalized_times[i, t] = normalized_times[i, after[0]]
                else:
                    normalized_times[i] = np.linspace(0, 1, T)
            
            visit_times = torch.FloatTensor(normalized_times)
        
        return static_cont, static_cat, temporal_cont, temporal_cat, temporal_mask, visit_times, initial_states
    
    def inverse_transform(
        self,
        static_continuous: Optional[np.ndarray],
        static_categorical: Optional[np.ndarray],
        temporal_continuous: Optional[np.ndarray],
        temporal_categorical: Optional[np.ndarray],
        temporal_mask: Optional[np.ndarray],
        visit_times: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Inverse transform."""
        result = {}
        
        # STATIC CONTINUOUS
        if static_continuous is not None:
            offset = 0
            for var in self.data_config.static_continuous:
                normalized = static_continuous[:, offset]
                min_v = self.continuous_min[var.name]
                max_v = self.continuous_max[var.name]
                original = normalized * (max_v - min_v) + min_v
                result[var.name] = original
                offset += 1
        
        # STATIC CATEGORICAL
        if static_categorical is not None:
            offset = 0
            for var in self.data_config.static_categorical:
                n_cats = len(var.categories)
                one_hot = static_categorical[:, offset:offset+n_cats]
                indices = np.argmax(one_hot, axis=1)
                
                reverse = {idx: cat for cat, idx in self.categorical_mappings[var.name].items()}
                result[var.name] = np.array([reverse[i] for i in indices])
                offset += n_cats
        
        # TEMPORAL CONTINUOUS
        if temporal_continuous is not None:
            offset = 0
            for var in self.data_config.temporal_continuous:
                normalized = temporal_continuous[:, :, offset]
                min_v = self.continuous_min[var.name]
                max_v = self.continuous_max[var.name]
                original = normalized * (max_v - min_v) + min_v
                
                # NaN solo dove mask=0 (visita completamente mancante)
                if temporal_mask is not None:
                    original[temporal_mask[:, :, 0] == 0] = np.nan
                
                result[var.name] = original
                offset += 1
        
        # TEMPORAL CATEGORICAL
        if temporal_categorical is not None:
            offset = 0
            for var in self.data_config.temporal_categorical:
                n_cats = len(var.categories)
                one_hot = temporal_categorical[:, :, offset:offset+n_cats]
                indices = np.argmax(one_hot, axis=2)
                
                reverse = {idx: cat for cat, idx in self.categorical_mappings[var.name].items()}
                original = np.array([[reverse[i] for i in sample] for sample in indices])
                
                if temporal_mask is not None:
                    original[temporal_mask[:, :, 0] == 0] = np.nan
                
                result[var.name] = original
                offset += n_cats
        
        # VISIT TIMES
        if visit_times is not None and self.data_config.visit_times_variable:
            original_times = visit_times * self.visit_times_max
            result[self.data_config.visit_times_variable] = original_times
        
        return result

