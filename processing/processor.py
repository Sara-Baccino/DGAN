"""
================================================================================
MODULO 4: PREPROCESSING.PY
Preprocessing completo con gestione missing values
================================================================================
"""
import numpy as np
import torch
from config.config import DataConfig

class LongitudinalDataPreprocessor:
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config
        self.continuous_min = {}
        self.continuous_max = {}
        self.categorical_mappings = {}
        self.visit_times_max = None
        self.is_fitted = False

    def fit(self, data):
        for var in self.data_config.static_continuous + self.data_config.temporal_continuous:
            vals = data[var.name]
            valid = vals[~np.isnan(vals)]
            self.continuous_min[var.name] = var.min_val if var.min_val is not None else float(valid.min())
            self.continuous_max[var.name] = var.max_val if var.max_val is not None else float(valid.max())

        for var in self.data_config.static_categorical + self.data_config.temporal_categorical:
            self.categorical_mappings[var.name] = {
                cat: i for i, cat in enumerate(var.categories)
            }

        if self.data_config.visit_times_variable:
            times = data[self.data_config.visit_times_variable]
            self.visit_times_max = (
                self.data_config.max_visit_time
                if self.data_config.max_visit_time is not None
                else np.nanmax(times)
            )

        self.is_fitted = True

    def transform(self, data):
        assert self.is_fitted
        N = next(iter(data.values())).shape[0]
        T = self.data_config.max_sequence_len

        # === STATIC CONTINUOUS ===
        static_cont = []
        for var in self.data_config.static_continuous:
            v = data[var.name]
            x = (v - self.continuous_min[var.name]) / (self.continuous_max[var.name] + 1e-8)
            x[np.isnan(x)] = 0
            static_cont.append(x[:, None])
        static_cont = torch.FloatTensor(np.concatenate(static_cont, axis=1)) if static_cont else None

        # === STATIC CATEGORICAL ===
        static_cat = []
        for var in self.data_config.static_categorical:
            v = data[var.name]
            one_hot = np.zeros((N, len(var.categories)))
            for i in range(N):
                if not np.isnan(v[i]):
                    one_hot[i, self.categorical_mappings[var.name][v[i]]] = 1
                else:
                    one_hot[i] = 1 / len(var.categories)
            static_cat.append(one_hot)
        static_cat = torch.FloatTensor(np.concatenate(static_cat, axis=1)) if static_cat else None

        # === VISIT MASK ===
        visit_mask = np.zeros((N, T), dtype=float)
        for i in range(N):
            for t in range(T):
                present = False
                for var in self.data_config.temporal_continuous + self.data_config.temporal_categorical:
                    if not np.isnan(data[var.name][i, t]):
                        present = True
                visit_mask[i, t] = float(present)
        temporal_mask = torch.FloatTensor(visit_mask[:, :, None])

        # === VISIT TIMES ===
        visit_times = None
        if self.data_config.visit_times_variable:
            times = data[self.data_config.visit_times_variable]
            vt = times / (self.visit_times_max + 1e-8)
            vt[np.isnan(vt)] = 0
            visit_times = torch.FloatTensor(vt)

        # === TEMPORAL CONTINUOUS ===
        temporal_cont = []
        for var in self.data_config.temporal_continuous:
            v = data[var.name]
            x = (v - self.continuous_min[var.name]) / (self.continuous_max[var.name] + 1e-8)
            x[np.isnan(x)] = 0
            temporal_cont.append(x[:, :, None])
        temporal_cont = torch.FloatTensor(np.concatenate(temporal_cont, axis=2)) if temporal_cont else None

        # === TEMPORAL CATEGORICAL + INITIAL STATES ===
        temporal_cat = []
        initial_states = []

        for var in self.data_config.temporal_categorical:
            v = data[var.name]
            one_hot = np.zeros((N, T, len(var.categories)))

            for i in range(N):
                for t in range(T):
                    if not np.isnan(v[i, t]):
                        one_hot[i, t, self.categorical_mappings[var.name][v[i, t]]] = 1
                    else:
                        one_hot[i, t] = 1 / len(var.categories)

            temporal_cat.append(one_hot)

            if var.is_irreversible:
                init = np.zeros((N, 2))
                for i in range(N):
                    idx = np.where(~np.isnan(v[i]))[0]
                    if len(idx) > 0:
                        init[i, self.categorical_mappings[var.name][v[i, idx[0]]]] = 1
                    else:
                        init[i, 0] = 1
                initial_states.append(init)

        temporal_cat = torch.FloatTensor(np.concatenate(temporal_cat, axis=2)) if temporal_cat else None
        initial_states = torch.FloatTensor(np.stack(initial_states, axis=1)) if initial_states else None

        return (
            static_cont, static_cat,
            temporal_cont, temporal_cat,
            temporal_mask,
            visit_times,
            initial_states
        )
