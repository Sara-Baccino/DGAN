import numpy as np
import pandas as pd
import torch
from typing import List, Dict
from config.config_loader import DataConfig

MAP_MISSING = "__MISSING__"


class Preprocessor:
    def __init__(self, data_cfg: DataConfig):
        self.vars = (
            data_cfg.static_cont +
            data_cfg.static_cat +
            data_cfg.temporal_cont +
            data_cfg.temporal_cat
        )

        self.max_len = data_cfg.max_len
        self.id_col = data_cfg.patient_id_col
        self.time_col = data_cfg.time_col

        self.means_std = {}
        self.scalers_cont = {}
        self.inverse_maps = {}


    # ======================================================
    # FIT + TRANSFORM
    # ======================================================
    def fit_transform(self, df: pd.DataFrame) -> Dict:
        df = self.force_types(df)
        df, cat_masks = self.encode_categoricals(df)
        df, cont_masks = self.process_continuous(df)

        padded = self.long_to_padded(df)

        padded["value_mask_cont"] = cont_masks
        padded["value_mask_cat"] = cat_masks

        # 👉 COSTRUISCI PRIMA GLI STATICI
        static = self.build_static_tensors(padded["df_static"])
        padded.update(static)

        # 👉 POI FITTA GLI SCALER
        padded = self.fit_scalers(padded)

        return self.to_tensors(padded)


    # ======================================================
    # FORCE TYPES
    # ======================================================
    def force_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for v in self.vars:
            if v.name not in df.columns:
                continue
            if v.kind == "continuous":
                df[v.name] = pd.to_numeric(df[v.name], errors="coerce")
            else:
                df[v.name] = df[v.name].fillna(MAP_MISSING).astype(str)
        return df

    # ======================================================
    # ENCODE CATEGORICALS
    # ======================================================
    def encode_categoricals(self, df: pd.DataFrame):
        df = df.copy()
        masks = {}

        for v in self.vars:
            if v.kind != "categorical" or v.name not in df.columns:
                continue

            mapping = v.mapping
            self.inverse_maps[v.name] = {v: k for k, v in mapping.items()}

            col = df[v.name]
            masks[v.name] = (col != MAP_MISSING).astype(float).values

            df[v.name] = col.map(lambda x: mapping.get(x, mapping[MAP_MISSING])).astype(int)

        return df, masks

    # ======================================================
    # CONTINUOUS
    # ======================================================
    def process_continuous(self, df: pd.DataFrame):
        df = df.copy()
        masks = {}

        for v in self.vars:
            if v.kind != "continuous" or v.name not in df.columns:
                continue
            masks[v.name] = (~df[v.name].isna()).astype(float).values
            df[v.name] = df[v.name].fillna(0.0)

        return df, masks

    # ======================================================
    # LONG → PADDED
    # ======================================================
    def long_to_padded(self, df: pd.DataFrame):
        temporal_cont = [v.name for v in self.vars if not v.static and v.kind == "continuous"]
        temporal_cat = [v for v in self.vars if not v.static and v.kind == "categorical"]

        ids = df[self.id_col].unique()
        N, T = len(ids), self.max_len

        Xc = np.zeros((N, T, len(temporal_cont)), dtype=np.float32)
        Xcat = {v.name: np.zeros((N, T), dtype=int) for v in temporal_cat}

        visit_mask = np.zeros((N, T, 1), dtype=np.float32)
        visit_times = np.zeros((N, T), dtype=np.float32)

        static_rows = []

        for i, pid in enumerate(ids):
            sub = df[df[self.id_col] == pid].sort_values(self.time_col)
            static_rows.append(sub.iloc[0])

            L = min(len(sub), T)
            visit_mask[i, :L, 0] = 1
            visit_times[i, :L] = sub[self.time_col].values[:L]

            for j, col in enumerate(temporal_cont):
                Xc[i, :L, j] = sub[col].values[:L]

            for v in temporal_cat:
                Xcat[v.name][i, :L] = sub[v.name].values[:L]

        # tempo in [0,1] per paziente
        denom = visit_times.max(axis=1, keepdims=True)
        visit_times /= np.where(denom == 0, 1, denom)

        return {
            "temporal_cont": Xc,
            "temporal_cat": Xcat,
            "visit_mask": visit_mask,
            "visit_times": visit_times,
            "df_static": pd.DataFrame(static_rows)
        }

    # ======================================================
    # STATIC
    # ======================================================
    def build_static_tensors(self, df: pd.DataFrame):
        cont, cat = [], []

        for v in self.vars:
            if not v.static or v.name not in df.columns:
                continue

            if v.kind == "continuous":
                cont.append(df[v.name].values[:, None])
            else:
                values = sorted(v.mapping.values())
                idx_map = {val: i for i, val in enumerate(values)}
                idx = np.array([idx_map[x] for x in df[v.name].values])
                cat.append(np.eye(len(values))[idx])

        out = {}
        if cont:
            out["static_cont"] = np.concatenate(cont, axis=1)
        if cat:
            out["static_cat"] = np.concatenate(cat, axis=1)

        return out

    # ======================================================
    # FIT SCALERS
    # ======================================================
    def fit_scalers(self, padded):
        # --- static continuous ---
        static_vars = [v for v in self.vars if v.kind == "continuous" and v.static]
        for j, v in enumerate(static_vars):
            if "static_cont" not in padded:
                continue
            data = padded["static_cont"][:, j]       # numpy
            mask = data != 0

            mean = data[mask].mean()
            std = data[mask].std() + 1e-6
            data_std = (data - mean) / std

            minv, maxv = data_std.min(), data_std.max()
            data_scaled = (data_std - minv) / (maxv - minv + 1e-8)

            self.means_std[v.name] = (mean, std)
            self.scalers_cont[v.name] = (minv, maxv)

            padded["static_cont"][:, j] = data_scaled

        # --- temporal continuous ---
        temporal_vars = [v for v in self.vars if v.kind == "continuous" and not v.static]
        for j, v in enumerate(temporal_vars):
            data = padded["temporal_cont"][:, :, j]   # numpy
            mask = padded["visit_mask"][:, :, 0] == 1

            mean = data[mask].mean()
            std = data[mask].std() + 1e-6
            data_std = (data - mean) / std

            minv, maxv = data_std.min(), data_std.max()
            data_scaled = (data_std - minv) / (maxv - minv + 1e-8)

            self.means_std[v.name] = (mean, std)
            self.scalers_cont[v.name] = (minv, maxv)

            padded["temporal_cont"][:, :, j] = data_scaled

        return padded

    # ======================================================
    # TO TORCH
    # ======================================================
    def to_tensors(self, padded):
        out = {
            "temporal_cont": torch.tensor(padded["temporal_cont"], dtype=torch.float32),
            "visit_mask": torch.tensor(padded["visit_mask"], dtype=torch.float32),
            "visit_time": torch.tensor(padded["visit_times"], dtype=torch.float32),
            "temporal_cat": {}
        }

        out["temporal_cont_mask"] = out["visit_mask"].repeat(
            1, 1, out["temporal_cont"].shape[-1]
        )

        for v in [v for v in self.vars if not v.static and v.kind == "categorical"]:
            values = sorted(v.mapping.values())
            idx_map = {val: i for i, val in enumerate(values)}
            idx = np.vectorize(idx_map.get)(padded["temporal_cat"][v.name])
            out["temporal_cat"][v.name] = torch.tensor(
                np.eye(len(values))[idx], dtype=torch.float32
            )

        if "static_cont" in padded:
            out["static_cont"] = torch.tensor(padded["static_cont"], dtype=torch.float32)
        if "static_cat" in padded:
            out["static_cat"] = torch.tensor(padded["static_cat"], dtype=torch.float32)

        return out

    # ======================================================
    # INVERSE TRANSFORM
    # ======================================================
    def inverse_transform(self, synthetic: Dict, complete_followup: bool = True) -> pd.DataFrame:
        records = []

        cont_vars = [v for v in self.vars if not v.static and v.kind == "continuous"]
        cat_vars = [v for v in self.vars if not v.static and v.kind == "categorical"]

        cat_maps = {
            v.name: {i: val for i, val in enumerate(sorted(v.mapping.values()))}
            for v in cat_vars
        }

        N, T, _ = synthetic["temporal_cont"].shape

        for i in range(N):
            for t in range(T):
                if synthetic["visit_mask"][i, t, 0] == 0 and not complete_followup:
                    continue

                row = {self.id_col: f"Synth_{i}", self.time_col: t}

                for j, v in enumerate(cont_vars):
                    val = synthetic["temporal_cont"][i, t, j].item()
                    mean, std = self.means_std[v.name]
                    minv, maxv = self.scalers_cont[v.name]
                    x = ((val * (maxv - minv) + minv) * std) + mean
                    row[v.name] = x

                for v in cat_vars:
                    oh = synthetic["temporal_cat"][v.name][i, t]
                    idx = int(torch.argmax(oh))
                    row[v.name] = cat_maps[v.name][idx]

                records.append(row)

        return pd.DataFrame(records)
