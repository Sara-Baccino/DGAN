import polars as pl
import json
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

class DataProcessor:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = json.load(f)

        self.scalers = {}
        self.encoders = {}

    def fit_transform(self, df: pl.DataFrame):
        df = df.clone()

        # ---- BASELINE CONTINUOUS ----
        for col in self.cfg["baseline"]["continuous"]:
            scaler = StandardScaler()
            vals = df[col].to_numpy().reshape(-1, 1)
            df = df.with_columns(
                pl.Series(col, scaler.fit_transform(vals).flatten())
            )
            self.scalers[col] = scaler

        # ---- FOLLOWUP CONTINUOUS ----
        for col in self.cfg["followup"]["continuous"]:
            obs_col = f"{col}_observed"
            df = df.with_columns([
                pl.when(pl.col(col).is_null())
                  .then(0.0)
                  .otherwise(pl.col(col))
                  .alias(col),
                pl.when(pl.col(col).is_null())
                  .then(0)
                  .otherwise(1)
                  .alias(obs_col)
            ])

            scaler = StandardScaler()
            vals = df.filter(pl.col(obs_col) == 1)[col].to_numpy().reshape(-1, 1)
            scaler.fit(vals)
            df = df.with_columns(
                pl.Series(col, scaler.transform(df[col].to_numpy().reshape(-1, 1)).flatten())
            )
            self.scalers[col] = scaler

        return df

    def inverse_transform(self, df: pl.DataFrame):
        df = df.clone()
        for col, scaler in self.scalers.items():
            df = df.with_columns(
                pl.Series(col, scaler.inverse_transform(
                    df[col].to_numpy().reshape(-1, 1)
                ).flatten())
            )
        return df

    def drop_observation_cols(self, df):
        obs_cols = [c for c in df.columns if c.endswith("_observed")]
        return df.drop(obs_cols)
