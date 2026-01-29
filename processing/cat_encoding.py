#cat_encoding.py
import pandas as pd
from typing import Dict


def encode_from_config(
    df: pd.DataFrame,
    baseline_cat: Dict,
    followup_cat: Dict
) -> pd.DataFrame:
    df = df.copy()

    full = {}

    for k, v in baseline_cat.items():
        full[k] = v

    for k, v in followup_cat.items():
        full[k] = v["mapping"]

    for col, mapping in full.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).astype(float)

    return df
