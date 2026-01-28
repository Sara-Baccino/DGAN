import pandas as pd
import numpy as np
from typing import Dict


def encode_categoricals(
    df: pd.DataFrame,
    categorical_map: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Mappa valori categorici originali → codici numerici
    secondo il config JSON.
    """
    df = df.copy()

    for col, mapping in categorical_map.items():
        if col not in df.columns:
            continue

        inverse = set(mapping.keys())
        observed = set(df[col].dropna().unique())

        if not observed.issubset(inverse):
            raise ValueError(
                f"Unexpected labels in {col}: {observed - inverse}"
            )

        df[col] = df[col].map(mapping).astype(float)

    return df


def decode_categoricals(
    df: pd.DataFrame,
    categorical_map: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Trasformazione inversa: codici → valori originali
    """
    df = df.copy()

    for col, mapping in categorical_map.items():
        if col not in df.columns:
            continue

        inv_map = {v: k for k, v in mapping.items()}

        df[col] = df[col].round().map(inv_map)

    return df
