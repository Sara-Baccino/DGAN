# ======================================================
# eval/config.py
# Shared configuration, colors, and utility functions
# ======================================================

import os
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, CategoricalDtype

# -- Colors ------------------------------------------
COLOR_REAL  = "#3B5998"   # Blue
COLOR_SYNTH = "#FF7F50"   # Coral


# -- Variable-type helpers ----------------------------
def get_variable_types(df: pd.DataFrame, exclude: list[str]):
    """Return (numeric_cols, categorical_cols), excluding IDs/time cols."""
    num, cat = [], []
    for c in df.columns:
        if c in exclude:
            continue
        if is_numeric_dtype(df[c]):
            num.append(c)
        elif is_string_dtype(df[c]) or isinstance(df[c].dtype, CategoricalDtype):
            cat.append(c)
    return num, cat


def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def filter_valid_visits(df: pd.DataFrame, visit_mask_col: str = "VISIT_MASK") -> pd.DataFrame:
    if visit_mask_col in df.columns:
        return df[df[visit_mask_col] == 1].copy()
    return df.copy()


def align_real_to_synth_max_visits(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    time_col: str,
    patient_col: str,
) -> pd.DataFrame:
    """Trim real dataset to the same max visit count as synthetic."""
    synth_counts = synth.groupby(patient_col)[time_col].count()
    if len(synth_counts) == 0:
        return real.copy()

    max_visits = int(synth_counts.max())
    print(f"[INFO] Max visits in synthetic: {max_visits}")

    real_sorted  = real.sort_values([patient_col, time_col]).copy()
    real_aligned = (
        real_sorted
        .groupby(patient_col, group_keys=False)
        .head(max_visits)
        .copy()
    )
    print(f"[INFO] Real rows: {len(real)} -> {len(real_aligned)} after alignment")
    return real_aligned


def make_plot_dir(path: str = "plots") -> str:
    os.makedirs(path, exist_ok=True)
    return path