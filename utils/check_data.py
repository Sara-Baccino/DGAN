import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_max_visits(df, id_col, max_visits):
    visits = df.groupby(id_col).size()
    if visits.max() > max_visits:
        raise ValueError(
            f"Found patient with {visits.max()} visits > max_visits={max_visits}"
        )

def check_time_bounds(df, time_col, max_time):
    if df[time_col].max() > max_time:
        raise ValueError(
            f"Visit time exceeds max_time ({df[time_col].max()} > {max_time})"
        )

def check_no_nan(tensors):
    for k, v in tensors.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if np.isnan(vv.numpy()).any():
                    raise ValueError(f"NaN detected in {k}/{kk}")
        else:
            if np.isnan(v.numpy()).any():
                raise ValueError(f"NaN detected in {k}")
    print("Nessun NaN.")

def check_one_hot(x):
    s = x.sum(axis=-1)
    if not (s == 1).all():
        raise ValueError("Invalid one-hot encoding detected")
    
    print("One-hot ok.")

def check_missing(df_synthetic, data_cfg, timestr):
    print("\nVerifying synthetic data has no missing values...")
    for col in df_synthetic.columns:
        if col in [data_cfg.patient_id_col, data_cfg.time_col, "Delta_t"]:
            continue
        n_missing = df_synthetic[col].isna().sum()
        if n_missing > 0:
            logger.warning(f"  ⚠ {col}: {n_missing} missing values (should be 0!)")
        else:
            print(f"  ✓ {col}: no missing values")
    
    # Salva dati sintetici
    Path("output").mkdir(exist_ok=True)
    df_synthetic.to_excel(f"output/exp_{timestr}/synthetic_data.xlsx", index=False)
    print(f"✓ Saved synthetic data: output/synthetic_data.xlsx")


def basic_validation(df_train, df_synthetic, data_cfg):
    # Confronta distribuzioni
    print("\n--- STATIC CONTINUOUS VARIABLES ---")
    static_cont_vars = [v.name for v in data_cfg.static_cont]
    for var in static_cont_vars[:5]:  # primi 5
        real_mean = df_train.groupby(data_cfg.patient_id_col)[var].first().mean()
        synth_mean = df_synthetic.groupby(data_cfg.patient_id_col)[var].first().mean()
        print(f"{var:15s} | Real: {real_mean:8.3f} | Synth: {synth_mean:8.3f}")
    
    print("\n--- STATIC CATEGORICAL VARIABLES ---")
    static_cat_vars = [v.name for v in data_cfg.static_cat]
    for var in static_cat_vars[:5]:  # primi 5
        real_dist = df_train.groupby(data_cfg.patient_id_col)[var].first().value_counts(normalize=True)
        synth_dist = df_synthetic.groupby(data_cfg.patient_id_col)[var].first().value_counts(normalize=True)
        print(f"\n{var}:")
        print(f"  Real top 3:  {dict(list(real_dist.items())[:3])}")
        print(f"  Synth top 3: {dict(list(synth_dist.items())[:3])}")
    
    print("\n--- TEMPORAL CONTINUOUS VARIABLES ---")
    temporal_cont_vars = [v.name for v in data_cfg.temporal_cont]
    for var in temporal_cont_vars[:3]:  # primi 3
        real_mean = df_train[var].mean()
        synth_mean = df_synthetic[var].mean()
        print(f"{var:15s} | Real: {real_mean:8.3f} | Synth: {synth_mean:8.3f}")
    
    print("\n--- IRREVERSIBLE VARIABLES (check monotonicity) ---")
    for idx in data_cfg.irreversible_idx:
        var = data_cfg.temporal_cat[idx]
        print(f"\n{var.name}:")
        
        # Check violations (1 → 0 transitions)
        for df, label in [(df_train, "Real"), (df_synthetic, "Synth")]:
            violations = 0
            total_transitions = 0
            
            for pid, group in df.groupby(data_cfg.patient_id_col):
                group_sorted = group.sort_values(data_cfg.time_col)
                values = group_sorted[var.name].astype(str).values
                
                for i in range(len(values) - 1):
                    # Converti in string per gestire vari formati
                    curr = str(values[i]).strip()
                    next_val = str(values[i+1]).strip()
                    
                    # Skip missing
                    if curr in ["nan", "NaN", "", "__MISSING__"]:
                        continue
                    if next_val in ["nan", "NaN", "", "__MISSING__"]:
                        continue
                    
                    # Check transizione 1→0
                    if curr == "1" and next_val == "0":
                        violations += 1
                    total_transitions += 1
            
            violation_rate = violations / total_transitions if total_transitions > 0 else 0
            print(f"  {label}: {violations}/{total_transitions} violations ({violation_rate*100:.2f}%)")
    
    print("\n--- SEQUENCE LENGTHS ---")
    real_lengths = df_train.groupby(data_cfg.patient_id_col).size()
    synth_lengths = df_synthetic.groupby(data_cfg.patient_id_col).size()
    print(f"Real  | Mean: {real_lengths.mean():.2f} | Std: {real_lengths.std():.2f} | Max: {real_lengths.max()}")
    print(f"Synth | Mean: {synth_lengths.mean():.2f} | Std: {synth_lengths.std():.2f} | Max: {synth_lengths.max()}")
    