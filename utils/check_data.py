import numpy as np

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
