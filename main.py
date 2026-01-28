import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from processing.processor import long_to_wide, LongitudinalDataPreprocessor
from processing.cat_encoding import encode_categoricals
from config.config import DataConfig, VariableConfig
from model.dgan import DGAN


def main():

    # ============================================================
    # 1. LOAD CONFIG
    # ============================================================
    with open("config/data_config.json", "r") as f:
        cfg = json.load(f)

    time_cfg = cfg["time"]
    base_cfg = cfg["baseline"]
    foll_cfg = cfg["followup"]
    model_cfg = cfg["model"]

    id_col = time_cfg["patient_id"]
    time_col = time_cfg["visit_column"]
    max_visits = time_cfg["max_visits"]

    static_cont = base_cfg["continuous"]
    static_cat_map = base_cfg["categorical"]
    temporal_cont = foll_cfg["continuous"]
    temporal_cat_map = foll_cfg["categorical"]

    static_cat = list(static_cat_map.keys())
    temporal_cat = list(temporal_cat_map.keys())

    full_cat_map = {}
    full_cat_map.update(static_cat_map)
    full_cat_map.update(temporal_cat_map)

    print("CONFIG LOADED")
    print(f"Static cont: {len(static_cont)}")
    print(f"Static cat : {len(static_cat)}")
    print(f"Temp cont  : {len(temporal_cont)}")
    print(f"Temp cat   : {len(temporal_cat)}")

    # ============================================================
    # 2. LOAD DATA (LONG)
    # ============================================================
    df = pd.read_excel("PBC_UDCA_long.xlsx")
    print(f"Raw data shape: {df.shape}")

    # ============================================================
    # 3. ENCODE CATEGORICALS
    # ============================================================
    df = encode_categoricals(df, full_cat_map)
    print("Categorical encoding completed")

    # ============================================================
    # 4. LONG → WIDE
    # ============================================================
    data_wide = long_to_wide(
        df=df,
        id_col=id_col,
        time_col=time_col,
        temporal_vars=temporal_cont + temporal_cat,
        static_vars=static_cont + static_cat,
        max_seq_len=max_visits
    )

    print("Converted to WIDE format")
    for k, v in data_wide.items():
        print(f"{k}: {v.shape}")

    # ============================================================
    # 5. BUILD DATACONFIG
    # ============================================================
    variables = []

    for v in static_cont:
        variables.append(
            VariableConfig(
                name=v,
                type="continuous",
                is_static=True
            )
        )

    for v, mapping in static_cat_map.items():
        variables.append(
            VariableConfig(
                name=v,
                type="categorical",
                is_static=True,
                categories=list(mapping.values())
            )
        )

    for v in temporal_cont:
        variables.append(
            VariableConfig(
                name=v,
                type="continuous",
                is_static=False
            )
        )

    for v, mapping in temporal_cat_map.items():
        variables.append(
            VariableConfig(
                name=v,
                type="categorical",
                is_static=False,
                categories=list(mapping.values())
            )
        )

    data_config = DataConfig(
        variables=variables,
        max_sequence_len=max_visits,
        visit_times_variable="visit_times"
    )

    # ============================================================
    # 6. PREPROCESSING
    # ============================================================
    preproc = LongitudinalDataPreprocessor(data_config)
    preproc.fit(data_wide)

    dataset = preproc.transform(data_wide)

    print("Preprocessing completed")
    for x in dataset:
        if x is not None:
            print(x.shape)

    # ============================================================
    # 7. TRAIN / VAL SPLIT
    # ============================================================
    N = dataset[0].shape[0]
    idx = list(range(N))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)

    def split(x):
        return (
            x[train_idx] if x is not None else None,
            x[val_idx] if x is not None else None
        )

    train_data = tuple(split(x)[0] for x in dataset)
    val_data = tuple(split(x)[1] for x in dataset)

    # ============================================================
    # 8. MODEL
    # ============================================================
    dgan = DGAN(
        data_config=data_config,
        config=model_cfg,
        device="cpu"
    )

    # ============================================================
    # 9. TRAIN
    # ============================================================
    dgan.fit(
        train_data,
        validation_data=val_data,
        verbose=True
    )

    # ============================================================
    # 10. VALIDATE
    # ============================================================
    metrics = dgan.validate(val_data)
    print("VALIDATION METRICS:", metrics)

    # ============================================================
    # 11. SAVE
    # ============================================================
    dgan.save("dgan_trained.pt")
    print("Model saved")


if __name__ == "__main__":
    main()
