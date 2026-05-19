# main_eval.py 
# Struttura della cartella eval/:
#   eval/metrics.py   — tutte le metriche (fidelity, utility, privacy)
#   eval/plots.py     — tutti i plot
#   eval/report.py    — run_validation_report() + costruzione PDF
# ======================================================

import os
import numpy as np
import pandas as pd
import torch

from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer, KNNImputer

from config.config_loader import load_config, build_data_config
from eval.report import run_validation_report


# ── Imputation helpers ────────────────────────────────────────────────────────

def apply_custom_imputation(df: pd.DataFrame,
                             num_vars: list[str],
                             cat_vars: list[str]) -> pd.DataFrame:
    """
    MICE (IterativeImputer) per variabili continue,
    KNN (KNNImputer) per variabili categoriche.
    Restituisce una copia del DataFrame imputato.
    """
    df_imputed = df.copy()

    if num_vars:
        print(f"    [impute] MICE su {len(num_vars)} variabili continue...")
        mice = IterativeImputer(random_state=42, max_iter=5, tol=1e-3)
        df_imputed[num_vars] = mice.fit_transform(df_imputed[num_vars])

    if cat_vars:
        print(f"    [impute] KNN su {len(cat_vars)} variabili categoriche...")
        temp_knn  = pd.DataFrame(index=df_imputed.index)
        str_maps: dict[str, np.ndarray] = {}

        for col in cat_vars:
            try:
                temp_knn[col] = pd.to_numeric(df_imputed[col], errors="raise").astype(float)
            except (ValueError, TypeError):
                codes, uniques = pd.factorize(df_imputed[col])
                temp_knn[col] = np.where(codes == -1, np.nan, codes.astype(float))
                str_maps[col] = uniques

        knn = KNNImputer(n_neighbors=5)
        arr = knn.fit_transform(temp_knn)
        imp = pd.DataFrame(arr, columns=cat_vars, index=df_imputed.index)

        for col in cat_vars:
            rounded = np.floor(imp[col].values + 0.5).astype(int)
            if col in str_maps:
                uniques = str_maps[col]
                df_imputed[col] = uniques[np.clip(rounded, 0, len(uniques) - 1)]
            else:
                df_imputed[col] = rounded

    return df_imputed


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    real_path:          str,
    synth_path:         str,
    config_path:        str,
    output_path:        str,
    preprocessor_path:  str | None = None,
    umap_color_vars:    list[str]  | None = None,   # variabili per UMAP colorato
    irr_vars:           list[str]  | None = None,
):
    """
    1. Carica Excel reale e sintetico.
    2. Carica config e preprocessore.
    3. Tronca + imputa il reale.
    4. Chiama run_validation_report() che produce metriche + PDF.

    Parameters
    ----------
    umap_color_vars : lista di colonne da usare come colore negli UMAP clusterizzati
                      (es. ['SEX', 'Risk_Level_Label']).  Se None vengono saltati.
    """
    os.makedirs(output_path, exist_ok=True)

    #  1. Dati 
    print("[main] Caricamento dati...")
    real_raw = pd.read_excel(real_path)
    synth    = pd.read_excel(synth_path)

    # 2. Config 
    print("[main] Caricamento config...")
    time_cfg, variables, _, prep_cfg = load_config(
        data_path  = config_path,
        model_path = "config/model_config.json",
    )
    data_cfg = build_data_config(time_cfg, variables)

    num_vars      = [v.name for v in data_cfg.static_cont  + data_cfg.temporal_cont]
    cat_vars      = [v.name for v in data_cfg.static_cat   + data_cfg.temporal_cat]
    temporal_vars = [v.name for v in data_cfg.temporal_cont]
    time_col      = data_cfg.time_col
    patient_col   = data_cfg.patient_id_col
    fup_col       = data_cfg.fup_col
    max_len       = getattr(time_cfg, "max_visits", 12)

    # 3. Preprocessore e inverse_maps 
    print("[main] Caricamento preprocessore...")
    if preprocessor_path and os.path.exists(preprocessor_path):
        processor    = torch.load(preprocessor_path, weights_only=False)["preprocessor"]
        inverse_maps = processor.inverse_maps
        print(f"    inverse_maps: {list(inverse_maps.keys())}")
    else:
        processor    = None
        inverse_maps = {}
        print("    [WARN] preprocessore non trovato — inverse_maps vuote")

    # 4. Troncamento + imputazione reale 
    print(f"[main] Troncamento reale a max {max_len} visite...")
    real = (real_raw.sort_values([patient_col, time_col]).groupby(patient_col).head(max_len).reset_index(drop=True))

    print("[main] Imputazione reale...")
    num_ok = [v for v in num_vars  if v in real.columns and v in synth.columns]
    cat_ok = [v for v in cat_vars  if v in real.columns and v in synth.columns]

    if fup_col and fup_col in synth.columns and fup_col not in num_ok:
        num_ok.append(fup_col)

    if processor is not None:
        real = processor._force_types(real)
        real = processor._impute(real)
    else:
        real = apply_custom_imputation(real, num_ok, cat_ok)

    print(f"[main] Pazienti — real: {real[patient_col].nunique()}  "
          f"synth: {synth[patient_col].nunique()}")

    # 5. Delega a run_validation_report
    results = run_validation_report(
        real           = real,
        synth          = synth,
        real_raw       = real_raw,
        num_ok         = num_ok,
        cat_ok         = cat_ok,
        temporal_vars  = [v for v in temporal_vars if v in real.columns and v in synth.columns],
        time_col       = time_col,
        patient_col    = patient_col,
        fup_col        = fup_col,
        max_len        = max_len,
        inverse_maps   = inverse_maps,
        output_path    = output_path,
        config_path    = config_path,
        real_path      = real_path,
        synth_path     = synth_path,
        umap_color_vars = umap_color_vars or [],
        irr_vars = irr_vars,
    )
    return results


if __name__ == "__main__":
    OUTPUT_PATH = "output/exp_5"
    CONFIG_PATH = "config/data_config.json"

    main(
        real_path          = "PBC_Risk2.xlsx",
        synth_path         = f"{OUTPUT_PATH}/synthetic_data.xlsx",
        config_path        = CONFIG_PATH,
        output_path        = OUTPUT_PATH,
        preprocessor_path  = f"processing/preprocessor_fitted.pt",
        umap_color_vars    = ["SEX", "Risk_Level_Label"],  # colonne per UMAP colorato
        irr_vars = ["HEPC", "ESOVAR", "ASCT", "VARB", "ENCP"],
    )