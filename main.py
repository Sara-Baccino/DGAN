# main.py

import pandas as pd
import torch
import logging
from pathlib import Path
import numpy as np
import random

from config.config_loader import load_config, build_data_config
from processing.processor import Preprocessor
from model.dgan import DGAN
import torch_directml

from datetime import datetime
# timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
timestr = "6"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.autograd.set_detect_anomaly(True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # =========================================================================
    # 1. CARICA CONFIGURAZIONE
    # =========================================================================
    set_seed(42)

    data_config_path  = "config/data_config2.json"
    model_config_path = "config/model_config.json"

    time_cfg, variables, model_cfg, prep_cfg = load_config(
        data_path  = data_config_path, model_path = model_config_path,
    )

    data_cfg = build_data_config(time_cfg, variables)

    print(f"Max visits:           {data_cfg.max_len}")
    print(f"Min visits (config):  {data_cfg.min_visits}")
    print(f"t_FUP column:         {data_cfg.fup_col}")
    print(f"Static continuous:    {data_cfg.n_static_cont}")
    print(f"Static categorical:   {data_cfg.n_static_cat}")
    print(f"Temporal continuous:  {data_cfg.n_temp_cont}")
    print(f"Temporal categorical: {data_cfg.n_temp_cat}")
    print(f"Irreversible indices: {data_cfg.irreversible_idx}")
    print(f"Preprocessing: mice_max_iter={prep_cfg.mice_max_iter}  "
          f"knn_neighbors={prep_cfg.knn_neighbors}  "
          f"log_vars={prep_cfg.log_vars}  clip_z={prep_cfg.clip_z}")

    # =========================================================================
    # 2. CARICA DATI
    # =========================================================================
    df_train = pd.read_excel("PBC_Risk.xlsx")
    print(f"\nLoaded {len(df_train)} rows, "
          f"{df_train[data_cfg.patient_id_col].nunique()} patients")

    # =========================================================================
    # 3. VERIFICA MAPPING CATEGORICI
    # =========================================================================
    print("\nVerifying config mappings...")
    for v in variables:
        if v.kind == "categorical" and v.mapping:
            invalid_keys = [k for k in v.mapping.keys()
                            if k in ["__MISSING__", "nan", "NaN", ""]]
            if invalid_keys:
                raise ValueError(
                    f"❌ Variable '{v.name}' has INVALID keys in mapping: {invalid_keys}\n"
                    f"   Current mapping: {list(v.mapping.keys())}\n\n"
                    f"   FIX: Edit config.json and remove these keys from the mapping.\n"
                    f"   Missing values will be handled automatically by the preprocessor."
                )
            logger.info(f"  ✓ {v.name}: {len(v.mapping)} categories (no missing placeholder)")

    # =========================================================================
    # 4. PREPROCESSING
    # =========================================================================
    # Configura embedding per CENTRE (48 categorie → 12 dimensioni)
    embedding_configs = {"CENTRE": 8}

    # ── Tutti i parametri di preprocessing vengono ora da prep_cfg (config JSON) ──
    preprocessor = Preprocessor(
        data_cfg,
        embedding_configs = prep_cfg.emb_vars,      #embedding_configs,
        log_vars          = prep_cfg.log_vars,
        mice_max_iter     = prep_cfg.mice_max_iter,
        knn_neighbors     = prep_cfg.knn_neighbors,
        clip_z            = prep_cfg.clip_z,
    )

    tensors = preprocessor.fit_transform(df_train)
    print("Saving FITTED preprocessor with inverse_maps...")
    torch.save({
        'preprocessor': preprocessor,  # Salva TUTTO: scalers, inverse_maps, imputers fitted
        'inverse_maps': preprocessor.inverse_maps,  # Esplicito per sicurezza
        'data_cfg': data_cfg,
    }, f"processing/preprocessor_fitted.pt")
    print(f"  Saved: preprocessor_fitted.pt (inverse_maps: {preprocessor.inverse_maps.keys() if preprocessor.inverse_maps else 'None'})")

    # ── NOTA: le vecchie maschere (temporal_cat_mask, visit_mask) sono rimosse ──
    # Il codice usa ora valid_flag [N,T] bool come unico indicatore di padding.
    # Non fare più assert su temporal_cat_mask — non esiste nel nuovo preprocessor.

    print("\nPreprocessing complete:")
    logger.info(f"  - valid_flag shape:     {tensors['valid_flag'].shape}")
    logger.info(f"  - temporal_cont shape:  {tensors['temporal_cont'].shape}")
    logger.info(f"  - static_cont shape:    {tensors.get('static_cont', torch.empty(0)).shape}")
    logger.info(f"  - static_cat shape:     {tensors.get('static_cat', torch.empty(0)).shape}")
    logger.info(f"  - static embed keys:    {list(tensors.get('static_cat_embed', {}).keys())}")
    logger.info(f"  - temporal_cat keys:    {list(tensors.get('temporal_cat', {}).keys())}")
    logger.info(f"  - followup_norm shape:  {tensors['followup_norm'].shape}")
    logger.info(f"  - n_visits shape:       {tensors['n_visits'].shape}")

    # Statistiche rapide
    vf       = tensors["valid_flag"]
    nv       = vf.sum(dim=1).float()
    fn       = tensors["followup_norm"]
    print(f"\n  n_visits: min={nv.min().item():.0f}  "
          f"median={nv.median().item():.0f}  max={nv.max().item():.0f}")
    print(f"  followup_norm: min={fn.min().item():.3f}  "
          f"mean={fn.mean().item():.3f}  max={fn.max().item():.3f}")

    # =========================================================================
    # 5. INIZIALIZZA DGAN
    # =========================================================================
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"      #torch_directml.device()    if torch_directml.is_available() else "cpu"
    logger.info(f"\nUsing device: {device}")

    dgan = DGAN(
        data_config  = data_cfg,
        model_config = model_cfg,
        preprocessor = preprocessor,
        device       = device,
    )

    print(f"\nModel initialized:")
    print(f"  Static dim:          {dgan.static_dim}")
    print(f"  Temporal dim:        {dgan.temporal_dim}")
    print(f"  Generator params:    {sum(p.numel() for p in dgan.generator.parameters()):,}")
    print(f"  Disc static params:  {sum(p.numel() for p in dgan.disc_static.parameters()):,}")
    print(f"  Disc temporal params:{sum(p.numel() for p in dgan.disc_temporal.parameters()):,}")
    print(f"  min_visits enforced: {dgan.generator.min_visits}")
    #print(f"  noise_ar_rho:        {dgan.generator.noise_ar_rho}")

    # =========================================================================
    # 6. TRAINING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    dgan.fit(tensors_dict=tensors, epochs=model_cfg.epochs)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    # =========================================================================
    # 7. SALVA MODELLO
    # =========================================================================
    model_path = "checkpoints/dgan_final.pt"
    Path("checkpoints").mkdir(exist_ok=True)
    dgan.save(model_path)
    logger.info(f"✓ Model saved: {model_path}")

    # =========================================================================
    # 8. GENERA DATI SINTETICI
    # =========================================================================
    print("\n" + "=" * 80)
    print("GENERATING SYNTHETIC DATA")
    print("=" * 80 + "\n")

    #n_synthetic = df_train[data_cfg.patient_id_col].nunique()
    n_synthetic = 729
    df_synthetic = dgan.generate(
        n_samples        = n_synthetic,
        temperature      = 0.5,
        return_dataframe = True,
    )

    print(f"Generated {len(df_synthetic)} rows, "
          f"{df_synthetic[data_cfg.patient_id_col].nunique()} patients")

    # Verifica n_visits >= min_visits
    synth_nv = df_synthetic.groupby(data_cfg.patient_id_col).size()
    below_min = (synth_nv < data_cfg.min_visits).sum()
    if below_min > 0:
        logger.warning(
            f"⚠ {below_min} pazienti sintetici hanno meno di "
            f"min_visits={data_cfg.min_visits} visite."
        )
    else:
        print(f"  ✓ Tutti i pazienti hanno >= {data_cfg.min_visits} visita/e")
    print(f"  n_visits sintetici: min={synth_nv.min()}  "
          f"median={synth_nv.median():.0f}  max={synth_nv.max()}")

    # Verifica che t_FUP sia presente nell'output
    if data_cfg.fup_col in df_synthetic.columns:
        fup_synth = df_synthetic.groupby(data_cfg.patient_id_col)[data_cfg.fup_col].first()
        print(f"  t_FUP sintetico: min={fup_synth.min():.1f}  "
              f"mean={fup_synth.mean():.1f}  max={fup_synth.max():.1f}")
    else:
        logger.warning(f"⚠ Colonna {data_cfg.fup_col} non trovata nel DataFrame sintetico.")

    # =========================================================================
    # 9. VALIDAZIONE
    # =========================================================================
    from utils.check_data import check_missing, basic_validation

    check_missing(df_synthetic, data_cfg, timestr)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    basic_validation(df_train, df_synthetic, data_cfg)

    # =========================================================================
    # 10. PLOT
    # =========================================================================
    from utils.plots import plot_training_history, plot_training_history2

    plot_training_history(dgan, timestr)
    plot_training_history2(dgan, timestr, config_path=data_config_path)

    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)


def test_loading():
    """Test caricamento modello salvato"""
    print("\n" + "=" * 80)
    print("TESTING MODEL LOADING")
    print("=" * 80 + "\n")

    config_path = "config/data_config.json"
    time_cfg, variables, model_cfg, prep_cfg = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)

    embedding_configs = {"CENTRE": 12}
    preprocessor = Preprocessor(
        data_cfg,
        embedding_configs = embedding_configs,
        log_vars          = prep_cfg.log_vars,
        mice_max_iter     = prep_cfg.mice_max_iter,
        knn_neighbors     = prep_cfg.knn_neighbors,
        clip_z            = prep_cfg.clip_z,
    )

    dgan_loaded = DGAN.load(
        "checkpoints/dgan_final.pt",
        data_cfg,
        model_cfg,
        preprocessor,
        device="cpu",
    )

    df_new = dgan_loaded.generate(n_samples=100, temperature=0.6, return_dataframe=True)
    Path("output").mkdir(exist_ok=True)
    df_new.to_excel(f"output/exp_{timestr}/test_data.xlsx", index=False)
    print(f"✓ Saved synthetic data: output/exp_{timestr}/test_data.xlsx")
    logger.info(f"✓ Generated {len(df_new)} rows with loaded model")

    return df_new


if __name__ == "__main__":
    main()
    # test_loading()