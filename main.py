#main.py

import pandas as pd
import torch
import logging
from pathlib import Path
import numpy as np
import random

from config.config_loader import load_config, build_data_config
from processing.processor import Preprocessor
from model.dgan import DGAN

from datetime import datetime
#timestr = datetime.now().strftime("%Y%m%d_%H%M%S")

timestr = "3"

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

    config_path = "config/data_config.json"

    time_cfg, variables, model_cfg = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)
    
    print(f"Max visits: {data_cfg.max_len}")
    print(f"Static continuous: {data_cfg.n_static_cont}")
    print(f"Static categorical: {data_cfg.n_static_cat}")
    print(f"Temporal continuous: {data_cfg.n_temp_cont}")
    print(f"Temporal categorical: {data_cfg.n_temp_cat}")
    print(f"Irreversible indices: {data_cfg.irreversible_idx}")
    
    # =========================================================================
    # 2. CARICA DATI
    # =========================================================================
    df_train = pd.read_excel("PBC_UDCA_long_strat.xlsx")
    

    print(f"Loaded {len(df_train)} rows, {df_train[data_cfg.patient_id_col].nunique()} patients")
    
    # =========================================================================
    # 3. PREPROCESSING CON EMBEDDING
    # =========================================================================
    # VERIFICA CRITICA: __MISSING__ non deve essere nei mapping
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
    
    # Configura embedding per CENTRE (48 categorie → 6 dimensioni)
    embedding_configs = {
        "CENTRE": 12
    }
    
    LOG_VARS = ["ALP", "BIL", "ALT", "AST", "GGT", "CRE", "TRIGVAL", "NEUTVAL"]

    preprocessor = Preprocessor(data_cfg, embedding_configs=embedding_configs, log_vars=LOG_VARS)

    #preprocessor = Preprocessor(data_cfg, log_vars=LOG_VARS)
    tensors = preprocessor.fit_transform(df_train)

    # Verifica che le maschere siano coerenti con le OHE
    print("\nVerifying masks and OHE consistency...")
    for name in tensors["temporal_cat"].keys():
        ohe = tensors["temporal_cat"][name]           # [N, T, K]
        mask = tensors["temporal_cat_mask"][name]     # [N, T]
        
        # Dove mask=0, la OHE dovrebbe essere [0, 0, ..., 0]
        missing_positions = (mask == 0)
        ohe_sum = ohe.sum(dim=-1)  # [N, T] — somma su K
        
        # Se questa assertion fallisce, c'è un bug nel preprocessor
        assert (ohe_sum[missing_positions] == 0).all(), \
            f"❌ {name}: OHE non è zero dove mask=0"
        
        n_missing = missing_positions.sum().item()
        n_total = mask.numel()
        print(f"  ✓ {name}: {n_missing}/{n_total} missing ({n_missing/n_total*100:.1f}%)")
    
    print("\nPreprocessing complete:")
    logger.info(f"  - Static cont shape: {tensors['static_cont'].shape}")
    logger.info(f"  - Static cat shape: {tensors.get('static_cat', torch.empty(0)).shape}")
    logger.info(f"  - Static embed: {list(tensors.get('static_cat_embed', {}).keys())}")
    logger.info(f"  - Temporal cont shape: {tensors['temporal_cont'].shape}")
    logger.info(f"  - Temporal cat variables: {list(tensors['temporal_cat'].keys())}")
    logger.info(f"  - Visit mask shape: {tensors['visit_mask'].shape}")
    
    # =========================================================================
    # 4. INIZIALIZZA DGAN
    # =========================================================================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\nUsing device: {device}")
    
    dgan = DGAN(data_config=data_cfg, model_config=model_cfg, preprocessor=preprocessor, device=device)
    
    print(f"Model initialized:")
    print(f"  - Static dim: {dgan.static_dim}")
    print(f"  - Temporal dim: {dgan.temporal_dim}")
    print(f"  - Generator params: {sum(p.numel() for p in dgan.generator.parameters()):,}")
    print(f"  - Disc static params: {sum(p.numel() for p in dgan.disc_static.parameters()):,}")
    print(f"  - Disc temporal params: {sum(p.numel() for p in dgan.disc_temporal.parameters()):,}")
    
    # =========================================================================
    # 5. TRAINING
    # =========================================================================
    logger.info("\n" + "="*80)
    print("STARTING TRAINING")
    logger.info("="*80 + "\n")
    
    dgan.fit(tensors_dict=tensors, epochs=model_cfg.epochs)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Visualizza loss history
    from utils.plots import plot_training_history
    
    plot_training_history(dgan, timestr)
    
    # =========================================================================
    # 6. SALVA MODELLO
    # =========================================================================
    model_path = "checkpoints/dgan_final.pt"
    Path("checkpoints").mkdir(exist_ok=True)
    dgan.save(model_path)
    logger.info(f"✓ Model saved: {model_path}")
    
    # =========================================================================
    # 7. GENERA DATI SINTETICI
    # =========================================================================
    logger.info("\n" + "="*80)
    print("GENERATING SYNTHETIC DATA")
    logger.info("="*80 + "\n")
    
    n_synthetic = df_train[data_cfg.patient_id_col].nunique()  # stesso numero di pazienti
    
    df_synthetic = dgan.generate(n_samples=n_synthetic,
        temperature=0.5,  # temperatura bassa per output più discreto
        return_dataframe=True
    )
    
    print(f"Generated {len(df_synthetic)} rows, {df_synthetic[data_cfg.patient_id_col].nunique()} patients")
    
    # Verifica che NON ci siano missing nell'output
    from utils.check_data import check_missing, basic_validation

    check_missing(df_synthetic, data_cfg, timestr)
    
    logger.info("\n" + "="*80)
    print("VALIDATION SUMMARY")
    logger.info("="*80)
    
    basic_validation(df_train, df_synthetic, data_cfg)
    
    logger.info("\n" + "="*80)
    print("✓ PIPELINE COMPLETE")
    logger.info("="*80)


def test_loading():
    """Test caricamento modello salvato"""
    logger.info("\n" + "="*80)
    logger.info("TESTING MODEL LOADING")
    logger.info("="*80 + "\n")
    
    # Ricarica config
    config_path = "config/data_config.json"
    time_cfg, variables, model_cfg = load_config(config_path)
    data_cfg = build_data_config(time_cfg, variables)
    
    # Ricarica preprocessor
    #embedding_configs = {"CENTRE": 6}
    #preprocessor = Preprocessor(data_cfg, embedding_configs=embedding_configs)
    preprocessor = Preprocessor(data_cfg)
    
    # Carica modello
    dgan_loaded = DGAN.load(
        "checkpoints/dgan_final.pt",
        data_cfg,
        model_cfg,
        preprocessor,
        device="cpu"
    )
    
    # Genera nuovi dati
    df_new = dgan_loaded.generate(n_samples=100, temperature=0.6, return_dataframe=True)
    Path("output").mkdir(exist_ok=True)
    df_new.to_excel(f"output/exp_{timestr}/test_data.xlsx", index=False)
    print(f"✓ Saved synthetic data: output/test_data.xlsx")
    logger.info(f"✓ Generated {len(df_new)} rows with loaded model")
    
    return df_new


if __name__ == "__main__":
    # Training pipeline completo
    main()
    
    # Test loading (opzionale)
    # test_loading()