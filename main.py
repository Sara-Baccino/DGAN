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
    df_train = pd.read_excel("PBC_UDCA_long.xlsx")
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
        "CENTRE": 16
    }
    
    preprocessor = Preprocessor(data_cfg, embedding_configs=embedding_configs)
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
    
    dgan.fit(
        tensors_dict=tensors,
        epochs=model_cfg.epochs  # 2000 nel config
    )
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    # Visualizza loss history
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        axes[0, 0].plot(dgan.loss_history['generator'])
        axes[0, 0].set_title('Generator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(dgan.loss_history['disc_static'], label='Static')
        axes[0, 1].plot(dgan.loss_history['disc_temporal'], label='Temporal')
        axes[0, 1].set_title('Discriminator Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(dgan.loss_history['irreversibility'])
        axes[1, 0].set_title('Irreversibility Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        if dgan.loss_history['epsilon']:
            axes[1, 1].plot(dgan.loss_history['epsilon'])
            axes[1, 1].set_title('Privacy Budget (ε)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150)
        print("✓ Saved training history plot: training_history.png")
    except Exception as e:
        logger.warning(f"Could not plot training history: {e}")
    
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
    
    df_synthetic = dgan.generate(
        n_samples=n_synthetic,
        temperature=0.6,  # temperatura bassa per output più discreto
        return_dataframe=True
    )
    
    print(f"Generated {len(df_synthetic)} rows, {df_synthetic[data_cfg.patient_id_col].nunique()} patients")
    
    # Verifica che NON ci siano missing nell'output
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
    df_synthetic.to_excel("output/synthetic_data.xlsx", index=False)
    print(f"✓ Saved synthetic data: output/synthetic_data.xlsx")
    
    # =========================================================================
    # 8. VALIDAZIONE BASICA
    # =========================================================================
    logger.info("\n" + "="*80)
    print("VALIDATION SUMMARY")
    logger.info("="*80)
    
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
    embedding_configs = {"CENTRE": 6}
    preprocessor = Preprocessor(data_cfg, embedding_configs=embedding_configs)
    
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
    logger.info(f"✓ Generated {len(df_new)} rows with loaded model")
    
    return df_new


if __name__ == "__main__":
    # Training pipeline completo
    main()
    
    # Test loading (opzionale)
    # test_loading()