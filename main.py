"""
================================================================================
MODULO 6: MAIN.PY
Esempio completo di utilizzo
================================================================================
"""

from config.config import VariableConfig, DataConfig, DGANConfig
import numpy as np
from processing.processor import LongitudinalDataPreprocessor
from model.dgan import DGAN
import matplotlib.pyplot as plt


def create_example_config():
    """Crea configurazione di esempio per dataset longitudinale."""
    
    variables = [
        # Static continuous
        VariableConfig(name='age', type='continuous', is_static=True, min_val=18, max_val=90),
        VariableConfig(name='bmi', type='continuous', is_static=True, min_val=15, max_val=50),
        
        # Static categorical
        VariableConfig(name='sex', type='categorical', is_static=True, categories=['M', 'F']),
        VariableConfig(name='ethnicity', type='categorical', is_static=True, 
                      categories=['White', 'Black', 'Hispanic', 'Asian', 'Other']),
        
        # Temporal continuous (biomarkers)
        VariableConfig(name='glucose', type='continuous', is_static=False, min_val=50, max_val=400),
        VariableConfig(name='hba1c', type='continuous', is_static=False, min_val=4, max_val=14),
        VariableConfig(name='ldl', type='continuous', is_static=False, min_val=40, max_val=300),
        VariableConfig(name='sbp', type='continuous', is_static=False, min_val=80, max_val=200),
        
        # Temporal categorical (eventi irreversibili)
        VariableConfig(name='diabetes', type='categorical', is_static=False, 
                      categories=[0, 1], is_irreversible=True),
        VariableConfig(name='cvd', type='categorical', is_static=False,
                      categories=[0, 1], is_irreversible=True),
    ]
    
    data_config = DataConfig(variables=variables, max_sequence_len=6)
    
    return data_config


def main():
    """Esempio completo di training e generation."""
    
    # === 1. CONFIGURAZIONE ===
    data_config = create_example_config()
    data_config.to_json('data_config.json')
    
    model_config = DGANConfig(
        z_static_dim=32,
        z_temporal_dim=16,
        hidden_dim=128,
        epochs=100,
        batch_size=64,
        use_dp=False,
        gumbel_temperature_start=1.0,
        gumbel_temperature_end=0.5
    )
    
    # === 2. DATI DI ESEMPIO ===
    # Simula dati reali
    n_samples = 1000
    T = data_config.max_sequence_len
    
    # Crea dict con dati raw
    raw_data = {
        'age': np.random.uniform(18, 90, n_samples),
        'bmi': np.random.uniform(15, 50, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'ethnicity': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
        'glucose': np.random.uniform(50, 400, (n_samples, T)),
        'hba1c': np.random.uniform(4, 14, (n_samples, T)),
        'ldl': np.random.uniform(40, 300, (n_samples, T)),
        'sbp': np.random.uniform(80, 200, (n_samples, T)),
        'diabetes': np.random.choice([0, 1], (n_samples, T)),
        'cvd': np.random.choice([0, 1], (n_samples, T)),
    }
    
    # Aggiungi missing values (20% casuale)
    for var_name, var_data in raw_data.items():
        if var_name not in ['sex', 'ethnicity']:  # Non mettere NaN in static categorical
            mask = np.random.random(var_data.shape) < 0.2
            raw_data[var_name][mask] = np.nan
    
    # === 3. PREPROCESSING ===
    preprocessor = LongitudinalDataPreprocessor(data_config)
    preprocessor.fit(raw_data)
    
    train_data = preprocessor.transform(raw_data)
    
    print(f"Data shapes:")
    for i, name in enumerate(['static_cont', 'static_cat', 'temporal_cont', 'temporal_cat', 'mask']):
        if train_data[i] is not None:
            print(f"  {name}: {train_data[i].shape}")
    
    # === 4. TRAINING ===
    dgan = DGAN(data_config, model_config)
    
    def progress_callback(epoch, batch, total_batches, losses):
        if batch % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch}/{total_batches}, "
                  f"G: {losses['generator']:.4f}, "
                  f"D_s: {losses['disc_static']:.4f}, "
                  f"D_t: {losses['disc_temporal']:.4f}")
    
    dgan.fit(train_data, progress_callback=progress_callback)
    
    # === 5. GENERATION ===
    print("\nGenerating synthetic data...")
    synthetic_outputs = dgan.generate(n_samples=500)
    
    # Inverse transform
    synthetic_data = preprocessor.inverse_transform(
        static_continuous=synthetic_outputs.get('static_continuous'),
        static_categorical=synthetic_outputs.get('static_categorical'),
        temporal_continuous=synthetic_outputs.get('temporal_continuous'),
        temporal_categorical=synthetic_outputs.get('temporal_categorical'),
        temporal_mask=synthetic_outputs.get('temporal_mask')
    )
    
    print("\nGenerated variables:")
    for var_name, var_data in synthetic_data.items():
        print(f"  {var_name}: {var_data.shape}, "
              f"missing: {np.isnan(var_data).sum() / var_data.size * 100:.1f}%")
    
    # === 6. SAVE/LOAD ===
    dgan.save('dgan_model.pt')
    print("\nModel saved!")
    
    dgan_loaded = DGAN.load('dgan_model.pt')
    print("Model loaded successfully!")
    
    # Test generation con modello caricato
    test_synthetic = dgan_loaded.generate(n_samples=100)
    print(f"\nTest generation: {len(test_synthetic)} outputs")
    
    # === 7. PLOT LOSSES ===
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].plot(dgan.loss_history['generator'])
    axes[0, 0].set_title('Generator Loss')
    axes[0, 0].set_xlabel('Epoch')
    
    axes[0, 1].plot(dgan.loss_history['disc_static'], label='Static')
    axes[0, 1].plot(dgan.loss_history['disc_temporal'], label='Temporal')
    axes[0, 1].set_title('Discriminator Losses')
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Epoch')
    
    axes[1, 0].plot(dgan.loss_history['gp_static'], label='Static')
    axes[1, 0].plot(dgan.loss_history['gp_temporal'], label='Temporal')
    axes[1, 0].set_title('Gradient Penalties')
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("\nTraining curves saved to training_curves.png")


if __name__ == "__main__":
    main()