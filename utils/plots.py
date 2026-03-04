#plots.py
import torch
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_loss_history(loss_history_path, save_path=None):
    """
    Plotta l'andamento delle loss e epsilon durante il training.

    Args:
        loss_history_path (str): path al file .pt o .pth salvato dal DGAN.
        save_path (str, optional): se specificato, salva il plot in PNG.
    """
    # Carica lo stato salvato del modello
    state = torch.load(loss_history_path, map_location="cpu")
    loss_history = state.get("loss_history", None)
    if loss_history is None:
        raise ValueError("loss_history non trovato nel file salvato")

    plt.figure(figsize=(12, 8))

    # --- Generator Loss ---
    if "generator" in loss_history:
        plt.plot(loss_history["generator"], label="Generator Loss", color="blue")
    
    # --- Discriminator Static ---
    if "disc_static" in loss_history:
        plt.plot(loss_history["disc_static"], label="Disc Static Loss", color="orange")
    
    # --- Discriminator Temporal ---
    if "disc_temporal" in loss_history:
        plt.plot(loss_history["disc_temporal"], label="Disc Temporal Loss", color="green")
    
    # --- Irreversibility Loss ---
    if "irreversibility" in loss_history:
        plt.plot(loss_history["irreversibility"], label="Irreversibility Loss", color="red")
    
    # --- DP epsilon ---
    if "epsilon" in loss_history and any(loss_history["epsilon"]):
        plt.plot(loss_history["epsilon"], label="DP Epsilon", color="purple", linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Epsilon")
    plt.title("Training Losses and DP Epsilon over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot salvato in {save_path}")
    else:
        plt.show()


def plot_training_history(dgan, timestr):
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
        plt.savefig(f'output/exp_{timestr}/training_history.png', dpi=150)
        print("✓ Saved training history plot: training_history.png")
    except Exception as e:
        logger.warning(f"Could not plot training history: {e}")
    
