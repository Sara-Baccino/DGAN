#plots.py
import torch
import matplotlib.pyplot as plt
import numpy as np

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



