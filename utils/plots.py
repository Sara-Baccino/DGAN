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

"""
utils/plots.py
================================================================================
Plot della storia di training DGAN.

  plot_training_history(dgan, timestr, config_path=None)
    → output/exp_{timestr}/plot_history.png   (loss curves, pannelli multipli)
    → output/exp_{timestr}/plot_config.png    (riepilogo parametri modello)

  Metriche monitorate:
    WGAN:       G, D_s, D_t
    Temporali:  Fup (followup_head), Fc (constraint ultima visita), NvL, Fgen
    Statiche:   Scat (categorical marginal), Fm (feature matching)
    Regime:     alpha_irr, temperatura, Aux, Irr

  Temperatura inference:
    0.5 (default in main.py) è il valore raccomandato per dati discreti.
    Temperature più basse (0.3) → output più deterministico, meno diversità.
    Temperature più alte (0.8-1.0) → più variabilità, rischio collapse.
================================================================================
"""

import os
import json
import textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


# ── palette colori coerente ───────────────────────────────────────────────────
C = {
    "G":      "#E63946",   # rosso — generatore
    "D_s":    "#457B9D",   # blu — disc statico
    "D_t":    "#1D3557",   # blu scuro — disc temporale
    "Fup":    "#2A9D8F",   # verde — followup
    "Fc":     "#E9C46A",   # giallo — constraint temporale
    "NvL":    "#F4A261",   # arancio — n_visits
    "Fgen":   "#264653",   # verde scuro — freq categoriche temporali
    "Scat":   "#A8DADC",   # celeste — categoriche statiche
    "Fm":     "#6A4C93",   # viola — feature matching
    "Irr":    "#B5838D",   # rosa — irreversibilità
    "alpha":  "#CCCCCC",   # grigio — alpha_irr
    "T":      "#888888",   # grigio scuro — temperatura
    "Aux":    "#95D5B2",   # verde chiaro — aux embed
}


def _smooth(values, w=5):
    """Media mobile semplice per leggibilità."""
    if len(values) < w * 2:
        return values
    kernel = np.ones(w) / w
    padded = np.pad(values, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _get(history, key, default=None):
    """Recupera una serie dalla history, restituisce None se assente o vuota."""
    v = history.get(key, default)
    if v is None or len(v) == 0:
        return None
    return np.array(v, dtype=float)


def plot_training_history2(dgan, timestr: str, config_path: str = None):
    """
    Genera due PNG nella cartella output/exp_{timestr}/:
      - plot_history.png : curve di training per tutte le loss
      - plot_config.png  : riepilogo parametri del modello
    """
    out_dir = Path(f"output/exp_{timestr}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = dgan.loss_history
    epochs = list(range(1, len(h.get("generator", [])) + 1))
    if not epochs:
        print("[plots] Nessuna storia di training disponibile.")
        return

    E = np.array(epochs)

    # ── 1. LOSS CURVES ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 18), facecolor="#0F0F0F")
    fig.suptitle(
        f"DGAN Training History  —  exp {timestr}  "
        f"({len(epochs)} epochs, {datetime.now().strftime('%Y-%m-%d')})",
        fontsize=14, color="white", y=0.98, fontweight="bold",
    )

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.42, wspace=0.30,
        left=0.05, right=0.97, top=0.94, bottom=0.05,
    )

    def _ax(row, col, title, ylabel=None):
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#1A1A1A")
        ax.set_title(title, color="white", fontsize=9, pad=4)
        ax.tick_params(colors="#AAAAAA", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.set_xlabel("Epoch", color="#777777", fontsize=7)
        if ylabel:
            ax.set_ylabel(ylabel, color="#777777", fontsize=7)
        ax.grid(True, color="#222222", linewidth=0.5)
        return ax

    def _plot(ax, key, label, color, lw=1.5, alpha_raw=0.25, smooth=True):
        vals = _get(h, key)
        if vals is None:
            return
        ax.plot(E, vals, color=color, alpha=alpha_raw, linewidth=0.8)
        if smooth:
            ax.plot(E, _smooth(vals), color=color, linewidth=lw, label=label)
        else:
            ax.plot(E, vals, color=color, linewidth=lw, label=label)

    # ── Riga 0 ──────────────────────────────────────────────────────────────

    # [0,0] WGAN scores
    ax = _ax(0, 0, "WGAN Scores", "score")
    _plot(ax, "generator",     "G",    C["G"])
    _plot(ax, "disc_static",   "D_s",  C["D_s"])
    _plot(ax, "disc_temporal", "D_t",  C["D_t"])
    ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,1] D_t / D_s ratio
    ax = _ax(0, 1, "|D_t| / |D_s|  (target ≈ 1)", "ratio")
    ds = _get(h, "disc_static")
    dt = _get(h, "disc_temporal")
    if ds is not None and dt is not None:
        ratio = np.abs(dt) / (np.abs(ds) + 1e-8)
        ax.plot(E, ratio, color="#FF9F1C", alpha=0.3, linewidth=0.8)
        ax.plot(E, _smooth(ratio), color="#FF9F1C", linewidth=1.5, label="|D_t|/|D_s|")
        ax.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="--", label="target=1")
        ax.axhline(2.5, color="#E63946", linewidth=0.6, linestyle=":", label="warn=2.5")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,2] Gradient penalty
    ax = _ax(0, 2, "Gradient Penalty", "GP")
    _plot(ax, "gp_static",   "GP_s", C["D_s"])
    _plot(ax, "gp_temporal", "GP_t", C["D_t"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,3] Irreversibility + alpha_irr
    ax = _ax(0, 3, "Irreversibility  (hazard entropy)", "entropy")
    _plot(ax, "irreversibility", "Irr", C["Irr"])
    ax.axhline(0.693, color=C["Irr"], linewidth=0.5, linestyle=":",
               alpha=0.5, label="ln2 (uniform)")
    alpha_v = _get(h, "alpha_irr")
    if alpha_v is not None:
        ax2 = ax.twinx()
        ax2.set_facecolor("#1A1A1A")
        ax2.plot(E, alpha_v, color=C["alpha"], linewidth=1.2, label="α_irr", alpha=0.7)
        ax2.tick_params(colors="#AAAAAA", labelsize=7)
        ax2.set_ylabel("alpha_irr", color="#777777", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7, facecolor="#111111",
                   labelcolor="white", framealpha=0.7)
    ax.legend(loc="upper left", fontsize=7, facecolor="#111111",
              labelcolor="white", framealpha=0.7)

    # ── Riga 1 ──────────────────────────────────────────────────────────────

    # [1,0] Followup + constraint
    ax = _ax(1, 0, "Follow-up Losses", "loss")
    _plot(ax, "fup_loss", "Fup (MSE followup_head)", C["Fup"])
    _plot(ax, "fc_loss",  "Fc  (t_last → fup_scale)", C["Fc"])
    ax.axhline(0.03, color=C["Fup"], linewidth=0.5, linestyle=":", alpha=0.5,
               label="Fup target≈0.03")
    ax.axhline(0.02, color=C["Fc"],  linewidth=0.5, linestyle=":", alpha=0.5,
               label="Fc  target≈0.02")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [1,1] n_visits supervision + mean
    ax = _ax(1, 1, "Visit Count", "NvL")
    _plot(ax, "nv_loss", "NvL (supervision)", C["NvL"])
    nv = _get(h, "mean_n_visits")
    if nv is not None:
        ax2 = ax.twinx()
        ax2.set_facecolor("#1A1A1A")
        ax2.plot(E, nv, color="#FFE66D", alpha=0.35, linewidth=0.8)
        ax2.plot(E, _smooth(nv), color="#FFE66D", linewidth=1.5, label="Nv mean")
        ax2.tick_params(colors="#AAAAAA", labelsize=7)
        ax2.set_ylabel("mean n_visits", color="#777777", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7, facecolor="#111111",
                   labelcolor="white", framealpha=0.7)
    ax.legend(loc="upper left", fontsize=7, facecolor="#111111",
              labelcolor="white", framealpha=0.7)

    # [1,2] Static categorical marginal
    ax = _ax(1, 2, "Static Categorical Marginal  (ScatL)", "loss")
    _plot(ax, "static_cat_loss", "ScatL", C["Scat"])
    ax.axhline(0.1, color=C["Scat"], linewidth=0.6, linestyle=":",
               alpha=0.7, label="target < 0.1")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [1,3] Feature matching
    ax = _ax(1, 3, "Feature Matching  (Fm)", "loss")
    _plot(ax, "fm_loss", "Fm", C["Fm"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # ── Riga 2 ──────────────────────────────────────────────────────────────

    # [2,0] Categorical frequency generator
    ax = _ax(2, 0, "Temporal Categorical Frequency  (Fgen)", "loss")
    _plot(ax, "freq_gen",  "Fgen", C["Fgen"])
    _plot(ax, "freq_disc", "Fdisc", C["D_t"], alpha_raw=0.15)
    ax.axhline(1.0, color="#888888", linewidth=0.6, linestyle="--", label="target<1")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,1] Aux embed
    ax = _ax(2, 1, "Auxiliary Embedding Loss  (CENTRE)", "CE")
    _plot(ax, "aux_embed", "Aux", C["Aux"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,2] Training regime: temperatura
    ax = _ax(2, 2, "Gumbel Temperature", "T")
    T_start = float(getattr(getattr(dgan, "model_config", None),
                            "gumbel_temperature_start", 1.0))
    T_min   = float(getattr(getattr(dgan, "model_config", None),
                            "temperature_min", 0.5))
    T_decay = 0.995
    T_curve = np.maximum(T_min, T_start * (T_decay ** (E - 1)))
    ax.plot(E, T_curve, color=C["T"], linewidth=1.8, label="T (reconstructed)")
    ax.axhline(T_min, color="#FFAA00", linewidth=0.7, linestyle="--",
               label=f"T_min={T_min}")
    ax.set_ylim(0, T_start * 1.05)
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,3] Summary table
    ax = _ax(2, 3, "Final Epoch Summary", "")
    ax.axis("off")

    def _last(key, fmt=".4f"):
        v = _get(h, key)
        return f"{v[-1]:{fmt}}" if v is not None else "n/a"

    def _min_val(key, fmt=".4f"):
        v = _get(h, key)
        return f"{np.nanmin(v):{fmt}}" if v is not None else "n/a"

    ratio_last = "n/a"
    if ds is not None and dt is not None:
        ratio_last = f"{abs(dt[-1]) / (abs(ds[-1]) + 1e-8):.2f}"

    rows = [
        ("Epochs",          f"{len(epochs)}"),
        ("G  (last)",       _last("generator",     ".3f")),
        ("D_s (last)",      _last("disc_static",   ".3f")),
        ("D_t (last)",      _last("disc_temporal", ".3f")),
        ("|D_t|/|D_s|",     ratio_last),
        ("─"*22,            ""),
        ("Fup (last)",      _last("fup_loss")),
        ("Fc  (last)",      _last("fc_loss")),
        ("Fc  (min)",       _min_val("fc_loss")),
        ("─"*22,            ""),
        ("NvL (last)",      _last("nv_loss", ".3f")),
        ("Nv  (last)",      _last("mean_n_visits", ".2f")),
        ("─"*22,            ""),
        ("ScatL (last)",    _last("static_cat_loss")),
        ("FmL  (last)",     _last("fm_loss")),
        ("Fgen (last)",     _last("freq_gen")),
        ("Irr  (last)",     _last("irreversibility")),
    ]

    y_pos = 0.97
    for label, value in rows:
        if label.startswith("─"):
            y_pos -= 0.015
            ax.plot([0.01, 0.97], [y_pos + 0.01, y_pos + 0.01],
                    color="#333333", linewidth=0.6, transform=ax.transAxes, clip_on=False)
            y_pos -= 0.015
            continue
        ax.text(0.02, y_pos, label, transform=ax.transAxes,
                fontsize=7.5, color="#AAAAAA", fontfamily="monospace", va="top")
        ax.text(0.62, y_pos, value, transform=ax.transAxes,
                fontsize=7.5, color="white", fontfamily="monospace", va="top",
                fontweight="bold")
        y_pos -= 0.052

    hist_path = out_dir / "plot_history.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"  -> plot_history.png saved: {hist_path}")

    # ── 2. CONFIG SUMMARY ─────────────────────────────────────────────────────
    _plot_config(dgan, timestr, out_dir, config_path)


def _plot_config(dgan, timestr, out_dir, config_path=None):
    """
    Genera plot_config.png con:
      - Parametri del modello (architettura, lambda, lr, ecc.)
      - Dimensioni dataset
      - Note sulla temperatura inference
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), facecolor="#0F0F0F")
    fig.suptitle(
        f"DGAN Configuration Summary  —  exp {timestr}",
        fontsize=13, color="white", fontweight="bold", y=0.97,
    )

    for ax in axes:
        ax.set_facecolor("#0F0F0F")
        ax.axis("off")

    def _section(ax, x, y, title, rows, col_w=0.55):
        ax.text(x, y, title, transform=ax.transAxes,
                fontsize=9, color="#E63946", fontweight="bold",
                fontfamily="monospace", va="top")
        y -= 0.045
        ax.plot([x, x + 0.88], [y + 0.01, y + 0.01],
                color="#333333", linewidth=0.8, transform=ax.transAxes,
                clip_on=False)
        y -= 0.01
        for label, value in rows:
            ax.text(x, y, f"{label}", transform=ax.transAxes,
                    fontsize=8, color="#AAAAAA", fontfamily="monospace", va="top")
            ax.text(x + col_w, y, f"{value}", transform=ax.transAxes,
                    fontsize=8, color="white", fontfamily="monospace", va="top",
                    fontweight="bold")
            y -= 0.042
        return y - 0.02

    mc = dgan.model_config
    dc = dgan.data_config

    def _g(attr, default="—"):
        return str(getattr(mc, attr, default))

    def _gc(sub, attr, default="—"):
        s = getattr(mc, sub, None)
        if s is None:
            return default
        return str(getattr(s, attr, default))

    # ── Colonna 0: Architettura ────────────────────────────────────────────
    ax = axes[0]
    y = 0.94
    y = _section(ax, 0.02, y, "DATASET", [
        ("Patients",       str(getattr(dc, "n_patients", "—"))),
        ("Max visits (T)", str(dc.max_len)),
        ("Static cont",    str(dc.n_static_cont)),
        ("Static cat vars",str(len(dc.n_static_cat))),
        ("Temporal cont",  str(dc.n_temp_cont)),
        ("Temporal cat",   str(len(dc.n_temp_cat))),
        ("Irreversible",   str(len(dc.irreversible_idx))),
        ("Static dim",     str(dgan.static_dim)),
        ("Temporal dim",   str(dgan.temporal_dim)),
    ])

    y = _section(ax, 0.02, y, "GENERATOR", [
        ("Backbone",       _gc("generator", "arch", "gru")),
        ("z_static_dim",   _g("z_static_dim")),
        ("z_temporal_dim", _g("z_temporal_dim")),
        ("hidden_dim",     _gc("generator", "hidden_dim", "128")),
        ("n_layers",       _gc("generator", "n_layers", _gc("generator", "gru_layers", "2"))),
        ("bidirectional",  _gc("generator", "bidirectional", "—")),
        ("dropout",        _gc("generator", "dropout", "—")),
        ("Params",         f"{sum(p.numel() for p in dgan.generator.parameters()):,}"),
    ])

    y = _section(ax, 0.02, y, "DISCRIMINATORS", [
        ("Static layers",  _gc("static_discriminator", "static_layers", "—")),
        ("Static hidden",  _gc("static_discriminator", "mlp_hidden_dim", "—")),
        ("Temporal arch",  _gc("temporal_discriminator", "arch", "cnn")),
        ("Temporal hidden",_gc("temporal_discriminator", "hidden_dim", "—")),
        ("Params (static)",f"{sum(p.numel() for p in dgan.disc_static.parameters()):,}"),
        ("Params (temp.)", f"{sum(p.numel() for p in dgan.disc_temporal.parameters()):,}"),
    ])

    # ── Colonna 1: Training hyperparameters ───────────────────────────────
    ax = axes[1]
    y = 0.94
    y = _section(ax, 0.02, y, "TRAINING", [
        ("Epochs",          _g("epochs")),
        ("Batch size",      _g("batch_size")),
        ("lr_g",            _g("lr_g")),
        ("lr_d_s",          _g("lr_d_s")),
        ("lr_d_t",          _g("lr_d_t")),
        ("critic_steps",    _g("critic_steps")),
        ("critic_steps_t",  _g("critic_steps_temporal", "—")),
        ("grad_clip",       _g("grad_clip")),
        ("T_start",         _g("gumbel_temperature_start")),
        ("T_min",           _g("temperature_min")),
        ("patience",        _g("patience")),
    ])

    y = _section(ax, 0.02, y, "LOSS WEIGHTS", [
        ("lambda_gp_s",       _g("lambda_gp_s")),
        ("lambda_gp_t",       _g("lambda_gp_t")),
        ("lambda_aux",        _g("lambda_aux")),
        ("lambda_freq_gen",   _g("lambda_freq_gen")),
        ("lambda_freq_disc",  _g("lambda_freq_disc")),
        ("freq_weight_power", _g("freq_weight_power")),
        ("lambda_fm",         _g("lambda_fm")),
        ("lambda_fup",        _g("lambda_fup")),
        ("lambda_fc",         _g("lambda_fc")),
        ("lambda_nv",         _g("lambda_nv")),
        ("lambda_static_cat", _g("lambda_static_cat")),
        ("alpha_irr (max)",   _g("alpha_irr")),
        ("n_visits_sharpness",_g("n_visits_sharpness")),
    ])

    # ── Colonna 2: Inference + note ───────────────────────────────────────
    ax = axes[2]
    y = 0.94
    y = _section(ax, 0.02, y, "INFERENCE", [
        ("Temperature used",  "0.5"),
        ("", ""),
        ("T=0.3",  "deterministico, poca diversità"),
        ("T=0.5",  "← raccomandato (bilanciato)"),
        ("T=0.8",  "più variabilità, rischio rumore"),
        ("T=1.0",  "Gumbel-Softmax uniforme"),
    ])

    # Note diagnostiche basate sull'ultima epoca
    h = dgan.loss_history
    notes = []

    fc  = _get(h, "fc_loss")
    fup = _get(h, "fup_loss")
    sc  = _get(h, "static_cat_loss")
    nv  = _get(h, "mean_n_visits")
    ds  = _get(h, "disc_static")
    dt  = _get(h, "disc_temporal")

    if fc is not None and fc[-1] > 0.05:
        notes.append(("⚠ Fc alto",    f"{fc[-1]:.4f} — visite non coprono il follow-up"))
    else:
        notes.append(("✓ Fc",          f"{fc[-1]:.4f} — OK"))

    if fup is not None and fup[-1] > 0.05:
        notes.append(("⚠ Fup alto",   f"{fup[-1]:.4f} — followup_head non converge"))
    else:
        notes.append(("✓ Fup",         f"{fup[-1]:.4f} — OK"))

    if sc is not None and sc[-1] > 0.15:
        notes.append(("⚠ ScatL",      f"{sc[-1]:.4f} — cat. statiche non converge"))
    else:
        notes.append(("✓ ScatL",       f"{sc[-1]:.4f} — OK"))

    if nv is not None:
        notes.append(("Nv (last)",     f"{nv[-1]:.2f}"))

    if ds is not None and dt is not None:
        ratio = abs(dt[-1]) / (abs(ds[-1]) + 1e-8)
        sym = "⚠" if ratio > 2.5 else "✓"
        notes.append((f"{sym} |D_t|/|D_s|", f"{ratio:.2f}  {'D_t dominante' if ratio>2.5 else 'bilanciato'}"))

    y = _section(ax, 0.02, y, "DIAGNOSTICS (last epoch)", notes)

    # Config JSON raw se disponibile
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            model_section = cfg.get("model", {})
            raw_lines = json.dumps(model_section, indent=2).split("\n")[:35]
            raw_text  = "\n".join(raw_lines)
            if len(json.dumps(model_section, indent=2).split("\n")) > 35:
                raw_text += "\n  ..."

            ax.text(0.02, y, "CONFIG JSON (model section)", transform=ax.transAxes,
                    fontsize=9, color="#E63946", fontweight="bold",
                    fontfamily="monospace", va="top")
            y -= 0.055
            ax.text(0.02, y, raw_text, transform=ax.transAxes,
                    fontsize=6.2, color="#888888", fontfamily="monospace", va="top",
                    linespacing=1.4)
        except Exception:
            pass

    cfg_path = out_dir / "plot_config.png"
    fig.savefig(cfg_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"  -> plot_config.png saved:  {cfg_path}")