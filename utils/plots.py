"""
utils/plots.py
================================================================================
Plot della storia di training DGAN  [v2 — allineato a loss_history gretel-style]

Chiavi loss_history supportate (nuova architettura):
  WGAN:        generator, disc_static, disc_temporal
  GP:          gp_static, gp_temporal
  Ausiliarie:  irr_loss, fup_loss, nv_loss, scat_loss, fm_loss, aux_loss, var_loss
  Monitoring:  mean_n_visits, fake_cont_mean, fake_cont_std, real_cont_mean, real_cont_std

  plot_training_history(dgan, timestr)
    → output/exp_{timestr}/training_history.png  (pannello rapido 2×2)

  plot_training_history2(dgan, timestr, config_path=None)
    → output/exp_{timestr}/plot_history.png      (pannello completo dark)
    → output/exp_{timestr}/plot_config.png       (riepilogo parametri)

Note temperatura inference:
  T=0.3 → deterministico, poca diversità
  T=0.5 → raccomandato (default in main.py)
  T=0.8 → più variabilità, rischio rumore
  T=1.0 → Gumbel-Softmax quasi uniforme
================================================================================
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Palette colori ────────────────────────────────────────────────────────────
C = {
    "G":     "#E63946",   # rosso      — generatore
    "D_s":   "#457B9D",   # blu        — disc statico
    "D_t":   "#1D3557",   # blu scuro  — disc temporale
    "Fup":   "#2A9D8F",   # verde      — followup
    "NvL":   "#F4A261",   # arancio    — n_visits
    "Scat":  "#A8DADC",   # celeste    — categoriche statiche
    "Fm":    "#6A4C93",   # viola      — feature matching
    "Irr":   "#B5838D",   # rosa       — irreversibilità
    "Var":   "#F6BD60",   # giallo oro — varianza feature continue
    "Aux":   "#95D5B2",   # verde chiaro — aux embed
    "GP":    "#8ECAE6",   # azzurro    — gradient penalty
    "Cont":  "#FF9F1C",   # arancio    — medie feature continue
    "T":     "#888888",   # grigio     — temperatura
}


def _smooth(values: np.ndarray, w: int = 5) -> np.ndarray:
    """Media mobile semplice per leggibilità delle curve."""
    if len(values) < w * 2:
        return values
    kernel = np.ones(w) / w
    padded = np.pad(values, (w // 2, w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def _get(history: dict, key: str) -> np.ndarray | None:
    """Recupera una serie dalla history; restituisce None se assente o vuota."""
    v = history.get(key)
    if v is None or len(v) == 0:
        return None
    arr = np.array(v, dtype=float)
    return arr if np.isfinite(arr).any() else None


def _last(history: dict, key: str, fmt: str = ".4f") -> str:
    v = _get(history, key)
    return f"{v[-1]:{fmt}}" if v is not None else "n/a"


def _min_val(history: dict, key: str, fmt: str = ".4f") -> str:
    v = _get(history, key)
    return f"{np.nanmin(v):{fmt}}" if v is not None else "n/a"


# ==============================================================================
# PLOT RAPIDO 2×2  (plot_training_history)
# ==============================================================================

def plot_training_history(dgan, timestr: str):
    """Pannello rapido 2×2: G, D, irr_loss, scat_loss."""
    try:
        out_dir = Path(f"output/exp_{timestr}")
        out_dir.mkdir(parents=True, exist_ok=True)
        h = dgan.loss_history

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        E = list(range(1, len(h.get("generator", [])) + 1))

        def _plot_ax(ax, keys, labels, colors, title):
            for k, lbl, col in zip(keys, labels, colors):
                v = _get(h, k)
                if v is not None:
                    ax.plot(E[:len(v)], v, color=col, alpha=0.3, linewidth=0.8)
                    ax.plot(E[:len(v)], _smooth(v), color=col, linewidth=1.8, label=lbl)
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        _plot_ax(axes[0, 0],
                 ["generator"], ["Generator"], [C["G"]],
                 "Generator Loss (WGAN)")

        _plot_ax(axes[0, 1],
                 ["disc_static", "disc_temporal"],
                 ["D_static", "D_temporal"],
                 [C["D_s"], C["D_t"]],
                 "Discriminator Losses")

        _plot_ax(axes[1, 0],
                 ["irr_loss", "var_loss"],
                 ["Irreversibility", "Variance"],
                 [C["Irr"], C["Var"]],
                 "Irreversibility & Variance Losses")

        _plot_ax(axes[1, 1],
                 ["scat_loss", "fup_loss", "nv_loss"],
                 ["Scat (cat marginal)", "Fup", "NvL"],
                 [C["Scat"], C["Fup"], C["NvL"]],
                 "Auxiliary Losses")

        plt.tight_layout()
        path = out_dir / "training_history.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  -> training_history.png saved: {path}")
    except Exception as e:
        logger.warning(f"Could not plot training_history: {e}")


# ==============================================================================
# PLOT COMPLETO DARK  (plot_training_history2)
# ==============================================================================

def plot_training_history2(dgan, timestr: str, config_path: str = None):
    """
    Genera due PNG:
      output/exp_{timestr}/plot_history.png  — tutte le loss curve
      output/exp_{timestr}/plot_config.png   — riepilogo parametri
    """
    out_dir = Path(f"output/exp_{timestr}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h = dgan.loss_history
    n_epochs = len(h.get("generator", []))
    if n_epochs == 0:
        print("[plots] Nessuna storia di training disponibile.")
        return

    E = np.arange(1, n_epochs + 1)

    # ── helpers grafici ────────────────────────────────────────────────────────
    def _ax(fig_obj, gs_obj, row, col, title, ylabel=None):
        ax = fig_obj.add_subplot(gs_obj[row, col])
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

    def _plot(ax, key, label, color, lw=1.5, alpha_raw=0.25):
        vals = _get(h, key)
        if vals is None:
            return
        e = E[:len(vals)]
        ax.plot(e, vals, color=color, alpha=alpha_raw, linewidth=0.8)
        ax.plot(e, _smooth(vals), color=color, linewidth=lw, label=label)

    def _twin(ax, key, label, color):
        """Asse Y secondario per una seconda serie sulla stessa figura."""
        vals = _get(h, key)
        if vals is None:
            return
        ax2 = ax.twinx()
        ax2.set_facecolor("#1A1A1A")
        e = E[:len(vals)]
        ax2.plot(e, vals, color=color, alpha=0.3, linewidth=0.8)
        ax2.plot(e, _smooth(vals), color=color, linewidth=1.5, label=label)
        ax2.tick_params(colors="#AAAAAA", labelsize=7)
        ax2.legend(loc="upper right", fontsize=7,
                   facecolor="#111111", labelcolor="white", framealpha=0.7)
        return ax2

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURA 1 — LOSS CURVES  (3 righe × 4 colonne)
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(22, 18), facecolor="#0F0F0F")
    fig.suptitle(
        f"DGAN Training History  —  exp {timestr}  "
        f"({n_epochs} epochs,  {datetime.now().strftime('%Y-%m-%d')})",
        fontsize=14, color="white", y=0.98, fontweight="bold",
    )
    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.42, wspace=0.30,
        left=0.05, right=0.97, top=0.94, bottom=0.05,
    )

    # ── Riga 0 ────────────────────────────────────────────────────────────────

    # [0,0] WGAN scores
    ax = _ax(fig, gs, 0, 0, "WGAN Scores", "score")
    _plot(ax, "generator",     "G",    C["G"])
    _plot(ax, "disc_static",   "D_s",  C["D_s"])
    _plot(ax, "disc_temporal", "D_t",  C["D_t"])
    ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,1] |D_t| / |D_s| ratio (indicatore di bilanciamento)
    ax = _ax(fig, gs, 0, 1, "|D_t| / |D_s|  (target ~= 1)", "ratio")
    ds = _get(h, "disc_static")
    dt = _get(h, "disc_temporal")
    if ds is not None and dt is not None:
        n = min(len(ds), len(dt))
        ratio = np.abs(dt[:n]) / (np.abs(ds[:n]) + 1e-8)
        ax.plot(E[:n], ratio, color="#FF9F1C", alpha=0.3, linewidth=0.8)
        ax.plot(E[:n], _smooth(ratio), color="#FF9F1C", linewidth=1.5, label="|D_t|/|D_s|")
        ax.axhline(1.0, color="#AAAAAA", linewidth=0.8, linestyle="--", label="target=1")
        ax.axhline(2.5, color="#E63946", linewidth=0.6, linestyle=":", label="warn=2.5")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,2] Gradient penalty
    ax = _ax(fig, gs, 0, 2, "Gradient Penalty", "GP")
    _plot(ax, "gp_static",   "GP_s", C["GP"])
    _plot(ax, "gp_temporal", "GP_t", C["D_t"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [0,3] Irreversibility loss
    ax = _ax(fig, gs, 0, 3, "Irreversibility Loss", "loss")
    _plot(ax, "irr_loss", "Irr", C["Irr"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # ── Riga 1 ────────────────────────────────────────────────────────────────

    # [1,0] Followup supervision
    ax = _ax(fig, gs, 1, 0, "Follow-up Supervision (Fup)", "loss")
    _plot(ax, "fup_loss", "Fup (MSE mean+std)", C["Fup"])
    ax.axhline(0.03, color=C["Fup"], linewidth=0.5, linestyle=":", alpha=0.6,
               label="target ~= 0.03")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [1,1] N_visits supervision + mean n_visits
    ax = _ax(fig, gs, 1, 1, "Visit Count Supervision", "NvL")
    _plot(ax, "nv_loss", "NvL (supervision)", C["NvL"])
    _twin(ax, "mean_n_visits", "Nv mean", "#FFE66D")
    ax.legend(loc="upper left", fontsize=7,
              facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [1,2] Static categorical marginal (scat_loss)
    ax = _ax(fig, gs, 1, 2, "Static Categorical Marginal (Scat)", "KL loss")
    _plot(ax, "scat_loss", "ScatL (KL divergence)", C["Scat"])
    ax.axhline(0.1, color=C["Scat"], linewidth=0.6, linestyle=":",
               alpha=0.7, label="target < 0.1")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [1,3] Feature matching + aux loss
    ax = _ax(fig, gs, 1, 3, "Feature Matching & Aux Loss", "loss")
    _plot(ax, "fm_loss",  "Fm  (feat. matching)", C["Fm"])
    _plot(ax, "aux_loss", "Aux (embed CE)",        C["Aux"])
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # ── Riga 2 ────────────────────────────────────────────────────────────────

    # [2,0] Variance + Interval losses
    ax = _ax(fig, gs, 2, 0, "Variance & Interval Losses", "loss")
    _plot(ax, "var_loss",      "VarL (feature std)",   C["Var"])
    _plot(ax, "interval_loss", "IvL  (inter-visit)",   C["Fup"])
    ax.axhline(0.05, color=C["Var"], linewidth=0.5, linestyle=":",
               alpha=0.5, label="target < 0.05")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,1] Cont mean monitoring: fake vs real
    ax = _ax(fig, gs, 2, 1, "Temporal Cont. Mean (fake vs real)", "z-score mean")
    _plot(ax, "fake_cont_mean", "fake mean", C["Cont"])
    real_m = _get(h, "real_cont_mean")
    if real_m is not None:
        ax.plot(E[:len(real_m)], real_m, color="#AAAAAA",
                linewidth=1.0, linestyle="--", label="real mean")
    ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,2] Cont std monitoring: fake vs real
    ax = _ax(fig, gs, 2, 2, "Temporal Cont. Std (fake vs real)", "z-score std")
    _plot(ax, "fake_cont_std", "fake std", C["Cont"])
    real_s = _get(h, "real_cont_std")
    if real_s is not None:
        ax.plot(E[:len(real_s)], real_s, color="#AAAAAA",
                linewidth=1.0, linestyle="--", label="real std")
    ax.axhline(1.0, color="#AAAAAA", linewidth=0.7, linestyle=":",
               label="target std=1 (z-score)")
    ax.legend(fontsize=7, facecolor="#111111", labelcolor="white", framealpha=0.7)

    # [2,3] Summary table
    ax = _ax(fig, gs, 2, 3, "Final Epoch Summary", "")
    ax.axis("off")

    ratio_last = "n/a"
    if ds is not None and dt is not None:
        ratio_last = f"{abs(dt[-1]) / (abs(ds[-1]) + 1e-8):.2f}"

    rows = [
        ("Epochs",            str(n_epochs)),
        ("G  (last)",         _last(h, "generator",     ".3f")),
        ("D_s (last)",        _last(h, "disc_static",   ".3f")),
        ("D_t (last)",        _last(h, "disc_temporal", ".3f")),
        ("|D_t|/|D_s|",       ratio_last),
        ("──────────────────", ""),
        ("GP_s (last)",       _last(h, "gp_static",  ".3f")),
        ("GP_t (last)",       _last(h, "gp_temporal", ".3f")),
        ("──────────────────", ""),
        ("Fup  (last)",       _last(h, "fup_loss")),
        ("NvL  (last)",       _last(h, "nv_loss",  ".3f")),
        ("Nv   (last)",       _last(h, "mean_n_visits", ".2f")),
        ("──────────────────", ""),
        ("ScatL (last)",      _last(h, "scat_loss")),
        ("FmL   (last)",      _last(h, "fm_loss")),
        ("VarL  (last)",      _last(h, "var_loss")),
        ("IvL   (last)",      _last(h, "interval_loss")),
        ("Irr   (last)",      _last(h, "irr_loss")),
        ("Aux   (last)",      _last(h, "aux_loss")),
        ("──────────────────", ""),
        ("Cont mean (fake)",  _last(h, "fake_cont_mean", ".3f")),
        ("Cont std  (fake)",  _last(h, "fake_cont_std",  ".3f")),
    ]

    y_pos = 0.97
    for label, value in rows:
        if label.startswith("──"):
            y_pos -= 0.012
            ax.plot([0.01, 0.97], [y_pos + 0.008, y_pos + 0.008],
                    color="#333333", linewidth=0.6,
                    transform=ax.transAxes, clip_on=False)
            y_pos -= 0.012
            continue
        ax.text(0.02, y_pos, label, transform=ax.transAxes,
                fontsize=7, color="#AAAAAA", fontfamily="monospace", va="top")
        ax.text(0.60, y_pos, value, transform=ax.transAxes,
                fontsize=7, color="white", fontfamily="monospace", va="top",
                fontweight="bold")
        y_pos -= 0.046

    hist_path = out_dir / "plot_history.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"  -> plot_history.png saved:  {hist_path}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURA 2 — CONFIG SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    _plot_config(dgan, timestr, out_dir, config_path)


# ==============================================================================
# CONFIG SUMMARY  (usato da plot_training_history2)
# ==============================================================================

def _plot_config(dgan, timestr: str, out_dir: Path, config_path: str = None):
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
                color="#333333", linewidth=0.8,
                transform=ax.transAxes, clip_on=False)
        y -= 0.01
        for label, value in rows:
            ax.text(x, y, label, transform=ax.transAxes,
                    fontsize=8, color="#AAAAAA", fontfamily="monospace", va="top")
            ax.text(x + col_w, y, str(value), transform=ax.transAxes,
                    fontsize=8, color="white", fontfamily="monospace", va="top",
                    fontweight="bold")
            y -= 0.042
        return y - 0.02

    mc = dgan.model_config
    dc = dgan.data_config
    h  = dgan.loss_history
    # Definiti localmente per evitare NameError se _plot_config viene chiamata standalone
    ds = _get(h, "disc_static")
    dt = _get(h, "disc_temporal")

    def _g(attr, default="--"):
        return str(getattr(mc, attr, default))

    def _gc(sub, attr, default="--"):
        s = getattr(mc, sub, None)
        return str(getattr(s, attr, default)) if s else default

    # ── Colonna 0: Architettura ───────────────────────────────────────────────
    ax = axes[0]
    y = 0.94
    y = _section(ax, 0.02, y, "DATASET", [
        ("Max visits (T)",  str(dc.max_len)),
        ("Min visits",      str(dc.min_visits)),
        ("Static cont",     str(dc.n_static_cont)),
        ("Static cat vars", str(len(dc.n_static_cat))),
        ("Temporal cont",   str(dc.n_temp_cont)),
        ("Temporal cat",    str(len(dc.n_temp_cat))),
        ("Irreversible",    str(len(dc.irreversible_idx))),
        ("Static dim",      str(dgan.static_dim)),
        ("Temporal dim",    str(dgan.temporal_dim)),
    ])

    y = _section(ax, 0.02, y, "GENERATOR", [
        ("z_static_dim",    _g("z_static_dim")),
        ("z_temporal_dim",  _g("z_temporal_dim")),
        ("hidden_dim",      _gc("generator", "hidden_dim")),
        ("n_layers",        _gc("generator", "n_layers")),
        ("dropout",         _gc("generator", "dropout")),
        ("noise_ar_rho",    _g("noise_ar_rho")),
        ("ema_decay",       _g("ema_decay")),
        ("min_visits",      str(dc.min_visits)),
        ("Params",          f"{sum(p.numel() for p in dgan.generator.parameters()):,}"),
    ])

    y = _section(ax, 0.02, y, "DISCRIMINATORS", [
        ("Static layers",   _gc("static_discriminator", "static_layers")),
        ("Static hidden",   _gc("static_discriminator", "mlp_hidden_dim")),
        ("Temporal arch",   _gc("temporal_discriminator", "arch")),
        ("Temporal hidden", _gc("temporal_discriminator", "hidden_dim")),
        ("Params (static)", f"{sum(p.numel() for p in dgan.disc_static.parameters()):,}"),
        ("Params (temp.)",  f"{sum(p.numel() for p in dgan.disc_temporal.parameters()):,}"),
    ])

    # ── Colonna 1: Hyperparameters ────────────────────────────────────────────
    ax = axes[1]
    y = 0.94
    y = _section(ax, 0.02, y, "TRAINING", [
        ("Epochs",          _g("epochs")),
        ("Batch size",      _g("batch_size")),
        ("lr_g",            _g("lr_g")),
        ("lr_d_s",          _g("lr_d_s")),
        ("lr_d_t",          _g("lr_d_t")),
        ("critic_steps",    _g("critic_steps")),
        ("critic_steps_t",  _g("critic_steps_temporal")),
        ("grad_clip",       _g("grad_clip")),
        ("T_start",         _g("gumbel_temperature_start")),
        ("T_min",           _g("temperature_min")),
        ("patience",        _g("patience")),
        ("instance_noise",  f"{_g('instance_noise_start')} -> {_g('instance_noise_end')}"),
        ("scat_warmup_ep",  _g("lambda_scat_warmup_epochs")),
    ])

    y = _section(ax, 0.02, y, "LOSS WEIGHTS", [
        ("lambda_gp_s",     _g("lambda_gp_s")),
        ("lambda_gp_t",     _g("lambda_gp_t")),
        ("lambda_fup",      _g("lambda_fup")),
        ("lambda_nv",       _g("lambda_nv")),
        ("lambda_var",      _g("lambda_var")),
        ("lambda_scat",     _g("lambda_static_cat")),
        ("lambda_fm",       _g("lambda_fm")),
        ("lambda_aux",      _g("lambda_aux")),
        ("alpha_irr",       _g("alpha_irr")),
        ("lambda_interval", _g("lambda_interval")),
    ])

    # ── Colonna 2: Diagnostics ────────────────────────────────────────────────
    ax = axes[2]
    y = 0.94

    # Diagnostics automatiche dall'ultima epoca
    notes = []
    fup = _get(h, "fup_loss")
    sc  = _get(h, "scat_loss")
    nv  = _get(h, "mean_n_visits")
    var = _get(h, "var_loss")
    fc_mean = _get(h, "fake_cont_mean")
    fc_std  = _get(h, "fake_cont_std")

    if fup is not None:
        sym = "OK" if fup[-1] <= 0.05 else "WARN"
        notes.append((f"[{sym}] Fup", f"{fup[-1]:.4f}  ({'<= 0.05' if sym=='OK' else '> 0.05'})"))

    if sc is not None:
        sym = "OK" if sc[-1] <= 0.10 else "WARN"
        notes.append((f"[{sym}] ScatL", f"{sc[-1]:.4f}  ({'<= 0.10' if sym=='OK' else '> 0.10, collapse?'})"))

    iv = _get(h, "interval_loss")
    if var is not None:
        sym = "OK" if var[-1] <= 0.10 else "WARN"
        notes.append((f"[{sym}] VarL",  f"{var[-1]:.4f}  ({'variance OK' if sym=='OK' else 'variance mismatch'})"))
    if iv is not None:
        sym = "OK" if iv[-1] <= 0.05 else "WARN"
        notes.append((f"[{sym}] IvL", f"{iv[-1]:.4f}  ({'intervals OK' if sym=='OK' else 'timing compressed'})"))

    if nv is not None:
        notes.append(("     Nv (last)", f"{nv[-1]:.2f}"))

    if fc_mean is not None:
        notes.append(("     Cont mean",  f"{fc_mean[-1]:.3f}  (target ~= 0)"))
    if fc_std is not None:
        notes.append(("     Cont std",   f"{fc_std[-1]:.3f}  (target ~= 1)"))

    if ds is not None and dt is not None:
        ratio = abs(dt[-1]) / (abs(ds[-1]) + 1e-8)
        sym = "OK" if ratio <= 2.5 else "WARN"
        notes.append((f"[{sym}] |D_t|/|D_s|",
                       f"{ratio:.2f}  ({'balanced' if sym=='OK' else 'D_t dominant'})"))

    y = _section(ax, 0.02, y, "DIAGNOSTICS (last epoch)", notes)

    # Note temperatura
    y = _section(ax, 0.02, y, "INFERENCE TEMPERATURE", [
        ("T=0.3", "deterministico, poca diversita"),
        ("T=0.5", "<-- raccomandato (default)"),
        ("T=0.8", "piu variabilita, rischio rumore"),
        ("T=1.0", "Gumbel quasi uniforme"),
    ])

    # Config JSON raw
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            model_section = cfg.get("model", {})
            raw_lines = json.dumps(model_section, indent=2).split("\n")[:30]
            raw_text  = "\n".join(raw_lines)
            if len(json.dumps(model_section, indent=2).split("\n")) > 30:
                raw_text += "\n  ..."
            ax.text(0.02, y, "CONFIG JSON (model)", transform=ax.transAxes,
                    fontsize=9, color="#E63946", fontweight="bold",
                    fontfamily="monospace", va="top")
            y -= 0.055
            ax.text(0.02, y, raw_text, transform=ax.transAxes,
                    fontsize=6.0, color="#888888", fontfamily="monospace",
                    va="top", linespacing=1.4)
        except Exception:
            pass

    cfg_path = out_dir / "plot_config.png"
    fig.savefig(cfg_path, dpi=150, bbox_inches="tight", facecolor="#0F0F0F")
    plt.close(fig)
    print(f"  -> plot_config.png saved:  {cfg_path}")


# ==============================================================================
# FUNZIONE LEGACY (compatibilità con chiamate dirette al .pt)
# ==============================================================================

def plot_loss_history(loss_history_path: str, save_path: str = None):
    """Plotta la loss history da un file checkpoint .pt."""
    state = torch.load(loss_history_path, map_location="cpu")
    loss_history = state.get("loss_history")
    if not loss_history:
        raise ValueError("loss_history non trovato nel checkpoint")

    plt.figure(figsize=(12, 8))
    for key, color, label in [
        ("generator",     "blue",   "Generator"),
        ("disc_static",   "orange", "Disc Static"),
        ("disc_temporal", "green",  "Disc Temporal"),
        ("irr_loss",      "red",    "Irreversibility"),
        ("scat_loss",     "purple", "Scat (cat marginal)"),
    ]:
        v = loss_history.get(key)
        if v and len(v) > 0:
            plt.plot(v, label=label, color=color)

    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend(); plt.grid(True); plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot salvato: {save_path}")
    else:
        plt.show()