"""
export_plots.py – Publication-quality matplotlib figure export.

Four-panel (2×2) layout:
  (a) Full deflection vs Z
  (b) Contact region deflection vs Z
  (c) Deflection (nm) vs ΔZ (nm) — INVOLS validation
  (d) Force (nN) vs δ (nm)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

FONT_SIZE  = 9
TICK_SIZE  = 8
LABEL_SIZE = 9
TITLE_SIZE = 10
LINE_W     = 0.9
MARKER_S   = 1.5
FIT_W      = 1.8

APP_COLOR  = "#0077bb"
RET_COLOR  = "#cc3311"
FAPP_COLOR = "#004488"
FRET_COLOR = "#882211"


def _style(ax, xlabel, ylabel, title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    if title:
        ax.set_title(title, fontsize=TITLE_SIZE, loc="left", weight="bold")
    ax.tick_params(labelsize=TICK_SIZE, direction="in", top=True, right=True)
    ax.legend(fontsize=FONT_SIZE - 1, frameon=True, edgecolor="grey",
              fancybox=False, loc="best")


@dataclass
class ExportData:
    # Full curves (V or µm vs V)
    app_z: np.ndarray;       app_defl: np.ndarray
    ret_z: np.ndarray;       ret_defl: np.ndarray
    # Contact-region (V or µm vs V)
    app_z_cr: np.ndarray;    app_defl_cr: np.ndarray
    ret_z_cr: np.ndarray;    ret_defl_cr: np.ndarray
    # Fit lines in Z-space
    fit_app_x: np.ndarray;   fit_app_y: np.ndarray
    fit_ret_x: np.ndarray;   fit_ret_y: np.ndarray
    # Panel (c): defl(nm) vs ΔZ(nm)
    app_dz_nm: np.ndarray;   app_defl_nm: np.ndarray
    ret_dz_nm: np.ndarray;   ret_defl_nm: np.ndarray
    # Panel (d): F(nN) vs δ(nm)
    app_delta: np.ndarray;   app_F: np.ndarray
    ret_delta: np.ndarray;   ret_F: np.ndarray
    # Fit lines panel (c)
    fit_app_dz: np.ndarray;  fit_app_defl: np.ndarray
    fit_ret_dz: np.ndarray;  fit_ret_defl: np.ndarray
    # Fit lines panel (d)
    fit_app_delta: np.ndarray; fit_app_F: np.ndarray
    fit_ret_delta: np.ndarray; fit_ret_F: np.ndarray
    # Meta
    x_label: str
    slope_app: float; slope_ret: float
    invols: float; k: float; phase_deg: float
    cp_app_idx: int; cp_ret_idx: int
    show: str
    filename: str = ""


def export_figure(data: ExportData, out_dir: Path,
                  fmt: str = "pdf", dpi: int = 300) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 7.1))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]
    show = data.show

    # ── (a) Full raw ─────────────────────────────────────────────────
    if show in ("Both", "Approach"):
        ax_a.plot(data.app_z, data.app_defl, color=APP_COLOR, lw=LINE_W,
                  label="Approach", zorder=2)
    if show in ("Both", "Retract"):
        ax_a.plot(data.ret_z, data.ret_defl, color=RET_COLOR, lw=LINE_W,
                  label="Retract", zorder=2)
    _style(ax_a, data.x_label, "Deflection (V)", "(a)")

    # ── (b) Contact region ───────────────────────────────────────────
    if show in ("Both", "Approach"):
        ax_b.scatter(data.app_z_cr, data.app_defl_cr, s=MARKER_S,
                     color=APP_COLOR, label="Approach", zorder=2)
        if len(data.fit_app_x) > 1:
            ax_b.plot(data.fit_app_x, data.fit_app_y, color=FAPP_COLOR,
                      lw=FIT_W, label=f"Fit (s={data.slope_app:.1f})", zorder=3)
    if show in ("Both", "Retract"):
        ax_b.scatter(data.ret_z_cr, data.ret_defl_cr, s=MARKER_S,
                     color=RET_COLOR, label="Retract", zorder=2)
        if len(data.fit_ret_x) > 1:
            ax_b.plot(data.fit_ret_x, data.fit_ret_y, color=FRET_COLOR,
                      lw=FIT_W, label=f"Fit (s={data.slope_ret:.1f})", zorder=3)
    _style(ax_b, data.x_label, "Deflection (V)", "(b)")

    # ── (c) Deflection (nm) vs ΔZ (nm) ──────────────────────────────
    if show in ("Both", "Approach"):
        ax_c.scatter(data.app_dz_nm, data.app_defl_nm, s=MARKER_S,
                     color=APP_COLOR, label="Approach", zorder=2)
        if len(data.fit_app_dz) > 1:
            ax_c.plot(data.fit_app_dz, data.fit_app_defl, color=FAPP_COLOR,
                      lw=FIT_W, zorder=3)
    if show in ("Both", "Retract"):
        ax_c.scatter(data.ret_dz_nm, data.ret_defl_nm, s=MARKER_S,
                     color=RET_COLOR, label="Retract", zorder=2)
        if len(data.fit_ret_dz) > 1:
            ax_c.plot(data.fit_ret_dz, data.fit_ret_defl, color=FRET_COLOR,
                      lw=FIT_W, zorder=3)
    # y=x reference
    all_dz = []
    if show in ("Both", "Approach") and len(data.app_dz_nm) > 0:
        all_dz.extend([data.app_dz_nm.min(), data.app_dz_nm.max()])
    if show in ("Both", "Retract") and len(data.ret_dz_nm) > 0:
        all_dz.extend([data.ret_dz_nm.min(), data.ret_dz_nm.max()])
    if all_dz:
        lo, hi = min(all_dz), max(all_dz)
        ax_c.plot([lo, hi], [lo, hi], '--', color='grey', lw=0.8,
                  label="slope = 1", zorder=1)
    ax_c.axhline(0, lw=0.4, color="grey", zorder=1)
    ax_c.axvline(0, lw=0.4, color="grey", zorder=1)
    _style(ax_c, "ΔZ (nm)", "Deflection (nm)", "(c)")

    # ── (d) F vs δ ──────────────────────────────────────────────────
    if show in ("Both", "Approach"):
        ax_d.scatter(data.app_delta, data.app_F, s=MARKER_S,
                     color=APP_COLOR, label="Approach", zorder=2)
        if len(data.fit_app_delta) > 1:
            ax_d.plot(data.fit_app_delta, data.fit_app_F, color=FAPP_COLOR,
                      lw=FIT_W, zorder=3)
    if show in ("Both", "Retract"):
        ax_d.scatter(data.ret_delta, data.ret_F, s=MARKER_S,
                     color=RET_COLOR, label="Retract", zorder=2)
        if len(data.fit_ret_delta) > 1:
            ax_d.plot(data.fit_ret_delta, data.fit_ret_F, color=FRET_COLOR,
                      lw=FIT_W, zorder=3)
    ax_d.axhline(0, lw=0.4, color="grey", zorder=1)
    ax_d.axvline(0, lw=0.4, color="grey", zorder=1)
    _style(ax_d, "δ (nm)", "F (nN)", "(d)")

    info = (f"INVOLS = {data.invols:.1f} nm/V   "
            f"k = {data.k:.3f} N/m   "
            f"φ = {data.phase_deg:.2f}°")
    fig.text(0.50, 0.005, info, ha="center", fontsize=FONT_SIZE - 1,
             color="grey")

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    stem = data.filename or "afm_force_curve"
    out_path = out_dir / f"{stem}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
