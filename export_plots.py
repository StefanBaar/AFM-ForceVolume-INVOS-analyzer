"""
export_plots.py – Publication-quality matplotlib figure export.

Six-panel (3×2) layout:
  (a) Full deflection vs Z  (with CP vertical lines)
  (b) Contact region deflection vs Z
  (c) Deflection (nm) vs ΔZ (nm) — INVOLS validation
  (d) Force (nN) vs δ (nm)
  (e) Local slope vs ΔZ from CP
  (f) Slope deviation from CP reference

Colors: cornflowerblue (approach) / orange (retract).
Data is downsampled for efficient PDF rendering.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass, field

FONT_SIZE  = 9
TICK_SIZE  = 8
LABEL_SIZE = 9
TITLE_SIZE = 10
LINE_W     = 0.6
MARKER_S   = 0.8
FIT_W      = 1.8
ALPHA      = 0.6

APP_COLOR  = "cornflowerblue"
RET_COLOR  = "orange"
FAPP_COLOR = "#3366aa"
FRET_COLOR = "#cc7700"
CP_LW      = 1.2

MAX_PTS    = 3000   # downsample target per trace


def _ds(x, y, n=MAX_PTS):
    """Downsample two arrays to at most *n* points."""
    if len(x) <= n:
        return x, y
    s = max(1, len(x) // n)
    return x[::s], y[::s]


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
    # Full curves
    app_z: np.ndarray;       app_defl: np.ndarray
    ret_z: np.ndarray;       ret_defl: np.ndarray
    # Contact-region
    app_z_cr: np.ndarray;    app_defl_cr: np.ndarray
    ret_z_cr: np.ndarray;    ret_defl_cr: np.ndarray
    # Fit lines
    fit_app_x: np.ndarray;   fit_app_y: np.ndarray
    fit_ret_x: np.ndarray;   fit_ret_y: np.ndarray
    # Panel (c)
    app_dz_nm: np.ndarray;   app_defl_nm: np.ndarray
    ret_dz_nm: np.ndarray;   ret_defl_nm: np.ndarray
    # Panel (d)
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
    cp_app_x: float; cp_ret_x: float    # x-axis value of CP in plot 1 units
    show: str
    filename: str = ""
    # Linearity (optional)
    lin_app_dz: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_app_slope: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_app_dev: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_app_ref: float = 0.0
    lin_app_end_nm: float = 0.0
    lin_ret_dz: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_ret_slope: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_ret_dev: np.ndarray = field(default_factory=lambda: np.array([]))
    lin_ret_ref: float = 0.0
    lin_ret_end_nm: float = 0.0
    dev_threshold: float = 10.0


def export_figure(data: ExportData, out_dir: Path,
                  fmt: str = "pdf", dpi: int = 300) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(7.5, 10))
    (ax_a, ax_b), (ax_c, ax_d), (ax_e, ax_f) = axes
    show = data.show

    # ── (a) Full raw + CP lines ──────────────────────────────────────
    if show in ("Both", "Approach"):
        x, y = _ds(data.app_z, data.app_defl)
        ax_a.plot(x, y, color=APP_COLOR, lw=LINE_W, alpha=ALPHA,
                  label="Approach", zorder=2)
        ax_a.axvline(data.cp_app_x, color=FAPP_COLOR, lw=CP_LW, ls="--",
                     label="CP app", zorder=3)
    if show in ("Both", "Retract"):
        x, y = _ds(data.ret_z, data.ret_defl)
        ax_a.plot(x, y, color=RET_COLOR, lw=LINE_W, alpha=ALPHA,
                  label="Retract", zorder=2)
        ax_a.axvline(data.cp_ret_x, color=FRET_COLOR, lw=CP_LW, ls="--",
                     label="CP ret", zorder=3)
    _style(ax_a, data.x_label, "Deflection (V)",
           "(a) Deflection vs Z – Full")

    # ── (b) Contact region ───────────────────────────────────────────
    if show in ("Both", "Approach"):
        x, y = _ds(data.app_z_cr, data.app_defl_cr)
        ax_b.scatter(x, y, s=MARKER_S, color=APP_COLOR, alpha=ALPHA,
                     label="Approach", zorder=2)
        if len(data.fit_app_x) > 1:
            ax_b.plot(data.fit_app_x, data.fit_app_y, color=FAPP_COLOR,
                      lw=FIT_W, label=f"Fit (s={data.slope_app:.1f})", zorder=3)
    if show in ("Both", "Retract"):
        x, y = _ds(data.ret_z_cr, data.ret_defl_cr)
        ax_b.scatter(x, y, s=MARKER_S, color=RET_COLOR, alpha=ALPHA,
                     label="Retract", zorder=2)
        if len(data.fit_ret_x) > 1:
            ax_b.plot(data.fit_ret_x, data.fit_ret_y, color=FRET_COLOR,
                      lw=FIT_W, label=f"Fit (s={data.slope_ret:.1f})", zorder=3)
    _style(ax_b, data.x_label, "Deflection (V)", "(b) Contact Region")

    # ── (c) Deflection (nm) vs ΔZ (nm) ──────────────────────────────
    if show in ("Both", "Approach"):
        x, y = _ds(data.app_dz_nm, data.app_defl_nm)
        ax_c.scatter(x, y, s=MARKER_S, color=APP_COLOR, alpha=ALPHA,
                     label="Approach", zorder=2)
        if len(data.fit_app_dz) > 1:
            ax_c.plot(data.fit_app_dz, data.fit_app_defl, color=FAPP_COLOR,
                      lw=FIT_W, zorder=3)
    if show in ("Both", "Retract"):
        x, y = _ds(data.ret_dz_nm, data.ret_defl_nm)
        ax_c.scatter(x, y, s=MARKER_S, color=RET_COLOR, alpha=ALPHA,
                     label="Retract", zorder=2)
        if len(data.fit_ret_dz) > 1:
            ax_c.plot(data.fit_ret_dz, data.fit_ret_defl, color=FRET_COLOR,
                      lw=FIT_W, zorder=3)
    all_dz = []
    if show in ("Both", "Approach") and len(data.app_dz_nm) > 0:
        all_dz.extend([data.app_dz_nm.min(), data.app_dz_nm.max()])
    if show in ("Both", "Retract") and len(data.ret_dz_nm) > 0:
        all_dz.extend([data.ret_dz_nm.min(), data.ret_dz_nm.max()])
    if all_dz:
        lo, hi = min(all_dz), max(all_dz)
        ax_c.plot([lo, hi], [lo, hi], '--', color='grey', lw=0.8,
                  label="slope = 1", zorder=1)
    ax_c.axhline(0, lw=0.4, color="grey"); ax_c.axvline(0, lw=0.4, color="grey")
    _style(ax_c, "ΔZ (nm)", "Deflection (nm)", "(c) Deflection vs ΔZ")

    # ── (d) F vs δ ──────────────────────────────────────────────────
    if show in ("Both", "Approach"):
        x, y = _ds(data.app_delta, data.app_F)
        ax_d.scatter(x, y, s=MARKER_S, color=APP_COLOR, alpha=ALPHA,
                     label="Approach", zorder=2)
        if len(data.fit_app_delta) > 1:
            ax_d.plot(data.fit_app_delta, data.fit_app_F, color=FAPP_COLOR,
                      lw=FIT_W, zorder=3)
    if show in ("Both", "Retract"):
        x, y = _ds(data.ret_delta, data.ret_F)
        ax_d.scatter(x, y, s=MARKER_S, color=RET_COLOR, alpha=ALPHA,
                     label="Retract", zorder=2)
        if len(data.fit_ret_delta) > 1:
            ax_d.plot(data.fit_ret_delta, data.fit_ret_F, color=FRET_COLOR,
                      lw=FIT_W, zorder=3)
    ax_d.axhline(0, lw=0.4, color="grey"); ax_d.axvline(0, lw=0.4, color="grey")
    _style(ax_d, "δ (nm)", "F (nN)", "(d) Force vs Indentation")

    # ── (e) Local slope vs ΔZ ────────────────────────────────────────
    if show in ("Both", "Approach") and len(data.lin_app_dz) > 0:
        x, y = _ds(data.lin_app_dz, data.lin_app_slope)
        ax_e.scatter(x, y, s=2, color=APP_COLOR, alpha=ALPHA, label="Approach")
        ax_e.axhline(data.lin_app_ref, color=FAPP_COLOR, lw=1, ls=":",
                     label=f"ref = {data.lin_app_ref:.1f}")
        if data.lin_app_end_nm > 0:
            ax_e.axvline(data.lin_app_end_nm, color=FAPP_COLOR, lw=1, ls="--")
    if show in ("Both", "Retract") and len(data.lin_ret_dz) > 0:
        x, y = _ds(data.lin_ret_dz, data.lin_ret_slope)
        ax_e.scatter(x, y, s=2, color=RET_COLOR, alpha=ALPHA, label="Retract")
        ax_e.axhline(data.lin_ret_ref, color=FRET_COLOR, lw=1, ls=":")
        if data.lin_ret_end_nm > 0:
            ax_e.axvline(data.lin_ret_end_nm, color=FRET_COLOR, lw=1, ls="--")
    _style(ax_e, "ΔZ from CP (nm)", "Local slope (V/V)",
           "(e) Local Slope vs ΔZ")

    # ── (f) Deviation (%) vs ΔZ ──────────────────────────────────────
    th = data.dev_threshold
    ax_f.axhspan(-th, th, alpha=0.08, color="green")
    ax_f.axhline(th, lw=0.8, ls=":", color="green")
    ax_f.axhline(-th, lw=0.8, ls=":", color="green")
    ax_f.axhline(0, lw=0.4, color="grey")
    if show in ("Both", "Approach") and len(data.lin_app_dz) > 0:
        x, y = _ds(data.lin_app_dz, data.lin_app_dev)
        ax_f.scatter(x, y, s=2, color=APP_COLOR, alpha=ALPHA, label="Approach")
        if data.lin_app_end_nm > 0:
            ax_f.axvline(data.lin_app_end_nm, color=FAPP_COLOR, lw=1, ls="--",
                         label=f"{data.lin_app_end_nm:.0f} nm")
    if show in ("Both", "Retract") and len(data.lin_ret_dz) > 0:
        x, y = _ds(data.lin_ret_dz, data.lin_ret_dev)
        ax_f.scatter(x, y, s=2, color=RET_COLOR, alpha=ALPHA, label="Retract")
        if data.lin_ret_end_nm > 0:
            ax_f.axvline(data.lin_ret_end_nm, color=FRET_COLOR, lw=1, ls="--",
                         label=f"{data.lin_ret_end_nm:.0f} nm")
    _style(ax_f, "ΔZ from CP (nm)", "Deviation (%)",
           f"(f) Slope Deviation (±{th:.0f}%)")

    info = (f"INVOLS = {data.invols:.1f} nm/V   "
            f"k = {data.k:.3f} N/m   "
            f"φ = {data.phase_deg:.2f}°")
    fig.text(0.50, 0.003, info, ha="center", fontsize=FONT_SIZE - 1,
             color="grey")

    fig.tight_layout(rect=[0, 0.015, 1, 1])
    stem = data.filename or "afm_force_curve"
    out_path = out_dir / f"{stem}.{fmt}"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path
