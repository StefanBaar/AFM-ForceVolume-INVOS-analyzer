"""
contact_point.py – Contact-point detection and linear fits for AFM force curves.

Primary method: **baseline-deviation**.
  1.  Fit a line to the known non-contact (baseline) portion.
  2.  Extrapolate it across the whole curve.
  3.  Walk *toward* the contact region and flag the first index where
      |deflection − baseline_prediction| exceeds  n_sigma · noise
      for n_consec consecutive points.

This avoids the length-bias inherent in piecewise-linear splitting
(which puts the CP too far into the contact region when contact is
longer than baseline).

A legacy piecewise-linear method is kept for comparison / fallback.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ContactResult:
    index: int
    z_V: float
    defl_V: float


# ─────────────────────────────────────────────────────────────────────────
#  Baseline-deviation  (recommended)
# ─────────────────────────────────────────────────────────────────────────

def _baseline_deviation(z_V: np.ndarray, defl_V: np.ndarray,
                        direction: str = "approach",
                        baseline_frac: float = 0.10,
                        n_sigma: float = 5.0,
                        n_consec: int = 10) -> ContactResult:
    """
    Baseline-extrapolation contact-point detection.

    Parameters
    ----------
    direction : 'approach' or 'retract'
        * approach – baseline is at the START, walk forward.
        * retract  – baseline is at the END, walk backward.
    baseline_frac : float
        Fraction of the curve used to define the baseline.
    n_sigma : float
        Deviation threshold in units of baseline noise.
    n_consec : int
        Number of consecutive above-threshold points required
        to declare contact onset (rejects isolated noise spikes).
    """
    n = len(z_V)

    # ── define baseline segment ──────────────────────────────────────
    if direction == "approach":
        bl_end = max(int(n * baseline_frac), 30)
        bl_z, bl_d = z_V[:bl_end], defl_V[:bl_end]
    else:
        bl_start = min(int(n * (1 - baseline_frac)), n - 30)
        bl_z, bl_d = z_V[bl_start:], defl_V[bl_start:]

    # ── fit & extrapolate ────────────────────────────────────────────
    coeffs = np.polyfit(bl_z, bl_d, 1)
    bl_pred = np.polyval(coeffs, z_V)

    # noise = std of baseline residuals
    if direction == "approach":
        noise = float(np.std(defl_V[:bl_end] - bl_pred[:bl_end]))
    else:
        noise = float(np.std(defl_V[bl_start:] - bl_pred[bl_start:]))

    threshold = n_sigma * noise
    deviation = np.abs(defl_V - bl_pred)

    # ── walk toward contact ──────────────────────────────────────────
    if direction == "approach":
        for i in range(bl_end, n - n_consec):
            if all(deviation[i + j] > threshold for j in range(n_consec)):
                return ContactResult(i, float(z_V[i]), float(defl_V[i]))
        # fallback
        return ContactResult(bl_end, float(z_V[bl_end]),
                             float(defl_V[bl_end]))
    else:
        bl_edge = min(int(n * (1 - baseline_frac)), n - 30)
        for i in range(bl_edge, n_consec, -1):
            if all(deviation[i - j] > threshold for j in range(n_consec)):
                return ContactResult(i, float(z_V[i]), float(defl_V[i]))
        return ContactResult(bl_edge, float(z_V[bl_edge]),
                             float(defl_V[bl_edge]))


def estimate_contact_approach(z_V: np.ndarray,
                              defl_V: np.ndarray) -> ContactResult:
    """Baseline-deviation CP for approach curves."""
    return _baseline_deviation(z_V, defl_V, direction="approach",
                               baseline_frac=0.10, n_sigma=5.0,
                               n_consec=10)


def estimate_contact_retract(z_V: np.ndarray,
                             defl_V: np.ndarray) -> ContactResult:
    """Baseline-deviation CP for retract curves."""
    return _baseline_deviation(z_V, defl_V, direction="retract",
                               baseline_frac=0.10, n_sigma=5.0,
                               n_consec=10)


# ─────────────────────────────────────────────────────────────────────────
#  Piecewise-linear  (legacy / comparison)
# ─────────────────────────────────────────────────────────────────────────

def _residual_two_lines(x: np.ndarray, y: np.ndarray, split: int) -> float:
    if split < 2 or split > len(x) - 2:
        return np.inf
    xl, yl = x[:split], y[:split]
    cl = np.polyfit(xl, yl, 1)
    rl = np.sum((yl - np.polyval(cl, xl)) ** 2)
    xr, yr = x[split:], y[split:]
    cr = np.polyfit(xr, yr, 1)
    rr = np.sum((yr - np.polyval(cr, xr)) ** 2)
    return rl + rr


def find_contact_piecewise(z_V: np.ndarray, defl_V: np.ndarray,
                           search_range: tuple = (0.05, 0.95),
                           n_coarse: int = 200,
                           n_fine: int = 100) -> ContactResult:
    """Two-segment piecewise-linear fit (kept for comparison)."""
    n = len(z_V)
    lo = max(int(n * search_range[0]), 10)
    hi = min(int(n * search_range[1]), n - 10)
    candidates = np.linspace(lo, hi, n_coarse, dtype=int)
    residuals = np.array([_residual_two_lines(z_V, defl_V, c)
                          for c in candidates])
    best_coarse = candidates[np.argmin(residuals)]
    half = max(int((hi - lo) / n_coarse), 20)
    fine_lo = max(best_coarse - half, lo)
    fine_hi = min(best_coarse + half, hi)
    fine_cands = np.linspace(fine_lo, fine_hi, n_fine, dtype=int)
    fine_res = np.array([_residual_two_lines(z_V, defl_V, c)
                         for c in fine_cands])
    best = fine_cands[np.argmin(fine_res)]
    return ContactResult(int(best), float(z_V[best]), float(defl_V[best]))


# ─────────────────────────────────────────────────────────────────────────
#  Linear fits
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class LinearFitResult:
    slope: float
    intercept: float
    invols_nm_per_V: float
    x_fit: np.ndarray
    y_fit: np.ndarray


def fit_contact_region(z_V: np.ndarray, defl_V: np.ndarray,
                       idx_start: int, idx_end: int,
                       z_scale: float = 30.0) -> LinearFitResult:
    s = max(0, idx_start)
    e = min(len(z_V), idx_end)
    z_seg = z_V[s:e]
    d_seg = defl_V[s:e]
    if len(z_seg) < 5:
        return LinearFitResult(0, 0, 0, z_seg, d_seg)
    coeffs = np.polyfit(z_seg, d_seg, 1)
    slope, intercept = coeffs[0], coeffs[1]
    y_fit = np.polyval(coeffs, z_seg)
    invols = abs(z_scale * 1000.0 / slope) if abs(slope) > 1e-12 else 0
    return LinearFitResult(slope, intercept, invols, z_seg, y_fit)


def compute_phase_angle(slope_app: float, slope_ret: float) -> float:
    v1 = np.array([1.0, slope_app])
    v2 = np.array([1.0, slope_ret])
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return float(np.degrees(np.arccos(cos_t)))
