"""
contact_point.py – Contact-point detection and linear fits for AFM force curves.

Piecewise-linear (two-segment) fit: sweep a split index, fit independent
lines to baseline and contact segments, pick the split that minimises
the total sum-of-squared residuals.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class ContactResult:
    index: int
    z_V: float
    defl_V: float


# ── Piecewise-linear fit ─────────────────────────────────────────────────────

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
                           search_range: tuple = (0.15, 0.95),
                           n_coarse: int = 150, n_fine: int = 80,
                           ) -> ContactResult:
    n = len(z_V)
    lo = max(int(n * search_range[0]), 10)
    hi = min(int(n * search_range[1]), n - 10)
    candidates = np.linspace(lo, hi, n_coarse, dtype=int)
    residuals = np.array([_residual_two_lines(z_V, defl_V, c) for c in candidates])
    best_coarse = candidates[np.argmin(residuals)]
    half = max(int((hi - lo) / n_coarse), 20)
    fine_lo = max(best_coarse - half, lo)
    fine_hi = min(best_coarse + half, hi)
    fine_candidates = np.linspace(fine_lo, fine_hi, n_fine, dtype=int)
    fine_residuals = np.array([_residual_two_lines(z_V, defl_V, c)
                               for c in fine_candidates])
    best = fine_candidates[np.argmin(fine_residuals)]
    return ContactResult(int(best), float(z_V[best]), float(defl_V[best]))


def estimate_contact_approach(z_V: np.ndarray, defl_V: np.ndarray) -> ContactResult:
    return find_contact_piecewise(z_V, defl_V, search_range=(0.50, 0.995))


def estimate_contact_retract(z_V: np.ndarray, defl_V: np.ndarray) -> ContactResult:
    return find_contact_piecewise(z_V, defl_V, search_range=(0.005, 0.50))


# ── Linear fits ──────────────────────────────────────────────────────────────

@dataclass
class LinearFitResult:
    slope: float           # V_defl / V_z
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
    slope = coeffs[0]
    intercept = coeffs[1]
    y_fit = np.polyval(coeffs, z_seg)
    invols = abs(z_scale * 1000.0 / slope) if abs(slope) > 1e-12 else 0
    return LinearFitResult(slope, intercept, invols, z_seg, y_fit)


def compute_phase_angle(slope_app: float, slope_ret: float) -> float:
    v1 = np.array([1.0, slope_app])
    v2 = np.array([1.0, slope_ret])
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1, 1)
    return float(np.degrees(np.arccos(cos_t)))
