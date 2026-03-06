"""
contact_point.py – Contact-point detection and linear fits for AFM force curves.

Primary method: **auto-baseline smoothed-gradient**.

The algorithm automatically locates the baseline *anywhere* in the curve
(start, end, or middle) by finding the region with the lowest gradient
variance, then walks toward the contact side.  This means:

  • It does NOT assume where the baseline is.
  • It works equally well whether contact is at the start, end, or
    middle of the data array.
  • It is immune to linear baseline drift (the derivative cancels any
    constant slope).

Steps
-----
1. Smooth deflection with a box kernel (width ≈ N/60).
2. Compute per-sample gradient  g[i] = d_smooth[i+1] − d_smooth[i].
3. Slide a window (10 % of curve) across g and find the position with
   **minimum variance** → that window is the baseline.
4. Compare the **peak |gradient|** on each side of the baseline.
   The side with the larger peak is where contact is.
5. Walk from the baseline toward that side and flag the first index
   where |g| exceeds  mean_bl ± n_σ · σ_bl  for  n_consec consecutive
   points.

Legacy methods (baseline-deviation, piecewise-linear) are kept for
comparison in the notebook.
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
#  Auto-baseline smoothed-gradient  (recommended)
# ─────────────────────────────────────────────────────────────────────────

def _find_baseline_region(grad: np.ndarray, scan_frac: float = 0.10,
                          n_scan: int = 200) -> tuple[int, int]:
    """Return (start, end) of the lowest-variance window in *grad*."""
    ng = len(grad)
    scan_win = max(int(ng * scan_frac), 50)
    step = max(1, (ng - scan_win) // n_scan)
    best_var, best_start = np.inf, 0
    for start in range(0, ng - scan_win, step):
        v = float(np.var(grad[start:start + scan_win]))
        if v < best_var:
            best_var = v
            best_start = start
    return best_start, best_start + scan_win


def _auto_gradient_cp(defl_V: np.ndarray,
                      direction: str | None = None,
                      n_sigma: float = 5.0) -> int:
    """Core auto-baseline gradient CP.  Returns index into *defl_V*.

    Parameters
    ----------
    direction : 'approach', 'retract', or None
        Hint about which side of the baseline contact is on.
        • 'approach' → contact expected at HIGHER indices.
        • 'retract'  → contact expected at LOWER indices.
        • None → auto-detect by comparing peak gradient on each side.
    """
    n = len(defl_V)
    win_smooth = max(n // 60, 20)
    kernel = np.ones(win_smooth) / win_smooth
    d_smooth = np.convolve(defl_V, kernel, mode="same")
    grad = np.diff(d_smooth)
    ng = len(grad)

    # ── locate baseline ──────────────────────────────────────────────
    bl_s, bl_e = _find_baseline_region(grad)
    bl_mean = float(np.mean(grad[bl_s:bl_e]))
    bl_std = float(np.std(grad[bl_s:bl_e]))
    if bl_std < 1e-15:
        bl_std = 1e-15

    # ── decide contact side ──────────────────────────────────────────
    if direction is None:
        peak_left = float(np.max(np.abs(grad[:bl_s]))) if bl_s > 0 else 0.0
        peak_right = float(np.max(np.abs(grad[bl_e:]))) if bl_e < ng else 0.0
        direction = "approach" if peak_right >= peak_left else "retract"

    # ── walk toward contact ──────────────────────────────────────────
    nc = max(win_smooth // 3, 5)

    if direction == "approach":
        thresh = bl_mean + n_sigma * bl_std
        for i in range(bl_e, ng - nc):
            if all(grad[i + j] > thresh for j in range(nc)):
                return i
        return ng - 1
    else:
        thresh_mag = abs(bl_mean) + n_sigma * bl_std
        for i in range(bl_s, nc, -1):
            if all(abs(grad[i - j]) > thresh_mag for j in range(nc)):
                return i
        return 0


def estimate_contact_approach(z_V: np.ndarray,
                              defl_V: np.ndarray) -> ContactResult:
    """Auto-baseline gradient CP for approach."""
    idx = _auto_gradient_cp(defl_V, direction="approach")
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


def estimate_contact_retract(z_V: np.ndarray,
                             defl_V: np.ndarray) -> ContactResult:
    """Auto-baseline gradient CP for retract."""
    idx = _auto_gradient_cp(defl_V, direction="retract")
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


def estimate_contact_auto(z_V: np.ndarray,
                          defl_V: np.ndarray) -> ContactResult:
    """Fully automatic CP — no direction hint needed."""
    idx = _auto_gradient_cp(defl_V, direction=None)
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


# ─────────────────────────────────────────────────────────────────────────
#  Baseline-deviation  (legacy — fails on sloped baselines)
# ─────────────────────────────────────────────────────────────────────────

def _baseline_deviation(z_V, defl_V, direction="approach",
                        baseline_frac=0.10, n_sigma=5.0,
                        n_consec=10) -> ContactResult:
    n = len(z_V)
    if direction == "approach":
        bl_end = max(int(n * baseline_frac), 30)
        bl_z, bl_d = z_V[:bl_end], defl_V[:bl_end]
    else:
        bl_start = min(int(n * (1 - baseline_frac)), n - 30)
        bl_z, bl_d = z_V[bl_start:], defl_V[bl_start:]
    coeffs = np.polyfit(bl_z, bl_d, 1)
    bl_pred = np.polyval(coeffs, z_V)
    if direction == "approach":
        noise = float(np.std(defl_V[:bl_end] - bl_pred[:bl_end]))
    else:
        noise = float(np.std(defl_V[bl_start:] - bl_pred[bl_start:]))
    threshold = n_sigma * noise
    deviation = np.abs(defl_V - bl_pred)
    if direction == "approach":
        for i in range(int(n * baseline_frac), n - n_consec):
            if all(deviation[i + j] > threshold for j in range(n_consec)):
                return ContactResult(i, float(z_V[i]), float(defl_V[i]))
        fb = int(n * baseline_frac)
        return ContactResult(fb, float(z_V[fb]), float(defl_V[fb]))
    else:
        bl_edge = min(int(n * (1 - baseline_frac)), n - 30)
        for i in range(bl_edge, n_consec, -1):
            if all(deviation[i - j] > threshold for j in range(n_consec)):
                return ContactResult(i, float(z_V[i]), float(defl_V[i]))
        return ContactResult(bl_edge, float(z_V[bl_edge]),
                             float(defl_V[bl_edge]))


# ─────────────────────────────────────────────────────────────────────────
#  Piecewise-linear  (legacy — length-biased)
# ─────────────────────────────────────────────────────────────────────────

def _residual_two_lines(x, y, split):
    if split < 2 or split > len(x) - 2:
        return np.inf
    cl = np.polyfit(x[:split], y[:split], 1)
    rl = np.sum((y[:split] - np.polyval(cl, x[:split])) ** 2)
    cr = np.polyfit(x[split:], y[split:], 1)
    rr = np.sum((y[split:] - np.polyval(cr, x[split:])) ** 2)
    return rl + rr


def find_contact_piecewise(z_V, defl_V, search_range=(0.05, 0.95),
                           n_coarse=200, n_fine=100) -> ContactResult:
    n = len(z_V)
    lo, hi = max(int(n * search_range[0]), 10), min(int(n * search_range[1]), n - 10)
    cands = np.linspace(lo, hi, n_coarse, dtype=int)
    res = np.array([_residual_two_lines(z_V, defl_V, c) for c in cands])
    bc = cands[np.argmin(res)]
    half = max(int((hi - lo) / n_coarse), 20)
    fc = np.linspace(max(bc - half, lo), min(bc + half, hi), n_fine, dtype=int)
    fr = np.array([_residual_two_lines(z_V, defl_V, c) for c in fc])
    best = fc[np.argmin(fr)]
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


def fit_contact_region(z_V, defl_V, idx_start, idx_end,
                       z_scale=30.0) -> LinearFitResult:
    s, e = max(0, idx_start), min(len(z_V), idx_end)
    z_seg, d_seg = z_V[s:e], defl_V[s:e]
    if len(z_seg) < 5:
        return LinearFitResult(0, 0, 0, z_seg, d_seg)
    coeffs = np.polyfit(z_seg, d_seg, 1)
    slope, intercept = coeffs
    y_fit = np.polyval(coeffs, z_seg)
    invols = abs(z_scale * 1000.0 / slope) if abs(slope) > 1e-12 else 0
    return LinearFitResult(slope, intercept, invols, z_seg, y_fit)


def compute_phase_angle(slope_app, slope_ret):
    v1 = np.array([1.0, slope_app])
    v2 = np.array([1.0, slope_ret])
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos_t, -1, 1))))
