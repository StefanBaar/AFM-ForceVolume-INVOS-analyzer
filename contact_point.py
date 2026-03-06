"""
contact_point.py – Contact-point detection and linear fits for AFM force curves.

Two-stage CP detection:
  1. **Gradient method** (primary): finds the baseline as the region with
     the smallest mean |gradient|, then walks toward contact.  Fast, works
     for most curves, but fails when the contact gradient is only marginally
     above baseline noise (smooth, shallow ramps).
  2. **Chord-distance method** (fallback): draws a chord from the baseline
     midpoint to the deflection peak after baseline subtraction, then finds
     the point of maximum perpendicular distance to that chord.  This is a
     global / geometric method that handles smooth ramps robustly.

If the gradient method returns an index too close to an edge (< 3 % or
> 97 %), it is considered a failure and the chord method takes over.
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
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _smooth_gradient(defl_V: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-sample gradient of box-smoothed deflection."""
    n = len(defl_V)
    win = max(n // 60, 20)
    kernel = np.ones(win) / win
    d_smooth = np.convolve(defl_V, kernel, mode="same")
    return np.diff(d_smooth), win


def _find_baseline_region(grad: np.ndarray,
                          scan_frac: float = 0.10,
                          n_scan: int = 200) -> tuple[int, int]:
    """Lowest mean-|gradient| window = baseline."""
    ng = len(grad)
    scan_win = max(int(ng * scan_frac), 50)
    step = max(1, (ng - scan_win) // n_scan)
    best_score, best_start = np.inf, 0
    for start in range(0, ng - scan_win, step):
        score = float(np.mean(np.abs(grad[start:start + scan_win])))
        if score < best_score:
            best_score = score
            best_start = start
    return best_start, best_start + scan_win


# ─────────────────────────────────────────────────────────────────────────
#  Gradient method
# ─────────────────────────────────────────────────────────────────────────

def _gradient_cp(defl_V: np.ndarray,
                 direction: str | None = None,
                 n_sigma: float = 5.0) -> int:
    """Auto-baseline gradient CP.  Returns index into *defl_V*."""
    n = len(defl_V)
    grad, win = _smooth_gradient(defl_V)
    ng = len(grad)

    bl_s, bl_e = _find_baseline_region(grad)
    bl_mean = float(np.mean(grad[bl_s:bl_e]))
    bl_std = float(np.std(grad[bl_s:bl_e]))
    if bl_std < 1e-15:
        bl_std = 1e-15

    if direction is None:
        peak_left = float(np.max(np.abs(grad[:bl_s]))) if bl_s > 0 else 0.0
        peak_right = float(np.max(np.abs(grad[bl_e:]))) if bl_e < ng else 0.0
        direction = "approach" if peak_right >= peak_left else "retract"

    nc = max(win // 3, 5)

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


# ─────────────────────────────────────────────────────────────────────────
#  Chord-distance method
# ─────────────────────────────────────────────────────────────────────────

def _chord_cp(defl_V: np.ndarray, z_V: np.ndarray,
              direction: str | None = None) -> int:
    """Perpendicular-distance-to-chord CP.

    After baseline subtraction, draw a chord from the baseline centre to
    the deflection peak.  The CP is the point of maximum perpendicular
    distance **near the baseline end** of the chord — i.e. where the curve
    bends away from the straight line as it transitions from contact to
    baseline.

    The ``direction`` hint restricts the search to the baseline-adjacent
    half of the peak→baseline span so that curvature artefacts at the
    high-deflection end do not produce a false maximum.
    """
    n = len(z_V)
    if n < 50:
        return n // 2

    grad, _ = _smooth_gradient(defl_V)
    bl_s, bl_e = _find_baseline_region(grad)

    bl_s_d = max(0, min(bl_s, n - 2))
    bl_e_d = max(bl_s_d + 2, min(bl_e, n))
    if bl_e_d - bl_s_d < 5:
        chunk = max(n // 10, 20)
        best_s, best_std = 0, np.inf
        for s in range(0, n - chunk, max(1, n // 50)):
            v = float(np.std(defl_V[s:s + chunk]))
            if v < best_std:
                best_std = v
                best_s = s
        bl_s_d, bl_e_d = best_s, best_s + chunk

    coeffs = np.polyfit(z_V[bl_s_d:bl_e_d], defl_V[bl_s_d:bl_e_d], 1)
    d_corr = defl_V - np.polyval(coeffs, z_V)

    bl_mid = (bl_s_d + bl_e_d) // 2
    peak = int(np.argmax(np.abs(d_corr)))
    if peak == bl_mid:
        peak = 0 if bl_mid > n // 2 else n - 1

    x0, y0 = z_V[bl_mid], d_corr[bl_mid]
    x1, y1 = z_V[peak], d_corr[peak]

    dx, dy = x1 - x0, y1 - y0
    denom = np.sqrt(dx ** 2 + dy ** 2)
    if denom < 1e-15:
        return n // 2

    dist = np.abs(dx * (d_corr - y0) - dy * (z_V - x0)) / denom

    # ── restrict search to the baseline-adjacent half ────────────────
    # The CP is where the curve transitions from contact to baseline,
    # which is always nearer to the baseline end of the chord.
    skip = max(40, n // 200)
    if peak < bl_mid:
        # baseline at high index, peak at low index (typical retract)
        span_mid = (peak + bl_mid) // 2
        lo, hi = max(span_mid, peak + skip), bl_mid
    else:
        # baseline at low index, peak at high index (typical approach)
        span_mid = (bl_mid + peak) // 2
        lo, hi = bl_mid + skip, min(span_mid, peak)

    lo = max(0, min(lo, n - 2))
    hi = max(lo + 1, min(hi, n))
    if hi - lo < 10:
        lo, hi = skip, max(n - skip, skip + 10)

    return int(np.argmax(dist[lo:hi]) + lo)


# ─────────────────────────────────────────────────────────────────────────
#  Combined: gradient + chord fallback
# ─────────────────────────────────────────────────────────────────────────

def _combined_cp(z_V: np.ndarray, defl_V: np.ndarray,
                 direction: str | None = None) -> int:
    """Run both gradient and chord, pick the one whose deflection is
    closest to the baseline level.

    Rationale: the true CP is at the transition between contact and
    baseline, so its deflection should be near baseline.  Whichever
    method gives a CP closer to baseline is more likely correct.
    """
    n = len(z_V)
    if n < 50:
        return n // 2

    # Baseline deflection (from gradient-found baseline region)
    grad, _ = _smooth_gradient(defl_V)
    bl_s, bl_e = _find_baseline_region(grad)
    bl_s_c = max(0, min(bl_s, n - 1))
    bl_e_c = min(n, max(bl_e, bl_s_c + 1))
    bl_defl = float(np.median(defl_V[bl_s_c:bl_e_c]))

    # Method 1: gradient
    idx_g = _gradient_cp(defl_V, direction=direction)

    # Method 2: chord
    idx_c = _chord_cp(defl_V, z_V, direction=direction)

    # Clamp
    idx_g = max(0, min(idx_g, n - 1))
    idx_c = max(0, min(idx_c, n - 1))

    # Pick the one closer to baseline deflection
    err_g = abs(defl_V[idx_g] - bl_defl)
    err_c = abs(defl_V[idx_c] - bl_defl)

    return idx_g if err_g <= err_c else idx_c


def estimate_contact_approach(z_V: np.ndarray,
                              defl_V: np.ndarray) -> ContactResult:
    idx = _combined_cp(z_V, defl_V, direction="approach")
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


def estimate_contact_retract(z_V: np.ndarray,
                             defl_V: np.ndarray) -> ContactResult:
    idx = _combined_cp(z_V, defl_V, direction="retract")
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


def estimate_contact_auto(z_V: np.ndarray,
                          defl_V: np.ndarray) -> ContactResult:
    idx = _combined_cp(z_V, defl_V, direction=None)
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


# ─────────────────────────────────────────────────────────────────────────
#  Standalone chord (exported for notebook / comparison)
# ─────────────────────────────────────────────────────────────────────────

def chord_distance_cp(z_V, defl_V, direction="approach") -> ContactResult:
    idx = _chord_cp(defl_V, z_V, direction)
    return ContactResult(idx, float(z_V[idx]), float(defl_V[idx]))


# ─────────────────────────────────────────────────────────────────────────
#  Legacy: baseline-deviation
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
#  Legacy: piecewise-linear
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
    lo = max(int(n * search_range[0]), 10)
    hi = min(int(n * search_range[1]), n - 10)
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
