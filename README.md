# AFM Force-Curve INVOLS Calibration Tool

A Streamlit application for visualising AFM (Atomic Force Microscopy) force curves and determining the **Inverse Optical Lever Sensitivity (INVOLS)** from hard-surface calibration measurements.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-1.30%2B-red)

---

## What is INVOLS?

The Optical Beam Deflection (OBD) system in an AFM measures cantilever bending as a **voltage** on a position-sensitive photodetector. To convert that voltage into a physical deflection (nm), you need the **Inverse Optical Lever Sensitivity**:

```
deflection (nm) = INVOLS (nm/V) × detector signal (V)
```

INVOLS is determined by pressing the cantilever against an **infinitely stiff** (hard) surface — in that regime every nanometre the Z-stage moves produces exactly one nanometre of cantilever deflection. The slope of the deflection-vs-Z curve in the contact region directly gives the conversion factor.

### Why do approach and retract INVOLS differ?

On a real measurement you will typically observe a small difference between approach and retract slopes.  The main causes are:

| Source | Effect |
|--------|--------|
| **Piezo hysteresis & creep** | The Z-stage uses a piezoelectric actuator whose extension-vs-voltage curve has inherent hysteresis. The stage traces a slightly different path on approach vs retract, so the *effective* Z-scale differs between the two directions. |
| **Viscous drag on the cantilever** | If the measurement is performed in liquid or at high speed in air, viscous drag adds a velocity-dependent force that shifts the apparent deflection — in opposite directions for approach (moving toward the surface) and retract (moving away). |
| **Detector non-linearity** | The photodetector's voltage-to-angle response is linear only near the centre; at the extremes of the contact region the sensitivity can drop slightly. Because approach and retract traverse this non-linear range in opposite order, the fitted slopes can differ. |
| **Surface adhesion / meniscus** | Even on a "hard" surface, capillary forces or adhesion create an extra load during retract that is absent on approach, subtly tilting the retract slope. |
| **Thermal drift** | Slow drift in the optical alignment or Z-stage temperature between approach and retract changes the effective INVOLS. |

**Best practice:** average the approach and retract values (the default in this tool), which cancels most symmetric errors.  If the difference exceeds ~10 %, check for excessive speed, liquid drag, or piezo non-linearity.

---

## Installation

```bash
git clone https://github.com/<your-user>/afm-invols-tool.git
cd afm-invols-tool
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.9
- `streamlit`, `numpy`, `plotly`, `matplotlib`

---

## Quick Start

```bash
streamlit run app.py
```

In the sidebar, set **Root data directory** to the top-level folder containing your measurements.

---

## Expected Directory Layout

```
root/
├── 2024-09-15/                  ← date
│   └── glass_substrate/         ← sample
│       └── 155753/              ← time
│           ├── config.txt       ← measurement parameters
│           └── ForceCurve/      ← force-curve directory
│               ├── ForceCurve_00000.lvm
│               ├── ForceCurve_00001.lvm
│               └── ...
└── 2024-09-16/
    └── ...
```

The app recursively searches for `ForceCurve_*.lvm` files and builds cascading dropdown menus from however many directory levels exist. Names are arbitrary — "Date / Sample / Time" are just labels.

### LVM File Format

Each `.lvm` file is a single-column text file with `4 × N` rows (`N` = `num_app` from `config.txt`, default 12 000):

| Row range | Content |
|-----------|---------|
| `1 .. N` | Approach deflection (V) |
| `N+1 .. 2N` | Retract deflection (V) |
| `2N+1 .. 3N` | Approach Z-stage (V) |
| `3N+1 .. 4N` | Retract Z-stage (V) |

> **Important:** the file-level split at `N` does *not* always coincide with the physical turnaround of the Z-stage.  The app concatenates both halves and splits at the actual peak of Z.

### config.txt

Comma-separated key-value pairs:

| Key | Meaning | Default |
|-----|---------|---------|
| `num_app` | Points per approach | 12 000 |
| `num_ret` | Points per retract | 12 000 |
| `Vtrig` | Trigger voltage (V) | 0.4 |
| `app_speed_ratio` | Approach speed ratio | 1.5 |
| `ret_speed_ratio` | Retract speed ratio | 1.5 |

---

## Features

### Three Interactive Plots (Plotly)

1. **Deflection (V) vs Z** — full approach + retract with contact-point markers and turnaround diamond.
2. **Deflection (V) vs Z — contact region** — cropped from each contact point, scatter + solid fit lines.
3. **F (nN) vs δ (nm)** — force–indentation, computed from INVOLS and spring constant k.

All three respect the sidebar **Show curves** filter (Both / Approach / Retract).

### Cantilever Presets

Built-in presets for AC40, AC160, and AC240.  An expandable panel lets you override k, half-angle (default 17.5°), and resonance frequency.

### INVOLS Determination

The INVOLS is auto-computed from the linear fit slope in the contact region. A dropdown selects the source: **Both (average)**, Approach, or Retract.  The value auto-updates when you change the source or the contact-point offset.  You can still override it manually.

### Contact-Point Detection

Threshold-free piecewise-linear (two-segment) fit: the algorithm sweeps a split index through the curve, fits independent lines to the baseline and contact segments, and picks the split that minimises the total residual.  Manual offset sliders let you refine the result.

### Publication Export

One-click export of a three-panel `(a) (b) (c)` matplotlib figure at 300 dpi in PDF, PNG, or SVG — sized for journal two-column width (180 mm).

---

## Project Structure

```
afm-invols-tool/
├── app.py              ← Streamlit UI
├── io_utils.py         ← LVM I/O, config parsing, directory discovery
├── contact_point.py    ← Contact-point detection, linear fits, phase angle
├── export_plots.py     ← Publication-quality matplotlib export
├── notebook.ipynb      ← Step-by-step data-reduction tutorial
├── requirements.txt
└── README.md
```

---

## Notebook

`notebook.ipynb` walks through the full data-reduction pipeline step by step:

1. Reading the raw LVM file
2. Concatenation and turnaround detection
3. Contact-point estimation (piecewise-linear fit)
4. INVOLS determination from the contact slope
5. Force–indentation conversion
6. Discussion of approach vs retract INVOLS differences

---

## License

MIT

---

## Contributing

Pull requests welcome.  Please keep the codebase simple — the target user is an experimentalist who wants a quick, reliable INVOLS calibration from force-curve data.
