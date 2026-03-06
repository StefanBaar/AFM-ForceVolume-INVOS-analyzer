"""
app.py – Streamlit AFM Force-Curve Viewer

Launch:  streamlit run app.py
"""
from __future__ import annotations
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from collections import OrderedDict
import tempfile

from io_utils import (
    FCConfig, discover_datasets, list_force_curves,
    read_lvm_raw, detect_turnaround, split_at_turnaround,
    build_directory_tree,
)
from contact_point import (
    estimate_contact_approach, estimate_contact_retract,
    fit_contact_region, compute_phase_angle,
)
from export_plots import export_figure, ExportData

# ─── Global appearance ───────────────────────────────────────────────────────
# Change this single value to adjust transparency of ALL data-point markers
ALPHA = 0.8

# ─── Cantilever presets ──────────────────────────────────────────────────────
CANTILEVERS = {
    "AC40":  {"k": 0.09,  "f0_kHz": 32,  "half_angle_deg": 17.5},
    "AC160": {"k": 26.0,  "f0_kHz": 300, "half_angle_deg": 17.5},
    "AC240": {"k": 2.0,   "f0_kHz": 70,  "half_angle_deg": 17.5},
}

NEON_BLUE      = "#00e5ff"
NEON_PINK      = "#ff10f0"
FIT_BLUE       = "#0099aa"
FIT_PINK       = "#aa0ea0"
NEON_GREEN     = "#39ff14"
FIT_GREEN      = "#28b30e"

default_path = str(Path.cwd())

st.set_page_config(page_title="AFM Force-Curve Viewer", layout="wide")

# ─── Neon CSS theme ──────────────────────────────────────────────────────────
st.markdown(f"""
<style>
div[data-testid="stSlider"] > div > div > div > div {{
    background: linear-gradient(90deg, {NEON_BLUE}, {NEON_PINK}) !important;
}}
div[data-testid="stSlider"] div[role="slider"] {{
    background-color: {NEON_PINK} !important;
    border: 2px solid {NEON_BLUE} !important;
    box-shadow: 0 0 8px {NEON_PINK}88, 0 0 16px {NEON_BLUE}44;
}}
div[data-testid="stSlider"] div[role="slider"]:hover {{
    box-shadow: 0 0 14px {NEON_PINK}cc, 0 0 28px {NEON_BLUE}88;
}}
div.stButton > button,
div[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, {NEON_BLUE}22, {NEON_PINK}22) !important;
    border: 1.5px solid {NEON_BLUE} !important;
    color: {NEON_BLUE} !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    text-shadow: 0 0 6px {NEON_BLUE}66;
}}
div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {{
    background: linear-gradient(135deg, {NEON_BLUE}44, {NEON_PINK}44) !important;
    border-color: {NEON_PINK} !important;
    color: {NEON_PINK} !important;
    box-shadow: 0 0 12px {NEON_PINK}66, 0 0 24px {NEON_BLUE}33;
    text-shadow: 0 0 8px {NEON_PINK}88;
}}
div[data-testid="stToggle"] label span[data-testid="stToggleSwitch"] > span {{
    background-color: {NEON_BLUE} !important;
    box-shadow: 0 0 6px {NEON_BLUE}88;
}}
div[data-testid="stRadio"] label span[data-checked="true"]::before {{
    background-color: {NEON_PINK} !important;
}}
div[role="radiogroup"] label[data-checked="true"] {{
    color: {NEON_PINK} !important;
}}
div[data-testid="stSelectbox"] > div > div {{
    border-color: {NEON_BLUE}66 !important;
}}
div[data-testid="stSelectbox"] > div > div:focus-within {{
    border-color: {NEON_PINK} !important;
    box-shadow: 0 0 8px {NEON_PINK}44;
}}
div[data-testid="stNumberInput"] input {{
    border-color: {NEON_BLUE}44 !important;
}}
div[data-testid="stNumberInput"] input:focus {{
    border-color: {NEON_PINK} !important;
    box-shadow: 0 0 6px {NEON_PINK}44;
}}
div[data-testid="stCheckbox"] label span[aria-checked="true"] {{
    background-color: {NEON_PINK} !important;
    border-color: {NEON_PINK} !important;
}}
details[data-testid="stExpander"] summary {{
    border-left: 3px solid {NEON_BLUE}66;
    padding-left: 8px;
}}
details[data-testid="stExpander"][open] summary {{
    border-left-color: {NEON_PINK};
}}
div[data-testid="stMetric"] label {{
    color: {NEON_BLUE} !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
    color: #fafafa !important;
    text-shadow: 0 0 4px {NEON_BLUE}44;
}}
section[data-testid="stSidebar"] h2 {{
    color: {NEON_BLUE} !important;
    font-size: 0.9rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid {NEON_BLUE}33;
    padding-bottom: 4px;
}}
h1 {{
    background: linear-gradient(90deg, {NEON_BLUE}, {NEON_PINK});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
</style>
""", unsafe_allow_html=True)

st.title("AFM Force-Curve Viewer")

# =====================================================================
#  SIDEBAR — Data, navigation, cantilever
# =====================================================================
st.sidebar.header("Data Location")
data_root = st.sidebar.text_input("Root data directory", value=default_path)
root_path = Path(data_root).expanduser().resolve()
if not root_path.is_dir():
    st.error(f"Directory not found: `{root_path}`"); st.stop()

with st.sidebar.expander("Data Path"):
    dataset_dirs = discover_datasets(root_path)
    if not dataset_dirs:
        st.warning(f"No `ForceCurve_*.lvm` found under `{root_path}`."); st.stop()
    entries, max_depth = build_directory_tree(dataset_dirs, root_path)
    st.header("Dataset")
    level_hints = ["Date", "Sample", "Time", "Folder", "Sub-folder"]
    filtered_entries = entries
    for level in range(max_depth):
        options = list(OrderedDict.fromkeys(
            parts[level] for parts, _ in filtered_entries if len(parts) > level))
        if not options: break
        label = level_hints[level] if level < len(level_hints) else f"Level {level}"
        chosen = st.selectbox(label, options, index=len(options) - 1,
                              key=f"level_{level}")
        filtered_entries = [(p, fp) for p, fp in filtered_entries
                            if len(p) > level and p[level] == chosen]
    if not filtered_entries:
        st.error("No matching dataset."); st.stop()
    selected_dir = filtered_entries[0][1]
    st.caption(f"`{selected_dir}`")
    config = FCConfig()
    for cand in [selected_dir, selected_dir.parent,
                 selected_dir.parent.parent, selected_dir.parent.parent.parent]:
        if (cand / "config.txt").is_file():
            config = FCConfig.from_file(cand / "config.txt"); break
    fc_files = list_force_curves(selected_dir)
    if not fc_files:
        st.warning("No ForceCurve_*.lvm found."); st.stop()

st.sidebar.header("Navigation")
fc_idx = st.sidebar.slider("Force Curve Index", 0, len(fc_files) - 1,
                           value=len(fc_files) - 1, key="fc_slider")
st.sidebar.write(f"**{fc_files[fc_idx].name}** ({fc_idx + 1}/{len(fc_files)})")

st.sidebar.header("Cantilever")
cant_name = st.sidebar.selectbox("Model", list(CANTILEVERS.keys()), index=1,
                                 key="cant_model")
cant = CANTILEVERS[cant_name]
with st.sidebar.expander("Edit cantilever parameters"):
    k_spring = st.number_input("k (N/m)", value=cant["k"],
                                min_value=0.001, step=0.01, format="%.3f")
    half_angle = st.number_input("Half-angle (°)", value=cant["half_angle_deg"],
                                  min_value=0.1, step=0.5, format="%.1f")
    f0_kHz = st.number_input("f₀ (kHz)", value=float(cant["f0_kHz"]),
                              min_value=0.1, step=1.0, format="%.1f")

st.sidebar.header("Parameters")
z_scale = st.sidebar.number_input("Z-stage scale (µm/V)", value=config.z_scale,
                                  step=1.0, format="%.1f")
x_in_um = st.sidebar.toggle("X-axis in µm", value=False, key="x_unit")

st.sidebar.header("Display")
show_curves = st.sidebar.radio("Show curves", ["Both", "Approach", "Retract"],
                               index=0, key="show_curves")

# ── Read & split ─────────────────────────────────────────────────────────
try:
    raw = read_lvm_raw(fc_files[fc_idx], num_app=config.num_app,
                       num_ret=config.num_ret)
except Exception as e:
    st.error(f"Read error: {e}"); st.stop()

auto_turn = detect_turnaround(raw)
total_pts = len(raw.z_V)

# Adaptive slider range: ±5 % of total points, minimum ±100
turn_range = max(int(total_pts * 0.05), 100)
st.sidebar.header("Turnaround")
turn_offset = st.sidebar.slider("Offset (pts)", -turn_range, turn_range, 0,
                                key="turn_off")
turnaround = int(np.clip(auto_turn + turn_offset, 1, total_pts - 2))
st.sidebar.caption(f"Auto {auto_turn} → {turnaround}")
fc = split_at_turnaround(raw, turnaround)

# ── Contact-point detection ──────────────────────────────────────────────
cp_app = estimate_contact_approach(fc.app_z_V, fc.app_defl_V)
cp_ret = estimate_contact_retract(fc.ret_z_V, fc.ret_defl_V)

# Adaptive CP offset sliders: ±10 % of each curve, minimum ±50
cp_range_a = max(int(len(fc.app_z_V) * 0.10), 50)
cp_range_r = max(int(len(fc.ret_z_V) * 0.10), 50)
st.sidebar.header("Contact point offsets")
cp_off_a = st.sidebar.slider("Approach (pts)", -cp_range_a, cp_range_a, 0,
                             key="cp_off_a")
cp_off_r = st.sidebar.slider("Retract (pts)", -cp_range_r, cp_range_r, 0,
                             key="cp_off_r")
cp_app_idx = int(np.clip(cp_app.index + cp_off_a, 0, len(fc.app_z_V) - 1))
cp_ret_idx = int(np.clip(cp_ret.index + cp_off_r, 0, len(fc.ret_z_V) - 1))

# ── Linearity analysis (determines linear fitting range) ────────────────

def linearity_analysis(z_V, defl_V, cp_idx, z_scale_um,
                       win_frac=0.025, n_steps=300):
    contact_z = z_V[cp_idx:]
    contact_d = defl_V[cp_idx:]
    n = len(contact_z)
    win = max(int(n * win_frac), 30)
    if n < win + 20:
        return None
    step = max(1, (n - win) // n_steps)
    z_cp = contact_z[0]
    dz_arr, slopes = [], []
    for i in range(0, n - win, step):
        try:
            coeffs = np.polyfit(contact_z[i:i + win], contact_d[i:i + win], 1)
        except (np.linalg.LinAlgError, ValueError):
            continue
        mid = i + win // 2
        dz_arr.append((contact_z[mid] - z_cp) * z_scale_um * 1000)
        slopes.append(coeffs[0])
    dz_arr = np.array(dz_arr)
    slopes = np.array(slopes)
    if len(slopes) < 10:
        return None
    skip = max(2, len(slopes) // 50)
    ref_end = skip + max(len(slopes) // 8, 5)
    ref_slope = float(np.median(slopes[skip:ref_end]))
    ref_invols = abs(z_scale_um * 1000.0 / ref_slope) if abs(ref_slope) > 1e-12 else 0.0
    dev_pct = 100.0 * (slopes - ref_slope) / abs(ref_slope)
    return dict(dz=dz_arr, slope=slopes, dev_pct=dev_pct,
                ref_slope=ref_slope, ref_invols=ref_invols,
                ref_end_idx=ref_end, win=win)


def find_linear_end(dev_pct, start_idx, threshold, n_consec=4):
    n = len(dev_pct)
    for i in range(start_idx, n - n_consec):
        if all(abs(dev_pct[i + j]) > threshold for j in range(n_consec)):
            return i
    return n - 1


# Default threshold for linearity (sidebar comes later, but we need it now)
st.sidebar.header("Linearity check")
dev_threshold = st.sidebar.slider(
    "Deviation threshold (%)", 1.0, 30.0, 10.0, step=0.5,
    key="dev_thresh",
    help="Local slope must stay within ±this % of the near-CP reference.")

# Run linearity on approach
lin_app = linearity_analysis(fc.app_z_V, fc.app_defl_V, cp_app_idx, z_scale)
lin_ret = None
if show_curves in ("Both", "Retract"):
    ret_z_flip = fc.ret_z_V[:cp_ret_idx + 1][::-1]
    ret_d_flip = fc.ret_defl_V[:cp_ret_idx + 1][::-1]
    lin_ret = linearity_analysis(ret_z_flip, ret_d_flip, 0, z_scale)

# ── Linear fits (within linearity range only) ───────────────────────────
# Approach: CP → linear-end  (instead of CP → turnaround)
if lin_app is not None:
    end_a_local = find_linear_end(lin_app["dev_pct"], lin_app["ref_end_idx"],
                                  dev_threshold)
    # Convert from linearity-analysis grid index to real array index
    # Each grid step ≈ (contact_length - win) / n_steps samples
    contact_n = len(fc.app_z_V) - cp_app_idx
    lin_win = lin_app["win"]
    lin_step = max(1, (contact_n - lin_win) // 300)
    fit_end_real = cp_app_idx + int(end_a_local * lin_step + lin_win // 2)
    fit_end_real = min(fit_end_real, len(fc.app_z_V))
else:
    fit_end_real = len(fc.app_z_V)

fit_app = fit_contact_region(fc.app_z_V, fc.app_defl_V,
                             cp_app_idx, fit_end_real, z_scale)

# Retract: linear-end → turnaround  (reversed sense)
if lin_ret is not None:
    end_r_local = find_linear_end(lin_ret["dev_pct"], lin_ret["ref_end_idx"],
                                  dev_threshold)
    contact_n_r = cp_ret_idx + 1
    lin_win_r = lin_ret["win"]
    lin_step_r = max(1, (contact_n_r - lin_win_r) // 300)
    # lin_ret was computed on flipped data, so end_r_local corresponds
    # to distance FROM turnaround.  Convert to real retract index:
    fit_start_ret = int(end_r_local * lin_step_r + lin_win_r // 2)
    fit_start_ret = max(0, min(fit_start_ret, cp_ret_idx))
else:
    fit_start_ret = 0

fit_ret = fit_contact_region(fc.ret_z_V, fc.ret_defl_V,
                             fit_start_ret, cp_ret_idx, z_scale)

phase_deg = compute_phase_angle(fit_app.slope, fit_ret.slope)
cp_distance_nm = abs(fc.app_z_V[cp_app_idx] - fc.ret_z_V[cp_ret_idx]) * z_scale * 1000
invols_mean = (fit_app.invols_nm_per_V + fit_ret.invols_nm_per_V) / 2.0

# ── INVOLS (default = Approach, from linear region) ──────────────────────
st.sidebar.header("INVOLS")
st.sidebar.caption(
    f"Approach (linear): **{fit_app.invols_nm_per_V:.1f}** nm/V  \n"
    f"Retract (linear): **{fit_ret.invols_nm_per_V:.1f}** nm/V  \n"
    f"Mean: **{invols_mean:.1f}** nm/V"
)
with st.sidebar.expander("INVOLS selection"):
    invols_src = st.selectbox(
        "Source", ["Approach", "Retract", "Mean", "Custom"],
        index=0, key="invols_src",
    )
    if invols_src == "Approach":
        invols = fit_app.invols_nm_per_V
    elif invols_src == "Retract":
        invols = fit_ret.invols_nm_per_V
    elif invols_src == "Mean":
        invols = invols_mean
    else:
        invols = st.number_input(
            "Custom INVOLS (nm/V)", value=100.0,
            min_value=0.1, step=1.0, format="%.1f", key="invols_custom",
        )
st.sidebar.caption(f"**INVOLS used: {invols:.1f} nm/V**")

# ── Export format ────────────────────────────────────────────────────────
st.sidebar.header("Export")
export_fmt = st.sidebar.selectbox("Format", ["pdf", "png", "svg"], index=0,
                                  key="exp_fmt")

# =====================================================================
#  Helpers
# =====================================================================
def _x(z_V, zero_first=False):
    v = z_V * z_scale if x_in_um else z_V
    if zero_first and len(v) > 0:
        v = v - v[0]
    return v

x_label = "Z (µm)" if x_in_um else "Z (V)"
x_label_rel = "ΔZ (µm)" if x_in_um else "Z (V)"

def dec2(a, b, n=4000):
    if len(a) <= n:
        return a, b
    s = max(1, len(a) // n)
    return a[::s], b[::s]

MAX = 4000
PH = 380
dark_layout = dict(height=PH, margin=dict(l=60, r=60, t=10, b=10),
                   font=dict(color="#fafafa", size=12))

def _legend_right():
    return dict(orientation="v", yanchor="top", y=1.0,
                xanchor="left", x=1.02, font=dict(size=11))

def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# =====================================================================
#  ROW 1 — Plot 1 (full) | Plot 2 (contact region w/ linear-region fit)
# =====================================================================
col1, col2 = st.columns(2)
with col1:
    zoom_cp = st.toggle("Zoom to contact region", value=False, key="zoom_cp")

# ── PLOT 1: Full deflection vs Z (dots + vertical CP lines) ─────────
fig1 = go.Figure()
if show_curves in ("Both", "Approach"):
    ax, ay = dec2(_x(fc.app_z_V), fc.app_defl_V, MAX)
    fig1.add_trace(go.Scattergl(x=ax, y=ay, mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=2.5), name="Approach"))
    fig1.add_vline(x=float(_x(fc.app_z_V)[cp_app_idx]),
                   line_width=1.8, line_dash="dash", line_color=NEON_BLUE,
                   opacity=0.85, annotation_text="CP app",
                   annotation_position="top left",
                   annotation_font_size=10, annotation_font_color=NEON_BLUE)
if show_curves in ("Both", "Retract"):
    rx, ry = dec2(_x(fc.ret_z_V), fc.ret_defl_V, MAX)
    fig1.add_trace(go.Scattergl(x=rx, y=ry, mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=2.5), name="Retract"))
    fig1.add_vline(x=float(_x(fc.ret_z_V)[cp_ret_idx]),
                   line_width=1.8, line_dash="dash", line_color=NEON_PINK,
                   opacity=0.85, annotation_text="CP ret",
                   annotation_position="top right",
                   annotation_font_size=10, annotation_font_color=NEON_PINK)
fig1.add_trace(go.Scatter(
    x=[_x(fc.app_z_V)[-1]], y=[fc.app_defl_V[-1]],
    mode="markers", marker=dict(size=8, color="white", symbol="diamond"),
    name="Turnaround"))
if zoom_cp:
    safe_idx = max(cp_app_idx - 200, 0)
    x_lo = float(_x(fc.app_z_V)[safe_idx])
    all_x_ends = [float(_x(fc.app_z_V)[-1])]
    if show_curves in ("Both", "Retract"):
        ret_x = _x(fc.ret_z_V)
        all_x_ends.extend([float(ret_x.min()), float(ret_x.max())])
    x_hi = max(all_x_ends)
    m = 0.05 * abs(x_hi - x_lo)
    fig1.update_xaxes(range=[x_lo - m, x_hi + m])
fig1.update_layout(
    title=dict(text="<b>Deflection vs Z – Full</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title=x_label, yaxis_title="Deflection (V)",
    legend=_legend_right(), **dark_layout)
with col1:
    st.plotly_chart(fig1, use_container_width=True)

# ── PLOT 2: Contact region with linear-region fit lines ──────────────
fig2 = go.Figure()
if show_curves in ("Both", "Approach"):
    a_zc = fc.app_z_V[cp_app_idx:]
    a_dc = fc.app_defl_V[cp_app_idx:]
    a_x = _x(a_zc, zero_first=x_in_um)
    cx, cy = dec2(a_x, a_dc, MAX)
    fig2.add_trace(go.Scattergl(x=cx, y=cy, mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=3), name="Approach"))
    if len(fit_app.x_fit) > 1:
        fit_x = _x(fit_app.x_fit)
        if x_in_um and len(a_zc) > 0:
            fit_x = fit_x - _x(a_zc)[0]
        fig2.add_trace(go.Scatter(x=fit_x, y=fit_app.y_fit, mode="lines",
            line=dict(color=FIT_BLUE, width=2.5),
            name=f"Fit app (s={fit_app.slope:.1f})"))
if show_curves in ("Both", "Retract"):
    r_zc = fc.ret_z_V[:cp_ret_idx + 1]
    r_dc = fc.ret_defl_V[:cp_ret_idx + 1]
    r_x = _x(r_zc, zero_first=x_in_um)
    cx, cy = dec2(r_x, r_dc, MAX)
    fig2.add_trace(go.Scattergl(x=cx, y=cy, mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=3), name="Retract"))
    if len(fit_ret.x_fit) > 1:
        fit_x = _x(fit_ret.x_fit)
        if x_in_um and len(r_zc) > 0:
            fit_x = fit_x - _x(r_zc)[0]
        fig2.add_trace(go.Scatter(x=fit_x, y=fit_ret.y_fit, mode="lines",
            line=dict(color=FIT_PINK, width=2.5),
            name=f"Fit ret (s={fit_ret.slope:.1f})"))
fig2.update_layout(
    title=dict(text="<b>Deflection vs Z – Contact Region</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title=x_label_rel if x_in_um else x_label,
    yaxis_title="Deflection (V)",
    legend=_legend_right(), **dark_layout)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
#  ROW 2 — Plot 3: Defl(nm) vs ΔZ(nm) | Plot 4: F(nN) vs δ(nm)
# =====================================================================
z_cp = fc.app_z_V[cp_app_idx]
d_cp = fc.app_defl_V[cp_app_idx]

a_zs = fc.app_z_V[cp_app_idx:]
a_ds = fc.app_defl_V[cp_app_idx:]
a_dz_nm   = (a_zs - z_cp) * z_scale * 1000
a_defl_nm = (a_ds - d_cp) * invols
a_delta   = a_dz_nm - a_defl_nm
a_F       = k_spring * a_defl_nm

r_zs = fc.ret_z_V[:cp_ret_idx + 1]
r_ds = fc.ret_defl_V[:cp_ret_idx + 1]
r_dz_nm   = (r_zs - z_cp) * z_scale * 1000
r_defl_nm = (r_ds - d_cp) * invols
r_delta   = r_dz_nm - r_defl_nm
r_F       = k_spring * r_defl_nm

def _fit_physical(fit_x, fit_y):
    dz    = (fit_x - z_cp) * z_scale * 1000
    d     = (fit_y - d_cp) * invols
    delta = dz - d
    F     = k_spring * d
    return dz, d, delta, F

if len(fit_app.x_fit) > 1:
    fa_dz, fa_defl, fa_delta, fa_F = _fit_physical(fit_app.x_fit, fit_app.y_fit)
else:
    fa_dz = fa_defl = fa_delta = fa_F = np.array([])
if len(fit_ret.x_fit) > 1:
    fr_dz, fr_defl, fr_delta, fr_F = _fit_physical(fit_ret.x_fit, fit_ret.y_fit)
else:
    fr_dz = fr_defl = fr_delta = fr_F = np.array([])

col3, col4 = st.columns(2)

# ── PLOT 3: Deflection (nm) vs ΔZ (nm) ──────────────────────────────
fig3 = go.Figure()
if show_curves in ("Both", "Approach"):
    dx, dy = dec2(a_dz_nm, a_defl_nm, MAX)
    fig3.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=3), name="Approach"))
    if len(fa_dz) > 1:
        fig3.add_trace(go.Scatter(x=fa_dz, y=fa_defl, mode="lines",
            line=dict(color=FIT_BLUE, width=2.5), name="Fit app"))
if show_curves in ("Both", "Retract"):
    dx, dy = dec2(r_dz_nm, r_defl_nm, MAX)
    fig3.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=3), name="Retract"))
    if len(fr_dz) > 1:
        fig3.add_trace(go.Scatter(x=fr_dz, y=fr_defl, mode="lines",
            line=dict(color=FIT_PINK, width=2.5), name="Fit ret"))
all_dz = []
if show_curves in ("Both", "Approach") and len(a_dz_nm) > 0:
    all_dz.extend([float(a_dz_nm.min()), float(a_dz_nm.max())])
if show_curves in ("Both", "Retract") and len(r_dz_nm) > 0:
    all_dz.extend([float(r_dz_nm.min()), float(r_dz_nm.max())])
if all_dz:
    ref_lo, ref_hi = min(all_dz), max(all_dz)
    fig3.add_trace(go.Scatter(x=[ref_lo, ref_hi], y=[ref_lo, ref_hi],
        mode="lines", line=dict(color="rgba(255,255,255,0.3)", width=1,
        dash="dash"), name="slope = 1"))
fig3.add_hline(y=0, line_width=0.5, line_color="grey")
fig3.add_vline(x=0, line_width=0.5, line_color="grey")
fig3.update_layout(
    title=dict(text="<b>Deflection vs ΔZ (INVOLS check)</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title="ΔZ (nm)", yaxis_title="Deflection (nm)",
    legend=_legend_right(), **dark_layout)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

# ── PLOT 4: F (nN) vs δ (nm) ────────────────────────────────────────
fig4 = go.Figure()
if show_curves in ("Both", "Approach"):
    dx, dy = dec2(a_delta, a_F, MAX)
    fig4.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=3), name="Approach"))
    if len(fa_delta) > 1:
        fig4.add_trace(go.Scatter(x=fa_delta, y=fa_F, mode="lines",
            line=dict(color=FIT_BLUE, width=2.5), name="Fit app"))
if show_curves in ("Both", "Retract"):
    dx, dy = dec2(r_delta, r_F, MAX)
    fig4.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=3), name="Retract"))
    if len(fr_delta) > 1:
        fig4.add_trace(go.Scatter(x=fr_delta, y=fr_F, mode="lines",
            line=dict(color=FIT_PINK, width=2.5), name="Fit ret"))
fig4.add_hline(y=0, line_width=0.5, line_color="grey")
fig4.add_vline(x=0, line_width=0.5, line_color="grey")
fig4.update_layout(
    title=dict(text="<b>Force vs Indentation</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title="δ (nm)", yaxis_title="F (nN)",
    legend=_legend_right(), **dark_layout)
with col4:
    st.plotly_chart(fig4, use_container_width=True)

# =====================================================================
#  ROW 3 — Plot 5 & 6: Cantilever Linearity Check
# =====================================================================
st.markdown("---")
st.markdown(
    f"#### <span style='color:{NEON_GREEN}'>Cantilever Linearity Check</span>"
    " — local slope &amp; deviation from CP reference",
    unsafe_allow_html=True)

col5, col6 = st.columns(2)

# ── PLOT 5: Local slope vs ΔZ ────────────────────────────────────────
fig5 = go.Figure()
if lin_app is not None:
    la = lin_app
    end_a = find_linear_end(la["dev_pct"], la["ref_end_idx"], dev_threshold)
    fig5.add_trace(go.Scattergl(x=la["dz"], y=la["slope"], mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=4), name="Approach"))
    fig5.add_hline(y=la["ref_slope"], line_width=1.5, line_dash="dot",
                   line_color=FIT_BLUE,
                   annotation_text=f"ref = {la['ref_slope']:.1f}",
                   annotation_font_color=FIT_BLUE, annotation_font_size=10)
    dz_end_a = float(la["dz"][end_a])
    fig5.add_vline(x=dz_end_a, line_width=1.5, line_dash="dash",
                   line_color=NEON_BLUE, opacity=0.7)
    fig5.add_vrect(x0=0, x1=dz_end_a, fillcolor=NEON_BLUE, opacity=0.07,
                   line_width=0, layer="below")
if lin_ret is not None:
    lr = lin_ret
    end_r = find_linear_end(lr["dev_pct"], lr["ref_end_idx"], dev_threshold)
    fig5.add_trace(go.Scattergl(x=lr["dz"], y=lr["slope"], mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=4), name="Retract"))
    fig5.add_hline(y=lr["ref_slope"], line_width=1.5, line_dash="dot",
                   line_color=FIT_PINK,
                   annotation_text=f"ref = {lr['ref_slope']:.1f}",
                   annotation_font_color=FIT_PINK, annotation_font_size=10,
                   annotation_position="bottom right")
    dz_end_r = float(lr["dz"][end_r])
    fig5.add_vline(x=dz_end_r, line_width=1.5, line_dash="dash",
                   line_color=NEON_PINK, opacity=0.7)
    fig5.add_vrect(x0=0, x1=dz_end_r, fillcolor=NEON_PINK, opacity=0.05,
                   line_width=0, layer="below")
fig5.update_layout(
    title=dict(text="<b>Local Slope vs ΔZ from CP</b>",
               x=0.01, y=0.98, font=dict(size=13)),
    xaxis_title="ΔZ from CP (nm)", yaxis_title="Local slope (V/V)",
    legend=_legend_right(), **{**dark_layout, "height": 400})
with col5:
    st.plotly_chart(fig5, use_container_width=True)

# ── PLOT 6: Deviation (%) vs ΔZ ─────────────────────────────────────
fig6 = go.Figure()
fig6.add_hrect(y0=-dev_threshold, y1=dev_threshold, fillcolor=NEON_GREEN,
               opacity=0.08, line_width=0, layer="below")
fig6.add_hline(y=dev_threshold, line_width=1, line_dash="dot",
               line_color=NEON_GREEN, opacity=0.5)
fig6.add_hline(y=-dev_threshold, line_width=1, line_dash="dot",
               line_color=NEON_GREEN, opacity=0.5)
fig6.add_hline(y=0, line_width=0.5, line_color="grey")
if lin_app is not None:
    la = lin_app
    end_a = find_linear_end(la["dev_pct"], la["ref_end_idx"], dev_threshold)
    fig6.add_trace(go.Scattergl(x=la["dz"], y=la["dev_pct"], mode="markers",
        marker=dict(color=_rgba(NEON_BLUE, ALPHA), size=4), name="Approach"))
    dz_end_a = float(la["dz"][end_a])
    fig6.add_vline(x=dz_end_a, line_width=1.5, line_dash="dash",
                   line_color=NEON_BLUE, opacity=0.7,
                   annotation_text=f"{dz_end_a:.0f} nm",
                   annotation_font_color=NEON_BLUE, annotation_font_size=10,
                   annotation_position="top left")
if lin_ret is not None:
    lr = lin_ret
    end_r = find_linear_end(lr["dev_pct"], lr["ref_end_idx"], dev_threshold)
    fig6.add_trace(go.Scattergl(x=lr["dz"], y=lr["dev_pct"], mode="markers",
        marker=dict(color=_rgba(NEON_PINK, ALPHA), size=4), name="Retract"))
    dz_end_r = float(lr["dz"][end_r])
    fig6.add_vline(x=dz_end_r, line_width=1.5, line_dash="dash",
                   line_color=NEON_PINK, opacity=0.7,
                   annotation_text=f"{dz_end_r:.0f} nm",
                   annotation_font_color=NEON_PINK, annotation_font_size=10,
                   annotation_position="top right")
fig6.update_layout(
    title=dict(text="<b>Slope Deviation from CP Reference</b>",
               x=0.01, y=0.98, font=dict(size=13)),
    xaxis_title="ΔZ from CP (nm)", yaxis_title="Deviation (%)",
    legend=_legend_right(), **{**dark_layout, "height": 400})
with col6:
    st.plotly_chart(fig6, use_container_width=True)

# ── Linearity summary ────────────────────────────────────────────────
lc1, lc2, lc3, lc4 = st.columns(4)
if lin_app is not None:
    end_a = find_linear_end(lin_app["dev_pct"], lin_app["ref_end_idx"],
                            dev_threshold)
    lc1.metric("App ref slope (V/V)", f"{lin_app['ref_slope']:.2f}")
    lc2.metric("App linear range (nm)", f"{lin_app['dz'][end_a]:.0f}")
if lin_ret is not None:
    end_r = find_linear_end(lin_ret["dev_pct"], lin_ret["ref_end_idx"],
                            dev_threshold)
    lc3.metric("Ret ref slope (V/V)", f"{lin_ret['ref_slope']:.2f}")
    lc4.metric("Ret linear range (nm)", f"{lin_ret['dz'][end_r]:.0f}")

# =====================================================================
#  METRICS
# =====================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("App slope (V/V)", f"{fit_app.slope:.2f}")
c2.metric("Ret slope (V/V)", f"{fit_ret.slope:.2f}")
c3.metric("App INVOLS (nm/V)", f"{fit_app.invols_nm_per_V:.1f}")
c4.metric("Ret INVOLS (nm/V)", f"{fit_ret.invols_nm_per_V:.1f}")
c5.metric("Mean INVOLS (nm/V)", f"{invols_mean:.1f}")

c6, c7, c8, c9, c10 = st.columns(5)
c6.metric("k (N/m)", f"{k_spring:.3f}")
c7.metric("INVOLS used (nm/V)", f"{invols:.1f}")
c8.metric("CP distance (nm)", f"{cp_distance_nm:.1f}")
c9.metric("Phase angle (°)", f"{phase_deg:.2f}")
c10.metric("Curves", len(fc_files))

# =====================================================================
#  EXPORT
# =====================================================================
if st.sidebar.button("Export publication figure", key="export_btn"):
    # Gather linearity data for export
    _la_dz = lin_app["dz"] if lin_app is not None else np.array([])
    _la_sl = lin_app["slope"] if lin_app is not None else np.array([])
    _la_dv = lin_app["dev_pct"] if lin_app is not None else np.array([])
    _la_ref = lin_app["ref_slope"] if lin_app is not None else 0.0
    _la_end = float(lin_app["dz"][find_linear_end(lin_app["dev_pct"], lin_app["ref_end_idx"], dev_threshold)]) if lin_app is not None else 0.0
    _lr_dz = lin_ret["dz"] if lin_ret is not None else np.array([])
    _lr_sl = lin_ret["slope"] if lin_ret is not None else np.array([])
    _lr_dv = lin_ret["dev_pct"] if lin_ret is not None else np.array([])
    _lr_ref = lin_ret["ref_slope"] if lin_ret is not None else 0.0
    _lr_end = float(lin_ret["dz"][find_linear_end(lin_ret["dev_pct"], lin_ret["ref_end_idx"], dev_threshold)]) if lin_ret is not None else 0.0

    edata = ExportData(
        app_z=_x(fc.app_z_V), app_defl=fc.app_defl_V,
        ret_z=_x(fc.ret_z_V), ret_defl=fc.ret_defl_V,
        app_z_cr=_x(fc.app_z_V[cp_app_idx:]),
        app_defl_cr=fc.app_defl_V[cp_app_idx:],
        ret_z_cr=_x(fc.ret_z_V[:cp_ret_idx + 1]),
        ret_defl_cr=fc.ret_defl_V[:cp_ret_idx + 1],
        fit_app_x=_x(fit_app.x_fit), fit_app_y=fit_app.y_fit,
        fit_ret_x=_x(fit_ret.x_fit), fit_ret_y=fit_ret.y_fit,
        app_dz_nm=a_dz_nm, app_defl_nm=a_defl_nm,
        ret_dz_nm=r_dz_nm, ret_defl_nm=r_defl_nm,
        app_delta=a_delta, app_F=a_F,
        ret_delta=r_delta, ret_F=r_F,
        fit_app_dz=fa_dz, fit_app_defl=fa_defl,
        fit_ret_dz=fr_dz, fit_ret_defl=fr_defl,
        fit_app_delta=fa_delta, fit_app_F=fa_F,
        fit_ret_delta=fr_delta, fit_ret_F=fr_F,
        x_label=x_label,
        slope_app=fit_app.slope, slope_ret=fit_ret.slope,
        invols=invols, k=k_spring, phase_deg=phase_deg,
        cp_app_idx=cp_app_idx, cp_ret_idx=cp_ret_idx,
        cp_app_x=float(_x(fc.app_z_V)[cp_app_idx]),
        cp_ret_x=float(_x(fc.ret_z_V)[cp_ret_idx]),
        show=show_curves, filename=fc_files[fc_idx].stem,
        lin_app_dz=_la_dz, lin_app_slope=_la_sl, lin_app_dev=_la_dv,
        lin_app_ref=_la_ref, lin_app_end_nm=_la_end,
        lin_ret_dz=_lr_dz, lin_ret_slope=_lr_sl, lin_ret_dev=_lr_dv,
        lin_ret_ref=_lr_ref, lin_ret_end_nm=_lr_end,
        dev_threshold=dev_threshold,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_figure(edata, Path(tmp), fmt=export_fmt, dpi=300)
        with open(out_path, "rb") as f:
            data_bytes = f.read()
        st.sidebar.download_button(
            f"Download .{export_fmt}", data=data_bytes,
            file_name=out_path.name,
            mime={"pdf": "application/pdf", "png": "image/png",
                  "svg": "image/svg+xml"}[export_fmt])
    st.sidebar.success("Figure ready for download.")
