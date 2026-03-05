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

# ─── Cantilever presets ──────────────────────────────────────────────────────
CANTILEVERS = {
    "AC40":  {"k": 0.09,  "f0_kHz": 32,  "half_angle_deg": 17.5},
    "AC160": {"k": 26.0,  "f0_kHz": 300, "half_angle_deg": 17.5},
    "AC240": {"k": 2.0,   "f0_kHz": 70,  "half_angle_deg": 17.5},
}

NEON_BLUE = "#00e5ff"
NEON_PINK = "#ff10f0"

#default_path = "D:\シュテファン"
default_path = str(Path.cwd())
#print(default_path)

st.set_page_config(page_title="AFM Force-Curve Viewer", layout="wide")
st.title("AFM Force-Curve Viewer")

# =====================================================================
#  SIDEBAR
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
        chosen = st.selectbox(label, options, index=len(options)-1,
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
fc_idx = st.sidebar.slider("Force Curve Index", 0, len(fc_files)-1,
                           value=len(fc_files)-1, key="fc_slider")
st.sidebar.write(f"**{fc_files[fc_idx].name}** ({fc_idx+1}/{len(fc_files)})")

# ── Cantilever ───────────────────────────────────────────────────────────
st.sidebar.header("Cantilever")
cant_name = st.sidebar.selectbox("Model", list(CANTILEVERS.keys()), index=1,
                                 key="cant_model")
cant = CANTILEVERS[cant_name]
with st.sidebar.expander("Edit cantilever parameters"):
    k_spring   = st.number_input("k (N/m)", value=cant["k"],
                                 min_value=0.001, step=0.01, format="%.3f")
    half_angle = st.number_input("Half-angle (°)", value=cant["half_angle_deg"],
                                 min_value=0.1, step=0.5, format="%.1f")
    f0_kHz     = st.number_input("f₀ (kHz)", value=float(cant["f0_kHz"]),
                                 min_value=0.1, step=1.0, format="%.1f")

st.sidebar.header("Parameters")
z_scale = st.sidebar.number_input("Z-stage scale (µm/V)", value=config.z_scale,
                                  step=1.0, format="%.1f")
x_in_um = st.sidebar.toggle("X-axis in µm", value=False, key="x_unit")

st.sidebar.header("Display")
show_curves = st.sidebar.radio("Show curves", ["Both", "Approach", "Retract"],
                               index=0, key="show_curves")

# ── Read data ────────────────────────────────────────────────────────────
try:
    raw = read_lvm_raw(fc_files[fc_idx], num_app=config.num_app,
                       num_ret=config.num_ret)
except Exception as e:
    st.error(f"Read error: {e}"); st.stop()

auto_turn  = detect_turnaround(raw)
total_pts  = len(raw.z_V)
st.sidebar.header("Turnaround")
turn_offset = st.sidebar.slider("Offset (pts)", -500, 500, 0, key="turn_off")
turnaround  = int(np.clip(auto_turn + turn_offset, 1, total_pts - 2))
st.sidebar.caption(f"Auto {auto_turn} → {turnaround}")
fc = split_at_turnaround(raw, turnaround)

cp_app = estimate_contact_approach(fc.app_z_V, fc.app_defl_V)
cp_ret = estimate_contact_retract(fc.ret_z_V, fc.ret_defl_V)

st.sidebar.header("Contact point offsets")
cp_off_a = st.sidebar.slider("Approach (pts)", -500, 500, 0, key="cp_off_a")
cp_off_r = st.sidebar.slider("Retract (pts)",  -500, 500, 0, key="cp_off_r")
cp_app_idx = int(np.clip(cp_app.index + cp_off_a, 0, len(fc.app_z_V)-1))
cp_ret_idx = int(np.clip(cp_ret.index + cp_off_r, 0, len(fc.ret_z_V)-1))

# ── Linear fits ──────────────────────────────────────────────────────────
fit_app = fit_contact_region(fc.app_z_V, fc.app_defl_V,
                             cp_app_idx, len(fc.app_z_V), z_scale)
fit_ret = fit_contact_region(fc.ret_z_V, fc.ret_defl_V,
                             0, cp_ret_idx, z_scale)
phase_deg = compute_phase_angle(fit_app.slope, fit_ret.slope)
cp_distance_nm = abs(fc.app_z_V[cp_app_idx] - fc.ret_z_V[cp_ret_idx]) * z_scale * 1000
invols_mean = (fit_app.invols_nm_per_V + fit_ret.invols_nm_per_V) / 2.0

# ── INVOLS ───────────────────────────────────────────────────────────────
st.sidebar.header("INVOLS")
st.sidebar.caption(
    f"Approach: **{fit_app.invols_nm_per_V:.1f}** nm/V  \n"
    f"Retract: **{fit_ret.invols_nm_per_V:.1f}** nm/V  \n"
    f"Mean: **{invols_mean:.1f}** nm/V"
)

use_mean = st.sidebar.checkbox("Use mean INVOLS", value=True, key="use_mean")

if use_mean:
    invols = invols_mean
else:
    with st.sidebar.expander("INVOLS selection", expanded=True):
        invols_src = st.selectbox(
            "Source", ["Retract", "Approach", "Mean", "Custom"],
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

# ── Export ────────────────────────────────────────────────────────────────
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

x_label     = "Z (µm)" if x_in_um else "Z (V)"
x_label_rel = "ΔZ (µm)" if x_in_um else "Z (V)"

def dec2(a, b, n=4000):
    if len(a) <= n: return a, b
    s = max(1, len(a)//n)
    return a[::s], b[::s]

MAX = 4000
PH  = 380

dark_layout = dict(
    height=PH,
    margin=dict(l=60, r=60, t=10, b=10),
    #template="plotly_dark",
    #paper_bgcolor="rgba(0,0,0,0)",
    #plot_bgcolor="#0e1117",
    font=dict(color="#fafafa", size=12),
)

def _legend_right():
    return dict(orientation="v", yanchor="top", y=1.0,
                xanchor="left", x=1.02, font=dict(size=11))

# =====================================================================
#  ROW 1 — Plot 1 (full raw) | Plot 2 (contact region)
# =====================================================================
col1, col2 = st.columns(2)

with col1:
    zoom_cp = st.toggle("Zoom to contact region", value=False, key="zoom_cp")

fig1 = go.Figure()
if show_curves in ("Both", "Approach"):
    ax, ay = dec2(_x(fc.app_z_V), fc.app_defl_V, MAX)
    fig1.add_trace(go.Scattergl(x=ax, y=ay, mode="lines",
        line=dict(color=NEON_BLUE, width=1.2), name="Approach"))
    fig1.add_trace(go.Scatter(
        x=[_x(fc.app_z_V)[cp_app_idx]], y=[fc.app_defl_V[cp_app_idx]],
        mode="markers", marker=dict(size=9, color=NEON_BLUE,
        symbol="circle-open", line=dict(width=2.5)), name="CP app"))
if show_curves in ("Both", "Retract"):
    rx, ry = dec2(_x(fc.ret_z_V), fc.ret_defl_V, MAX)
    fig1.add_trace(go.Scattergl(x=rx, y=ry, mode="lines",
        line=dict(color=NEON_PINK, width=1.2), name="Retract"))
    fig1.add_trace(go.Scatter(
        x=[_x(fc.ret_z_V)[cp_ret_idx]], y=[fc.ret_defl_V[cp_ret_idx]],
        mode="markers", marker=dict(size=9, color=NEON_PINK,
        symbol="circle-open", line=dict(width=2.5)), name="CP ret"))
fig1.add_trace(go.Scatter(
    x=[_x(fc.app_z_V)[-1]], y=[fc.app_defl_V[-1]],
    mode="markers", marker=dict(size=8, color="white", symbol="diamond"),
    name="Turnaround"))

if zoom_cp:
    x_lo = float(_x(fc.app_z_V)[cp_app_idx-200])
    all_x_ends = [float(_x(fc.app_z_V)[-1])]
    if show_curves in ("Both", "Retract"):
        ret_x = _x(fc.ret_z_V)
        all_x_ends.extend([float(ret_x.min()), float(ret_x.max())])
    x_hi = max(all_x_ends)
    x_margin = 0.05 * abs(x_hi - x_lo)
    fig1.update_xaxes(range=[x_lo - x_margin, x_hi + x_margin])

fig1.update_layout(
    title=dict(text="<b>Deflection vs Z – Full</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title=x_label, yaxis_title="Deflection (V)",
    legend=_legend_right(), **dark_layout,
)
with col1:
    st.plotly_chart(fig1, use_container_width=True)

# ── PLOT 2: Contact region ───────────────────────────────────────────────
fig2 = go.Figure()
if show_curves in ("Both", "Approach"):
    a_zc = fc.app_z_V[cp_app_idx:]
    a_dc = fc.app_defl_V[cp_app_idx:]
    a_x  = _x(a_zc, zero_first=x_in_um)
    cx, cy = dec2(a_x, a_dc, MAX)
    fig2.add_trace(go.Scattergl(x=cx, y=cy, mode="markers",
        marker=dict(color=NEON_BLUE, size=3), name="Approach"))
    if len(fit_app.x_fit) > 1:
        fit_x = _x(fit_app.x_fit)
        if x_in_um and len(a_zc) > 0:
            fit_x = fit_x - _x(a_zc)[0]
        fig2.add_trace(go.Scatter(
            x=fit_x, y=fit_app.y_fit, mode="lines",
            line=dict(color=NEON_BLUE, width=2.5),
            name=f"Fit app (s={fit_app.slope:.1f})"))

if show_curves in ("Both", "Retract"):
    r_zc = fc.ret_z_V[:cp_ret_idx+1]
    r_dc = fc.ret_defl_V[:cp_ret_idx+1]
    r_x  = _x(r_zc, zero_first=x_in_um)
    cx, cy = dec2(r_x, r_dc, MAX)
    fig2.add_trace(go.Scattergl(x=cx, y=cy, mode="markers",
        marker=dict(color=NEON_PINK, size=3), name="Retract"))
    if len(fit_ret.x_fit) > 1:
        fit_x = _x(fit_ret.x_fit)
        if x_in_um and len(r_zc) > 0:
            fit_x = fit_x - _x(r_zc)[0]
        fig2.add_trace(go.Scatter(
            x=fit_x, y=fit_ret.y_fit, mode="lines",
            line=dict(color=NEON_PINK, width=2.5),
            name=f"Fit ret (s={fit_ret.slope:.1f})"))

fig2.update_layout(
    title=dict(text="<b>Deflection vs Z – Contact Region</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title=x_label_rel if x_in_um else x_label,
    yaxis_title="Deflection (V)",
    legend=_legend_right(), **dark_layout,
)
with col2:
    st.plotly_chart(fig2, use_container_width=True)

# =====================================================================
#  ROW 2 — Plot 3: Defl(nm) vs ΔZ(nm)  |  Plot 4: F(nN) vs δ(nm)
# =====================================================================
z_cp = fc.app_z_V[cp_app_idx]
d_cp = fc.app_defl_V[cp_app_idx]

# Approach: CP → turnaround
a_zs = fc.app_z_V[cp_app_idx:]
a_ds = fc.app_defl_V[cp_app_idx:]
a_dz_nm   = (a_zs - z_cp) * z_scale * 1000
a_defl_nm = (a_ds - d_cp) * invols
a_delta   = a_dz_nm - a_defl_nm
a_F       = k_spring * a_defl_nm

# Retract: turnaround → CP
r_zs = fc.ret_z_V[:cp_ret_idx+1]
r_ds = fc.ret_defl_V[:cp_ret_idx+1]
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

# ── PLOT 3: Deflection (nm) vs ΔZ (nm) ──────────────────────────────────
fig3 = go.Figure()
if show_curves in ("Both", "Approach"):
    dx, dy = dec2(a_dz_nm, a_defl_nm, MAX)
    fig3.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=NEON_BLUE, size=3), name="Approach"))
    if len(fa_dz) > 1:
        fig3.add_trace(go.Scatter(x=fa_dz, y=fa_defl, mode="lines",
            line=dict(color=NEON_BLUE, width=2.5), name="Fit app"))
if show_curves in ("Both", "Retract"):
    dx, dy = dec2(r_dz_nm, r_defl_nm, MAX)
    fig3.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=NEON_PINK, size=3), name="Retract"))
    if len(fr_dz) > 1:
        fig3.add_trace(go.Scatter(x=fr_dz, y=fr_defl, mode="lines",
            line=dict(color=NEON_PINK, width=2.5), name="Fit ret"))

all_dz = []
if show_curves in ("Both", "Approach") and len(a_dz_nm) > 0:
    all_dz.extend([float(a_dz_nm.min()), float(a_dz_nm.max())])
if show_curves in ("Both", "Retract") and len(r_dz_nm) > 0:
    all_dz.extend([float(r_dz_nm.min()), float(r_dz_nm.max())])
if all_dz:
    ref_lo, ref_hi = min(all_dz), max(all_dz)
    fig3.add_trace(go.Scatter(
        x=[ref_lo, ref_hi], y=[ref_lo, ref_hi], mode="lines",
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
        name="slope = 1"))

fig3.add_hline(y=0, line_width=0.5, line_color="grey")
fig3.add_vline(x=0, line_width=0.5, line_color="grey")
fig3.update_layout(
    title=dict(text="<b>Deflection vs ΔZ (INVOLS check)</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title="ΔZ (nm)", yaxis_title="Deflection (nm)",
    legend=_legend_right(), **dark_layout,
)
with col3:
    st.plotly_chart(fig3, use_container_width=True)

# ── PLOT 4: F (nN) vs δ (nm) ────────────────────────────────────────────
fig4 = go.Figure()
if show_curves in ("Both", "Approach"):
    dx, dy = dec2(a_delta, a_F, MAX)
    fig4.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=NEON_BLUE, size=3), name="Approach"))
    if len(fa_delta) > 1:
        fig4.add_trace(go.Scatter(x=fa_delta, y=fa_F, mode="lines",
            line=dict(color=NEON_BLUE, width=2.5), name="Fit app"))
if show_curves in ("Both", "Retract"):
    dx, dy = dec2(r_delta, r_F, MAX)
    fig4.add_trace(go.Scattergl(x=dx, y=dy, mode="markers",
        marker=dict(color=NEON_PINK, size=3), name="Retract"))
    if len(fr_delta) > 1:
        fig4.add_trace(go.Scatter(x=fr_delta, y=fr_F, mode="lines",
            line=dict(color=NEON_PINK, width=2.5), name="Fit ret"))
fig4.add_hline(y=0, line_width=0.5, line_color="grey")
fig4.add_vline(x=0, line_width=0.5, line_color="grey")
fig4.update_layout(
    title=dict(text="<b>Force vs Indentation</b>", x=0.01, y=0.98,
               font=dict(size=13)),
    xaxis_title="δ (nm)", yaxis_title="F (nN)",
    legend=_legend_right(), **dark_layout,
)
with col4:
    st.plotly_chart(fig4, use_container_width=True)

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
    edata = ExportData(
        app_z=_x(fc.app_z_V), app_defl=fc.app_defl_V,
        ret_z=_x(fc.ret_z_V), ret_defl=fc.ret_defl_V,
        app_z_cr=_x(fc.app_z_V[cp_app_idx:]),
        app_defl_cr=fc.app_defl_V[cp_app_idx:],
        ret_z_cr=_x(fc.ret_z_V[:cp_ret_idx+1]),
        ret_defl_cr=fc.ret_defl_V[:cp_ret_idx+1],
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
        show=show_curves, filename=fc_files[fc_idx].stem,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = export_figure(edata, Path(tmp), fmt=export_fmt, dpi=300)
        with open(out_path, "rb") as f:
            data_bytes = f.read()
        st.sidebar.download_button(
            f"Download .{export_fmt}", data=data_bytes,
            file_name=out_path.name,
            mime={"pdf": "application/pdf", "png": "image/png",
                  "svg": "image/svg+xml"}[export_fmt],
        )
    st.sidebar.success("Figure ready for download.")
