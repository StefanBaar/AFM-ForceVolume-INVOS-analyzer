"""
io_utils.py – I/O helpers for AFM force-curve LVM files.

LVM layout (single column, tab-prefixed):
  1 .. num_app             : approach deflection (V)
  num_app+1 .. 2*num_app   : retract  deflection (V)
  2*num_app+1 .. 3*num_app : approach Z-stage    (V)
  3*num_app+1 .. 4*num_app : retract  Z-stage    (V)

The file-level split (num_app) may NOT coincide with the physical
turnaround.  We concatenate both halves and split at the actual Z peak.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import re


# ── Config ───────────────────────────────────────────────────────────────────

def read_config(config_path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    with open(config_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                parts = line.split(",", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) == 2:
                cfg[parts[0].strip()] = parts[1].strip()
    return cfg


@dataclass
class FCConfig:
    num_app: int = 12000
    num_ret: int = 12000
    vtrig: float = 0.4
    z_scale: float = 30.0
    app_speed_ratio: float = 1.5
    ret_speed_ratio: float = 1.5
    xlength: float = 3.0
    ylength: float = 3.0
    xstep: int = 5
    ystep: int = 5
    ret_length: float = 8.0
    loop_time: float = 20.0

    @classmethod
    def from_file(cls, path: Path) -> "FCConfig":
        raw = read_config(path)
        def _int(k, d):
            try: return int(raw.get(k, d))
            except: return d
        def _float(k, d):
            try: return float(raw.get(k, d))
            except: return d
        return cls(
            num_app=_int("num_app", 12000), num_ret=_int("num_ret", 12000),
            vtrig=_float("Vtrig", 0.4),
            app_speed_ratio=_float("app_speed_ratio", 1.5),
            ret_speed_ratio=_float("ret_speed_ratio", 1.5),
            xlength=_float("xlength", 3.0), ylength=_float("ylength", 3.0),
            xstep=_int("Xstep", 5), ystep=_int("Ystep", 5),
            ret_length=_float("ret_length", 8.0),
            loop_time=_float("loop_time", 20.0),
        )


# ── LVM reading ──────────────────────────────────────────────────────────────

@dataclass
class RawLVM:
    defl_V: np.ndarray
    z_V: np.ndarray
    num_app: int
    filepath: Path = None


@dataclass
class ForceCurve:
    app_defl_V: np.ndarray
    ret_defl_V: np.ndarray
    app_z_V: np.ndarray
    ret_z_V: np.ndarray
    turnaround: int
    filepath: Path = None


def read_lvm_raw(filepath: Path, num_app: int = 12000,
                 num_ret: int = 12000) -> RawLVM:
    data = np.loadtxt(filepath)
    total = 2 * num_app + 2 * num_ret
    if data.size < total:
        raise ValueError(f"{filepath.name}: {data.size} values, need {total}.")
    i = 0
    app_defl = data[i:i + num_app]; i += num_app
    ret_defl = data[i:i + num_ret]; i += num_ret
    app_z    = data[i:i + num_app]; i += num_app
    ret_z    = data[i:i + num_ret]
    return RawLVM(
        np.concatenate([app_defl, ret_defl]),
        np.concatenate([app_z, ret_z]),
        num_app, filepath,
    )


def detect_turnaround(raw: RawLVM) -> int:
    return int(np.argmax(raw.z_V))


def split_at_turnaround(raw: RawLVM, turnaround: int) -> ForceCurve:
    t = max(1, min(turnaround, len(raw.z_V) - 2))
    return ForceCurve(
        app_defl_V=raw.defl_V[: t + 1],
        ret_defl_V=raw.defl_V[t:],
        app_z_V=raw.z_V[: t + 1],
        ret_z_V=raw.z_V[t:],
        turnaround=t,
        filepath=raw.filepath,
    )


# ── Directory discovery ──────────────────────────────────────────────────────

def discover_datasets(root: Path) -> List[Path]:
    lvm_files = sorted(root.rglob("ForceCurve_*.lvm"))
    seen = set()
    dirs = []
    for f in lvm_files:
        d = f.parent
        if d not in seen:
            seen.add(d)
            dirs.append(d)
    return dirs


def list_force_curves(directory: Path) -> List[Path]:
    return sorted(directory.glob("ForceCurve_*.lvm"),
                  key=lambda p: _extract_index(p))


def _extract_index(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0


def build_directory_tree(dataset_dirs: List[Path], root: Path):
    entries = []
    for d in dataset_dirs:
        try:
            rel = d.relative_to(root)
        except ValueError:
            rel = d
        entries.append((list(rel.parts), d))
    if not entries:
        return [], 0
    max_depth = max(len(parts) for parts, _ in entries)
    return entries, max_depth
