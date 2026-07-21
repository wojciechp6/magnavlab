"""Loading flight data from HDF5 files (SGL / MagNav.jl XYZ20/XYZ21 format)."""
from __future__ import annotations

from dataclasses import dataclass, field

import h5py
import numpy as np

from ..interfaces import MagV

# Canonical field names -> list of aliases in the file. The loader takes the first existing one.
FIELD_ALIASES: dict[str, list[str]] = {
    "tt":       ["tt", "time"],
    "line":     ["line"],
    "lat":      ["lat", "latitude"],           # "true" trajectory (GPS) [deg]
    "lon":      ["lon", "longitude"],
    "alt":      ["alt", "msl", "hae", "gps_alt"],
    "roll":     ["ins_roll", "roll"],
    "pitch":    ["ins_pitch", "pitch"],
    "yaw":      ["ins_yaw", "yaw", "ins_azim", "heading"],
    "mag_1_c":  ["mag_1_c"],                    # compensated stinger (reference)
    "mag_1_uc": ["mag_1_uc"],
    "mag_2_uc": ["mag_2_uc"],
    "mag_3_uc": ["mag_3_uc"],
    "mag_4_uc": ["mag_4_uc"],
    "mag_5_uc": ["mag_5_uc"],
    "flux_b_x": ["flux_b_x"], "flux_b_y": ["flux_b_y"], "flux_b_z": ["flux_b_z"],
    "flux_a_x": ["flux_a_x"], "flux_a_y": ["flux_a_y"], "flux_a_z": ["flux_a_z"],
    "flux_c_x": ["flux_c_x"], "flux_c_y": ["flux_c_y"], "flux_c_z": ["flux_c_z"],
    "flux_d_x": ["flux_d_x"], "flux_d_y": ["flux_d_y"], "flux_d_z": ["flux_d_z"],
    "igrf":     ["igrf", "mag_1_igrf"],         # NOTE: in SGL data this is the ANOMALY field
    "diurnal":  ["diurnal"],
}

# Fluxgate sensor preference order (B is cleanest on the track).
_FLUX_SENSORS = ("flux_b", "flux_a", "flux_c", "flux_d")


@dataclass
class FlightData:
    """Single flight data: dict of equal-length arrays + sampling step."""
    raw: dict = field(default_factory=dict)
    dt: float = 0.1
    available: list[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return self.raw.get(name) is not None

    def get(self, name: str) -> np.ndarray:
        if not self.has(name):
            raise KeyError(f"Missing field '{name}'. Available raw: {self.available}")
        return self.raw[name]

    @property
    def n(self) -> int:
        return len(self.raw["tt"])

    def flux(self, indices: np.ndarray | slice | None = None) -> MagV:
        """Returns the vector magnetometer (first available sensor) as :class:`MagV`."""
        for s in _FLUX_SENSORS:
            if self.has(f"{s}_x"):
                sel = (lambda a: a) if indices is None else (lambda a: a[indices])
                return MagV(sel(self.get(f"{s}_x")),
                            sel(self.get(f"{s}_y")),
                            sel(self.get(f"{s}_z")))
        raise KeyError("No vector magnetometer available (flux_*).")


def load_flight(path: str) -> FlightData:
    """Loads a flight HDF5 file, mapping field names via :data:`FIELD_ALIASES`."""
    fl = FlightData()
    with h5py.File(path, "r") as f:
        available = set(f.keys())
        fl.available = sorted(available)
        for canonical, aliases in FIELD_ALIASES.items():
            fl.raw[canonical] = next(
                (np.asarray(f[a][()], dtype=float).ravel() for a in aliases if a in available),
                None,
            )
    if not fl.has("tt"):
        raise RuntimeError(f"Missing time axis 'tt' in {path}.")
    tt = fl.raw["tt"]
    if len(tt) > 1:
        fl.dt = float(np.median(np.diff(tt)))
    return fl


def segment_indices(fl: FlightData, t_start: float, t_end: float) -> np.ndarray:
    """Indices of the longest CONTIGUOUS block of samples in the time window [t_start, t_end].

    The SGL time axis is sometimes discontinuous (resets between flight phases) - the same time window
    may fall into several disjoint fragments; we choose the longest.
    """
    tt = fl.get("tt")
    idx = np.where((tt >= t_start) & (tt <= t_end))[0]
    if idx.size == 0:
        return idx
    splits = np.where(np.diff(idx) > 1)[0] + 1
    return max(np.split(idx, splits), key=len)


def inspect_h5(path: str) -> None:
    """Diagnostics: prints top-level datasets with shape and range."""
    with h5py.File(path, "r") as f:
        print(f"# File: {path}  (datasets: {len(f.keys())})")
        for k in sorted(f.keys()):
            o = f[k]
            if not hasattr(o, "shape"):
                print(f"  {k:16s} <group>")
                continue
            try:
                a = np.asarray(o[()], dtype=float).ravel()
                fin = a[np.isfinite(a)]
                lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (np.nan, np.nan)
                print(f"  {k:16s} shape={str(o.shape):14s} min={lo:.4g} max={hi:.4g}")
            except Exception as e:  # noqa: BLE001
                print(f"  {k:16s} shape={str(o.shape):14s} (non-numeric: {e})")
