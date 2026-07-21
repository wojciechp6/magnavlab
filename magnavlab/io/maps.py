"""Loading and interpolation of magnetic anomaly maps (MapS format from MagNav.jl)."""
from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


@dataclass
class MagMap:
    """Gridded anomaly map; implements the :class:`~magnavlab.interfaces.MapLike` protocol.

    The ``lat``/``lon`` axes are in radians (increasing), ``grid`` in (n_lat, n_lon) layout.
    """
    lat: np.ndarray
    lon: np.ndarray
    grid: np.ndarray
    alt: float

    def __post_init__(self) -> None:
        self._interp = RegularGridInterpolator(
            (self.lat, self.lon), self.grid,
            method="linear", bounds_error=False, fill_value=np.nan)

    def value(self, lat_rad, lon_rad) -> np.ndarray:
        lat_rad = np.atleast_1d(lat_rad)
        lon_rad = np.atleast_1d(lon_rad)
        return self._interp(np.column_stack([lat_rad, lon_rad]))

    def gradient(self, lat_rad, lon_rad, h: float | None = None):
        """Gradient (∂/∂lat, ∂/∂lon) [nT/rad] using central differences."""
        if h is None:
            h = 0.5 * float(self.lat[1] - self.lat[0])
        d_lat = (self.value(lat_rad + h, lon_rad) - self.value(lat_rad - h, lon_rad)) / (2 * h)
        d_lon = (self.value(lat_rad, lon_rad + h) - self.value(lat_rad, lon_rad - h)) / (2 * h)
        return d_lat, d_lon

    def extent_deg(self) -> tuple[float, float, float, float]:
        return (float(np.degrees(self.lon.min())), float(np.degrees(self.lon.max())),
                float(np.degrees(self.lat.min())), float(np.degrees(self.lat.max())))


def _to_rad(a: np.ndarray) -> np.ndarray:
    """Auto-detect axis units: radians stay, degrees -> radians."""
    return a if np.nanmax(np.abs(a)) < 2 * np.pi + 1e-6 else np.radians(a)


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Fills NaN with the nearest value (nearest) - typically survey edges."""
    mask = np.isnan(a)
    if not mask.any():
        return a
    idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def load_map(path: str) -> MagMap:
    """Loads an HDF5 map. Auto-detects axis units and grid orientation."""
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

        def first(*names):
            return next((np.asarray(f[n][()], dtype=float) for n in names if n in keys), None)

        grid = first("map", "mag", "anomaly")
        xx = first("xx", "lon", "longitude", "x")   # longitude
        yy = first("yy", "lat", "latitude", "y")    # latitude
        alt = first("alt", "altitude")
        if grid is None or xx is None or yy is None:
            raise RuntimeError(f"Unknown map format {path}. Fields: {sorted(keys)}")
        alt_val = float(np.nanmean(alt)) if alt is not None else 0.0

    lon, lat = _to_rad(xx.ravel()), _to_rad(yy.ravel())

    if grid.shape == (lat.size, lon.size):
        pass
    elif grid.shape == (lon.size, lat.size):
        grid = grid.T
    else:
        raise RuntimeError(f"Inconsistent map shape {grid.shape} vs (lat={lat.size}, lon={lon.size})")

    if lat[0] > lat[-1]:
        lat, grid = lat[::-1], grid[::-1, :]
    if lon[0] > lon[-1]:
        lon, grid = lon[::-1], grid[:, ::-1]
    grid = _fill_nan(grid)
    return MagMap(lat=lat, lon=lon, grid=grid, alt=alt_val)
