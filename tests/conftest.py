"""Shared synthetic data factories for tests (fast, no files).

Unit tests do not require data from Zenodo - they use synthetic maps and trajectories,
which makes them deterministic and fast. Integration tests (on real data)
are in test_integration.py and skipped when files are missing in data/.
"""
from __future__ import annotations

import numpy as np
import pytest

from magnavlab.io.maps import MagMap
from magnavlab.interfaces import MagV


@pytest.fixture
def linear_map() -> MagMap:
    """Map with a linear field: grid = A*lat + B*lon (exact value and gradient)."""
    lat = np.radians(np.linspace(44.5, 45.0, 120))
    lon = np.radians(np.linspace(-76.2, -75.6, 150))
    A, B = 5000.0, -3000.0
    grid = A * lat[:, None] + B * lon[None, :]
    m = MagMap(lat=lat, lon=lon, grid=grid, alt=400.0)
    m._A, m._B = A, B          # remember coefficients for assertions
    return m


@pytest.fixture
def wavy_map() -> MagMap:
    """Map with 2D structure (sums of sinusoids) - observable position for filters."""
    lat = np.radians(np.linspace(44.4, 45.1, 400))
    lon = np.radians(np.linspace(-76.3, -75.5, 480))
    kx, ky = 900.0, 1100.0
    grid = (250 * np.sin(ky * lat[:, None]) * np.cos(kx * lon[None, :])
            + 120 * np.sin(2.3 * ky * lat[:, None] + 1.0))
    return MagMap(lat=lat, lon=lon, grid=grid, alt=400.0)


@pytest.fixture
def straight_traj():
    """Straight trajectory (constant NE velocity), 500 samples @ 1 Hz (~8 min)."""
    n, dt = 500, 1.0
    lat0, lon0 = np.radians(44.6), np.radians(-76.0)
    # ~60 m/s on NE -> lat/lon increment per step
    dlat = 60.0 * dt / 6.37e6
    dlon = 60.0 * dt / (6.37e6 * np.cos(lat0))
    lat = lat0 + dlat * np.arange(n)
    lon = lon0 + dlon * np.arange(n)
    alt = np.full(n, 400.0)
    tt = dt * np.arange(n)
    return dict(lat=lat, lon=lon, alt=alt, dt=dt, tt=tt, n=n)


@pytest.fixture
def synthetic_flux():
    """Vector magnetometer factory from given angles (for T-L tests)."""
    def make(n=500, seed=0):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 50, n)
        # field direction ~unit, slowly rotated (maneuvers)
        yaw = 0.3 * np.sin(0.5 * t) + 0.1 * rng.standard_normal(n).cumsum() / n
        pitch = 0.2 * np.sin(0.3 * t)
        Bt = 53000.0
        x = Bt * np.cos(pitch) * np.cos(yaw)
        y = Bt * np.cos(pitch) * np.sin(yaw)
        z = Bt * np.sin(pitch)
        return MagV(x=x, y=y, z=z)
    return make
