import numpy as np
import pytest

from magnavlab import geo


def test_meridian_radii_equator():
    Rn, Re = geo.meridian_radii(0.0)
    assert Re == pytest.approx(geo.WGS84_A)             # transverse = a at the equator
    assert Rn == pytest.approx(geo.WGS84_A * (1 - geo.WGS84_E2))
    assert Re > Rn                                       # flattening


def test_velocity_from_track_constant():
    dt, n = 0.5, 100
    lat0 = np.radians(45.0)
    Rn, Re = geo.meridian_radii(lat0)
    vN_true, vE_true = 60.0, -30.0
    alt = np.full(n, 400.0)
    lat = lat0 + vN_true * dt / (Rn + 400.0) * np.arange(n)
    lon = np.radians(-76.0) + vE_true * dt / ((Re + 400.0) * np.cos(lat0)) * np.arange(n)
    vN, vE = geo.velocity_from_track(lat, lon, alt, dt)
    assert vN[n // 2] == pytest.approx(vN_true, rel=1e-3)
    assert vE[n // 2] == pytest.approx(vE_true, rel=1e-2)


def test_horizontal_error():
    lat = np.radians(np.array([45.0]))
    lon = np.radians(np.array([-76.0]))
    assert geo.horizontal_error(lat, lon, lat, lon, lat)[0] == pytest.approx(0.0)
    Rn, _ = geo.meridian_radii(lat)
    dlat = 100.0 / Rn                                    # 100 m north
    err = geo.horizontal_error(lat + dlat, lon, lat, lon, lat)[0]
    assert err == pytest.approx(100.0, rel=1e-6)


def test_skew_equals_cross():
    v, w = np.array([1.0, 2.0, -3.0]), np.array([0.5, -1.0, 2.0])
    assert np.allclose(geo.skew(v) @ w, np.cross(v, w))


def test_dcm_orthonormal():
    C = geo.dcm_from_euler(0.2, -0.1, 1.3)
    assert np.allclose(C @ C.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(C) == pytest.approx(1.0)
