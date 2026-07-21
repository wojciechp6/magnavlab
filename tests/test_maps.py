import h5py
import numpy as np
import pytest

from magnavlab.io.maps import load_map


def test_linear_map_value_and_gradient(linear_map):
    m = linear_map
    lat = m.lat[30:60]
    lon = m.lon[40:70]
    # value at nodes = A*lat + B*lon
    expected = m._A * lat + m._B * lon
    got = m.value(lat, lon)
    assert np.allclose(got, expected, rtol=1e-9)
    # constant gradient (A, B)
    d_lat, d_lon = m.gradient(lat, lon)
    assert np.allclose(d_lat, m._A, rtol=1e-4)
    assert np.allclose(d_lon, m._B, rtol=1e-4)


def test_out_of_bounds_is_nan(linear_map):
    v = linear_map.value(np.radians(90.0), np.radians(0.0))
    assert np.isnan(v[0])


def test_load_map_orientation_and_units(tmp_path):
    # save the map in degrees, grid (nlon, nlat) - as in SGL files
    lon_deg = np.linspace(-76.5, -75.5, 40)
    lat_deg = np.linspace(44.5, 45.5, 30)
    grid_lonlat = np.outer(lon_deg, lat_deg)          # shape (nlon, nlat)
    p = tmp_path / "m.h5"
    with h5py.File(p, "w") as f:
        f["map"] = grid_lonlat
        f["xx"] = lon_deg
        f["yy"] = lat_deg
        f["alt"] = 395.0
    m = load_map(str(p))
    assert m.grid.shape == (30, 40)                   # transposed to (nlat, nlon)
    assert m.lat.max() < 2 * np.pi                    # degrees -> radians
    assert m.alt == pytest.approx(395.0)
    # value at the corner matches the outer(lon, lat) definition
    got = m.value(np.radians(lat_deg[5]), np.radians(lon_deg[7]))[0]
    assert got == pytest.approx(lon_deg[7] * lat_deg[5], rel=1e-6)
