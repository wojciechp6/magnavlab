"""Integration tests on real SGL data (skipped when data files are absent).

Pipelines are built from primitives, mirroring the notebooks.
"""
import os

import numpy as np
import pytest

from magnavlab import data
from magnavlab.calibration import BuiltinTL, MapBasedModifiedTL
from magnavlab.filters import Canciani38EKF, EKFNav
from magnavlab.ins import build_kinematics, simulate_ins_pinson, simulate_ins_velocity
from magnavlab.interfaces import NavProblem, NavResult
from magnavlab.io import load_flight, load_map, segment_indices
from magnavlab.metrics import drms

DATA = ["data/Flt1003_train.h5", "data/Flt1002_train.h5", "data/maps/Eastern_395.h5"]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in DATA),
    reason="SGL/map data missing from data/ - fetch with tools/get_data.py (see README).")


def _segment():
    nav = load_flight("data/Flt1003_train.h5")
    mag_map = load_map("data/maps/Eastern_395.h5")
    sl = segment_indices(nav, 50713.0, 54497.0)[::5]
    dt = nav.dt * 5
    lat = np.radians(nav.get("lat")[sl])
    lon = np.radians(nav.get("lon")[sl])
    alt = nav.get("alt")[sl]
    return nav, mag_map, sl, dt, lat, lon, alt


def test_canciani_tightly_beats_loosely():
    nav, mag_map, sl, dt, lat, lon, alt = _segment()
    n = sl.size
    kin = build_kinematics(lat, lon, alt, dt,
                           np.radians(nav.get("roll")[sl]), np.radians(nav.get("pitch")[sl]))
    flux = nav.flux(sl)
    z = nav.get("mag_4_uc")[sl].astype(float)
    core = nav.get("mag_1_c")[sl] - nav.get("igrf")[sl] - nav.get("diurnal")[sl]
    cos_x, cos_y, cos_z = flux.x / z, flux.y / z, flux.z / z
    cos_dot = (np.gradient(cos_x, dt), np.gradient(cos_y, dt), np.gradient(cos_z, dt))
    nominal, _ = simulate_ins_pinson(lat, lon, alt, kin, dt)
    half = n // 2
    earth = mag_map.value(lat[:half], lon[:half]) + core[:half]
    tl = MapBasedModifiedTL().fit(nav.flux(sl[:half]), z[:half], dt, target=earth)

    problem = NavProblem(dt=dt, map=mag_map, lat=nominal["lat"], lon=nominal["lon"],
                         alt=nominal["alt"], vN=nominal["vN"], vE=nominal["vE"],
                         vD=nominal["vD"], meas=z, core=core, flux=flux, cos_dot=cos_dot,
                         fn=kin["fn"], fe=kin["fe"], fd=kin["fd"], Cnb=kin["Cnb"],
                         tl0=tl.coef, P_tl0=tl.P_cov)
    loose = Canciani38EKF(mode="loosely").run(problem)
    tight = Canciani38EKF(mode="tightly").run(problem)

    d_ins = drms(NavResult(nominal["lat"], nominal["lon"]), lat, lon)
    assert d_ins > 300.0                                  # INS drifts (hundreds of metres)
    assert drms(loose, lat, lon) < d_ins
    assert drms(tight, lat, lon) < drms(loose, lat, lon)  # online calibration helps
    assert drms(tight, lat, lon) < 120.0                  # ~57 m in practice


def test_simple_nav_ekf_beats_ins():
    nav, mag_map, sl, dt, lat, lon, alt = _segment()
    cal = load_flight("data/Flt1002_train.h5")
    ci = np.concatenate([segment_indices(cal, a, b) for a, b in data.CAL_SEGMENTS["Flt1002"]])
    comp = BuiltinTL().fit(cal.flux(ci), cal.get("mag_4_uc")[ci], cal.dt) \
        .compensate(nav.flux(sl), nav.get("mag_4_uc")[sl], dt)
    meas = comp - nav.get("diurnal")[sl]
    map_along = mag_map.value(lat, lon)
    ok = np.isfinite(map_along)
    meas = meas - np.median(meas[ok] - map_along[ok])
    resid = (meas - map_along)[ok]
    sigma = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 5.0)

    ins_lat, ins_lon, vN, vE = simulate_ins_velocity(lat, lon, alt, dt, seed=0)
    problem = NavProblem(dt=dt, map=mag_map, lat=ins_lat, lon=ins_lon, alt=alt,
                         vN=vN, vE=vE, vD=np.zeros(sl.size), meas=meas)
    ekf = EKFNav(sigma_meas=sigma).run(problem)
    assert drms(ekf, lat, lon) < drms(NavResult(ins_lat, ins_lon), lat, lon)
    assert drms(ekf, lat, lon) < 200.0
