import numpy as np

from magnavlab.geo import horizontal_error, meridian_radii
from magnavlab.io.flight import FlightData
from magnavlab.ins import build_kinematics, real_ins, simulate_ins_pinson, simulate_ins_velocity
from magnavlab.interfaces import NavResult
from magnavlab.metrics import drms, summary


def test_metrics_constant_error(straight_traj):
    t = straight_traj
    Rn, _ = meridian_radii(t["lat"][0])
    res = NavResult(lat=t["lat"] + 70.0 / Rn, lon=t["lon"].copy())
    assert drms(res, t["lat"], t["lon"]) == np.float64(drms(res, t["lat"], t["lon"]))
    m = summary(res, t["lat"], t["lon"])
    assert abs(m["drms"] - 70.0) < 0.05
    assert abs(m["cep50"] - 70.0) < 0.05


def test_simulate_ins_velocity_drifts(straight_traj):
    t = straight_traj
    il, io, vN, vE = simulate_ins_velocity(t["lat"], t["lon"], t["alt"], t["dt"], seed=0)
    from magnavlab.geo import horizontal_error
    e = horizontal_error(il, io, t["lat"], t["lon"], t["lat"])
    assert e[0] < 1.0                                     # start = truth
    assert e[-1] > e[len(e) // 2] > 1.0                  # drift accumulates


def test_real_ins_units_frame_and_anchor(straight_traj):
    """real_ins: rad/deg units, wander->NED velocity frame, and cold-start anchoring."""
    t = straight_traj
    n = t["n"]
    lat_deg = np.degrees(t["lat"]); lon_deg = np.degrees(t["lon"])
    raw = dict(
        tt=t["tt"], lat=lat_deg, lon=lon_deg,
        ins_lat=t["lat"] + 1e-4,          # radians, offset ~600 m from truth
        ins_lon=t["lon"] - 2e-4,
        ins_alt=t["alt"].copy(),
        ins_vn=np.full(n, 60.0),          # north
        ins_vw=np.full(n, -5.0),          # west  -> east = +5
        ins_vu=np.full(n, 1.0),           # up    -> down = -1
        roll=np.full(n, 3.0), pitch=np.full(n, -2.0),   # degrees
    )
    fl = FlightData(raw=raw, dt=t["dt"], available=list(raw))

    ins = real_ins(fl, anchor=(t["lat"][0], t["lon"][0]))
    # velocity frame: wander N/U/W -> NED
    assert np.allclose(ins["vN"], 60.0)
    assert np.allclose(ins["vE"], 5.0)     # -ins_vw
    assert np.allclose(ins["vD"], -1.0)    # -ins_vu
    # ins_lat is already radians -> stays radians; attitude deg -> rad
    assert 0.7 < ins["lat"][0] < 0.8
    assert np.allclose(ins["roll"], np.radians(3.0))
    # anchoring: first sample coincides with truth, drift preserved afterwards
    assert abs(ins["lat"][0] - t["lat"][0]) < 1e-12
    assert abs(ins["lon"][0] - t["lon"][0]) < 1e-12
    e = horizontal_error(ins["lat"], ins["lon"], t["lat"], t["lon"], t["lat"])
    assert e[0] < 1e-6                      # zero error at start (anchored)

    # without anchoring the recorded offset remains
    ins_raw = real_ins(fl)
    e_raw = horizontal_error(ins_raw["lat"], ins_raw["lon"], t["lat"], t["lon"], t["lat"])
    assert e_raw[0] > 100.0


def test_simulate_ins_pinson_consistency(straight_traj):
    t = straight_traj
    kin = build_kinematics(t["lat"], t["lon"], t["alt"], t["dt"])
    nominal, e_true = simulate_ins_pinson(t["lat"], t["lon"], t["alt"], kin, t["dt"],
                                          seed=0)
    assert e_true.shape == (15, t["n"])
    # nominal = truth - error  =>  nominal + error == truth
    assert np.allclose(nominal["lat"] + e_true[0], t["lat"])
    assert np.allclose(nominal["lon"] + e_true[1], t["lon"])
