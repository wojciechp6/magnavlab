import numpy as np

from magnavlab.geo import meridian_radii
from magnavlab.ins import build_kinematics, simulate_ins_pinson, simulate_ins_velocity
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


def test_simulate_ins_pinson_consistency(straight_traj):
    t = straight_traj
    kin = build_kinematics(t["lat"], t["lon"], t["alt"], t["dt"])
    nominal, e_true = simulate_ins_pinson(t["lat"], t["lon"], t["alt"], kin, t["dt"],
                                          seed=0)
    assert e_true.shape == (15, t["n"])
    # nominal = truth - error  =>  nominal + error == truth
    assert np.allclose(nominal["lat"] + e_true[0], t["lat"])
    assert np.allclose(nominal["lon"] + e_true[1], t["lon"])
