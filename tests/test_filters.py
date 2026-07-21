"""Filter tests on a synthetic map: correction of a drifting INS.

Filters correct INS drift (through velocity error / calibration states), so the test
scenario is a drifting INS, and the criterion - the filter is substantially more accurate than the INS alone.
"""
import numpy as np

from magnavlab.filters import Canciani38EKF, EKFNav, ParticleFilterNav
from magnavlab.ins import build_kinematics, simulate_ins_pinson, simulate_ins_velocity
from magnavlab.interfaces import MagV, NavProblem, NavResult
from magnavlab.metrics import drms, error_series


def _ins_drms(il, io, lat, lon):
    return drms(NavResult(lat=il, lon=io), lat, lon)


def test_ekf_beats_drifting_ins(wavy_map, straight_traj):
    t = straight_traj
    lat, lon, alt, dt = t["lat"], t["lon"], t["alt"], t["dt"]
    il, io, vN, vE = simulate_ins_velocity(lat, lon, alt, dt,
                                           vel_bias=(0.6, -0.5), rw_sigma=1e-3, seed=0)
    prob = NavProblem(dt=dt, map=wavy_map, lat=il, lon=io, alt=alt,
                      vN=vN, vE=vE, vD=np.zeros(t["n"]), meas=wavy_map.value(lat, lon))
    res = EKFNav(sigma_meas=3.0).run(prob)
    d_ins = _ins_drms(il, io, lat, lon)
    assert d_ins > 100.0                                  # INS drifts noticeably
    assert drms(res, lat, lon) < 0.3 * d_ins              # filter significantly better
    assert np.mean(error_series(res, lat, lon)[-50:]) < 40.0


def test_pf_beats_drifting_ins(wavy_map, straight_traj):
    t = straight_traj
    lat, lon, alt, dt = t["lat"], t["lon"], t["alt"], t["dt"]
    il, io, vN, vE = simulate_ins_velocity(lat, lon, alt, dt,
                                           vel_bias=(0.6, -0.5), rw_sigma=1e-3, seed=0)
    prob = NavProblem(dt=dt, map=wavy_map, lat=il, lon=io, alt=alt,
                      vN=vN, vE=vE, vD=np.zeros(t["n"]), meas=wavy_map.value(lat, lon))
    res = ParticleFilterNav(n_particles=1500, sigma_meas=3.0, seed=0).run(prob)
    d_ins = _ins_drms(il, io, lat, lon)
    assert d_ins > 100.0
    assert drms(res, lat, lon) < 0.5 * d_ins


def test_ekf38_beats_drifting_ins(wavy_map, straight_traj):
    t = straight_traj
    lat, lon, alt, dt, n = t["lat"], t["lon"], t["alt"], t["dt"], t["n"]
    kin = build_kinematics(lat, lon, alt, dt)
    # noticeable INS drift from injected IMU biases
    nominal, _ = simulate_ins_pinson(lat, lon, alt, kin, dt,
                                     accel_bias=(1e-3, -8e-4, 0.0),
                                     gyro_bias=(1e-6, -8e-7, 0.0), seed=0)
    flux = MagV(np.full(n, 1200.0), np.full(n, -800.0), np.full(n, 400.0))
    z = wavy_map.value(lat, lon)                          # tl0=0 -> C=0, core=0
    prob = NavProblem(dt=dt, map=wavy_map, lat=nominal["lat"], lon=nominal["lon"],
                      alt=nominal["alt"], vN=nominal["vN"], vE=nominal["vE"],
                      vD=nominal["vD"], meas=z, core=np.zeros(n), flux=flux,
                      cos_dot=(np.zeros(n), np.zeros(n), np.zeros(n)),
                      fn=kin["fn"], fe=kin["fe"], fd=kin["fd"], Cnb=kin["Cnb"],
                      tl0=np.zeros(19), P_tl0=np.eye(19) * 1e-3)
    res = Canciani38EKF(mode="tightly", R=4.0, Qf=100.0**2).run(prob)
    d_ins = _ins_drms(nominal["lat"], nominal["lon"], lat, lon)
    assert d_ins > 80.0
    assert drms(res, lat, lon) < 0.6 * d_ins
