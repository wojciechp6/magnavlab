"""Trajectory kinematics and simulation of drifting inertial navigation (INS).

Two drift-simulation methods:
  - :func:`simulate_ins_velocity` - error at the velocity level (for simple filters),
  - :func:`simulate_ins_pinson`   - integration of the Pinson error model with IMU biases
    (consistent with the EKF38 filter: the same F matrix generates drift and is used in the filter).
"""
from __future__ import annotations

import numpy as np

from .geo import G0, dcm_from_euler, meridian_radii, velocity_from_track
from .models.pinson import pinson_F


def build_kinematics(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, dt: float,
                     roll: np.ndarray | None = None,
                     pitch: np.ndarray | None = None) -> dict:
    """Computes velocities, specific forces, heading and DCM from the true trajectory."""
    n = len(lat)
    vN, vE = velocity_from_track(lat, lon, alt, dt)
    vD = -np.gradient(alt, dt)
    fn, fe = np.gradient(vN, dt), np.gradient(vE, dt)
    fd = np.gradient(vD, dt) - G0                 # specific force downward ~ -g
    yaw = np.arctan2(vE, vN)
    roll = np.zeros(n) if roll is None else roll
    pitch = np.zeros(n) if pitch is None else pitch
    Cnb = np.array([dcm_from_euler(roll[k], pitch[k], yaw[k]) for k in range(n)])
    return dict(vN=vN, vE=vE, vD=vD, fn=fn, fe=fe, fd=fd, yaw=yaw, Cnb=Cnb)


def simulate_ins_velocity(lat: np.ndarray, lon: np.ndarray, alt: np.ndarray, dt: float,
                          vel_bias: tuple[float, float] = (0.25, -0.20),
                          rw_sigma: float = 1e-3, seed: int = 0):
    """INS drift at the velocity level (bias + random walk). Start = truth.

    Returns (ins_lat, ins_lon, vN_ins, vE_ins)."""
    rng = np.random.default_rng(seed)
    vN, vE = velocity_from_track(lat, lon, alt, dt)
    n = len(vN)
    alt = np.asarray(alt) if np.ndim(alt) else np.full(n, float(alt))
    eN = vel_bias[0] + np.cumsum(rng.normal(0, rw_sigma, n)) * np.sqrt(dt)
    eE = vel_bias[1] + np.cumsum(rng.normal(0, rw_sigma, n)) * np.sqrt(dt)
    vN_ins, vE_ins = vN + eN, vE + eE
    ins_lat, ins_lon = np.empty(n), np.empty(n)
    ins_lat[0], ins_lon[0] = lat[0], lon[0]
    for k in range(1, n):
        Rn, Re = meridian_radii(ins_lat[k - 1])
        ins_lat[k] = ins_lat[k - 1] + vN_ins[k - 1] * dt / (Rn + alt[k - 1])
        ins_lon[k] = ins_lon[k - 1] + vE_ins[k - 1] * dt / ((Re + alt[k - 1]) * np.cos(ins_lat[k - 1]))
    return ins_lat, ins_lon, vN_ins, vE_ins


def simulate_ins_pinson(lat, lon, alt, kin: dict, dt: float,
                        accel_bias: tuple = (3e-4, -2e-4, 1e-4),
                        gyro_bias: tuple = (3e-7, -2e-7, 1e-7),
                        seed: int = 0):
    """Generates INS drift by integrating the Pinson error with injected IMU biases.

    Returns (nominal, e_true), where nominal = (lat,lon,alt,vN,vE,vD) INS = truth - error,
    and e_true[15,N] is the true error state (for validation)."""
    n = len(lat)
    vN, vE, vD = kin["vN"], kin["vE"], kin["vD"]
    fn, fe, fd, Cnb = kin["fn"], kin["fe"], kin["fd"], kin["Cnb"]
    e = np.zeros((15, n))
    e[9:12, 0] = accel_bias
    e[12:15, 0] = gyro_bias
    for k in range(1, n):
        F = pinson_F(lat[k - 1], alt[k - 1], vN[k - 1], vE[k - 1], vD[k - 1],
                     fn[k - 1], fe[k - 1], fd[k - 1], Cnb[k - 1])
        e[:, k] = e[:, k - 1] + (F @ e[:, k - 1]) * dt
    nominal = dict(lat=lat - e[0], lon=lon - e[1], alt=alt - e[2],
                   vN=vN - e[3], vE=vE - e[4], vD=vD - e[5])
    return nominal, e
