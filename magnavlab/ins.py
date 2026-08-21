"""Trajectory kinematics and inertial-navigation (INS) trajectories.

  - :func:`real_ins`              - the aircraft's REAL onboard INS solution (already drifting),
  - :func:`simulate_ins_velocity` - synthetic INS drift at the velocity level (simple filters),
  - :func:`simulate_ins_pinson`   - synthetic INS drift from the Pinson error model with IMU
    biases (consistent with the EKF38 filter: the same F matrix generates the drift).

Prefer :func:`real_ins` for experiments on flights that record an onboard solution (SGL data);
the ``simulate_*`` helpers remain for controlled studies and for data without a recorded INS.
"""
from __future__ import annotations

import numpy as np

from .geo import G0, dcm_from_euler, meridian_radii, velocity_from_track
from .models.pinson import pinson_F


def _to_rad(a: np.ndarray) -> np.ndarray:
    """Auto-detect angle units: radians stay, degrees -> radians."""
    a = np.asarray(a, dtype=float)
    return a if np.nanmax(np.abs(a)) < 2 * np.pi + 1e-6 else np.radians(a)


def real_ins(fl, idx=None, anchor: tuple | None = None, dt: float | None = None) -> dict:
    """Real onboard INS trajectory - drop-in replacement for the ``simulate_ins_*`` drift.

    Reads the recorded onboard navigation solution (``ins_lat``, ``ins_lon``, ``ins_alt``,
    ``ins_vn``, ``ins_vw``, ``ins_vu``) - a genuine, already-drifting trajectory - instead of
    injecting a synthetic drift. Units/frames are normalized here:

      - ``ins_lat``/``ins_lon`` are stored in **radians** (unlike GPS ``lat``/``lon`` in degrees);
      - velocities are wander-frame north/up/west -> NED: ``vE = -ins_vw``, ``vD = -ins_vu``.

    Parameters
    ----------
    fl : FlightData
        Loaded flight; must contain the ``ins_*`` fields (present in SGL 2020/2021 data).
    idx : ndarray | slice | None
        Sub-selection (e.g. a navigation window, subsampled). ``None`` = whole flight.
    anchor : (lat0, lon0) | None
        Truth start point [rad]. If given, the horizontal position is shifted so the first
        sample coincides with it - a cold/warm start (INS position error begins at 0), matching
        how the onboard solution would be re-initialized at the start of a navigation run.
    dt : float | None
        Only needed as a fallback if recorded velocities are missing (then they are
        differentiated from the trajectory).

    Returns
    -------
    dict
        ``lat, lon, alt, vN, vE, vD, roll, pitch`` (radians / metres / m·s⁻¹) - the same keys
        as :func:`simulate_ins_pinson`'s ``ins`` output, plus INS attitude for kinematics.
    """
    def col(name: str) -> np.ndarray:
        a = fl.get(name)
        return a if idx is None else np.asarray(a)[idx]

    lat = _to_rad(col("ins_lat"))
    lon = _to_rad(col("ins_lon"))
    alt = col("ins_alt")
    if fl.has("ins_vn") and fl.has("ins_vw") and fl.has("ins_vu"):
        vN, vE, vD = col("ins_vn"), -col("ins_vw"), -col("ins_vu")
    elif dt is not None:
        vN, vE = velocity_from_track(lat, lon, alt, dt)
        vD = -np.gradient(alt, dt)
    else:
        raise ValueError("recorded INS velocities (ins_vn/ins_vw/ins_vu) missing; pass dt to "
                         "differentiate the trajectory instead.")
    # attitude is in degrees in this dataset; np.radians unconditionally (small angles would
    # be misread as radians by a magnitude test).
    roll = np.radians(col("roll")) if fl.has("roll") else np.zeros_like(lat)
    pitch = np.radians(col("pitch")) if fl.has("pitch") else np.zeros_like(lat)

    if anchor is not None:
        lat = lat - (lat[0] - anchor[0])
        lon = lon - (lon[0] - anchor[1])

    return dict(lat=lat, lon=lon, alt=alt, vN=vN, vE=vE, vD=vD, roll=roll, pitch=pitch)


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
                        accel_bias: tuple = (6e-5, -4e-5, 2e-5),
                        gyro_bias: tuple = (6e-8, -4e-8, 2e-8),
                        seed: int = 0):
    """Generate a drifting INS by integrating the Pinson error with constant IMU biases.

    The default biases give a realistic navigation-grade drift of roughly a kilometre over a
    ~1-hour flight. Returns (ins, error), where ``ins`` = (lat,lon,alt,vN,vE,vD) = truth - error,
    and ``error[15, N]`` is the true error state (for validation)."""
    n = len(lat)
    vN, vE, vD = kin["vN"], kin["vE"], kin["vD"]
    fn, fe, fd, Cnb = kin["fn"], kin["fe"], kin["fd"], kin["Cnb"]
    error = np.zeros((15, n))
    error[9:12, 0] = accel_bias
    error[12:15, 0] = gyro_bias
    for k in range(1, n):
        F = pinson_F(lat[k - 1], alt[k - 1], vN[k - 1], vE[k - 1], vD[k - 1],
                     fn[k - 1], fe[k - 1], fd[k - 1], Cnb[k - 1])
        error[:, k] = error[:, k - 1] + (F @ error[:, k - 1]) * dt
    ins = dict(lat=lat - error[0], lon=lon - error[1], alt=alt - error[2],
               vN=vN - error[3], vE=vE - error[4], vD=vD - error[5])
    return ins, error
