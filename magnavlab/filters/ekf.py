"""Simple terrain-navigation EKF: state [lat, lon, evN, evE, bias].

Estimates position and the INS velocity error (evN, evE), so it actively compensates for drift,
not merely reacts to it. Measurement: matching a single anomaly value to the map.
"""
from __future__ import annotations

import numpy as np

from ..geo import meridian_radii
from ..interfaces import NavFilter, NavProblem, NavResult


class EKFNav(NavFilter):
    """5-state extended Kalman filter (position + velocity error + measurement bias)."""

    def __init__(self, sigma_meas: float = 60.0, sigma_vel: float = 1e-3,
                 sigma_bias_rw: float = 0.05,
                 p0_pos: float = 30.0, p0_vel: float = 0.5, p0_bias: float = 50.0):
        self.sigma_meas = sigma_meas
        self.sigma_vel = sigma_vel
        self.sigma_bias_rw = sigma_bias_rw
        self.p0 = (p0_pos, p0_pos, p0_vel, p0_bias)

    def run(self, problem: NavProblem) -> NavResult:
        p = problem
        n, dt = p.n, p.dt
        alt = p.alt
        x = np.array([p.lat[0], p.lon[0], 0.0, 0.0, 0.0])
        Rn0, Re0 = meridian_radii(x[0])
        P = np.diag([(self.p0[0] / Rn0)**2, (self.p0[1] / (Re0 * np.cos(x[0])))**2,
                     self.p0[2]**2, self.p0[2]**2, self.p0[3]**2])
        lat_est = np.empty(n); lon_est = np.empty(n); bias_est = np.empty(n)

        for k in range(n):
            if k > 0:
                Rn, Re = meridian_radii(x[0])
                f_lat = dt / (Rn + alt[k - 1])
                f_lon = dt / ((Re + alt[k - 1]) * np.cos(x[0]))
                x[0] += (p.vN[k - 1] - x[2]) * f_lat
                x[1] += (p.vE[k - 1] - x[3]) * f_lon
                F = np.eye(5)
                F[0, 2] = -f_lat
                F[1, 3] = -f_lon
                Q = np.diag([(0.02 * f_lat)**2, (0.02 * f_lon)**2,
                             self.sigma_vel**2 * dt, self.sigma_vel**2 * dt,
                             self.sigma_bias_rw**2 * dt])
                P = F @ P @ F.T + Q

            m_val = float(p.map.value(x[0], x[1])[0])
            if not np.isfinite(m_val):
                lat_est[k], lon_est[k], bias_est[k] = x[0], x[1], x[4]
                continue
            d_lat, d_lon = p.map.gradient(x[0], x[1])
            H = np.array([float(d_lat[0]), float(d_lon[0]), 0.0, 0.0, 1.0])
            innov = p.meas[k] - (m_val + x[4])
            S = float(H @ P @ H.T + self.sigma_meas**2)
            K = (P @ H) / S
            x = x + K * innov
            P = (np.eye(5) - np.outer(K, H)) @ P
            lat_est[k], lon_est[k], bias_est[k] = x[0], x[1], x[4]

        return NavResult(lat=lat_est, lon=lon_est, extras={"bias": bias_est})
