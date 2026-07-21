"""Particle filter (bootstrap) for terrain navigation.

Particle: [lat, lon, evN, evE, bias] - as in the EKF, but the distribution is represented by samples,
which better handles map multimodality under large initial uncertainty.
"""
from __future__ import annotations

import numpy as np

from ..geo import meridian_radii
from ..interfaces import NavFilter, NavProblem, NavResult


class ParticleFilterNav(NavFilter):
    """Bootstrap PF with systematic resampling and velocity-error estimation."""

    def __init__(self, n_particles: int = 4000, sigma_meas: float = 60.0,
                 sigma_vel: float = 5e-3, sigma_bias_rw: float = 0.05,
                 init_pos_std: float = 30.0, init_vel_std: float = 0.4,
                 init_bias_std: float = 50.0, seed: int = 1):
        self.N = n_particles
        self.sigma_meas = sigma_meas
        self.sigma_vel = sigma_vel
        self.sigma_bias_rw = sigma_bias_rw
        self.init = (init_pos_std, init_vel_std, init_bias_std)
        self.seed = seed

    def run(self, problem: NavProblem) -> NavResult:
        p = problem
        n, dt, N = p.n, p.dt, self.N
        alt = p.alt
        rng = np.random.default_rng(self.seed)
        Rn0, Re0 = meridian_radii(p.lat[0])
        pos_std, vel_std, bias_std = self.init
        p_lat = p.lat[0] + rng.normal(0, pos_std / Rn0, N)
        p_lon = p.lon[0] + rng.normal(0, pos_std / (Re0 * np.cos(p.lat[0])), N)
        p_evN = rng.normal(0, vel_std, N)
        p_evE = rng.normal(0, vel_std, N)
        p_bias = rng.normal(0, bias_std, N)
        w = np.full(N, 1.0 / N)
        lat_est = np.empty(n); lon_est = np.empty(n); bias_est = np.empty(n)

        for k in range(n):
            if k > 0:
                Rn, Re = meridian_radii(p_lat)
                p_evN += rng.normal(0, self.sigma_vel, N) * np.sqrt(dt)
                p_evE += rng.normal(0, self.sigma_vel, N) * np.sqrt(dt)
                p_lat += (p.vN[k - 1] - p_evN) * dt / (Rn + alt[k - 1])
                p_lon += (p.vE[k - 1] - p_evE) * dt / ((Re + alt[k - 1]) * np.cos(p_lat))
                p_bias += rng.normal(0, self.sigma_bias_rw, N) * np.sqrt(dt)

            m_val = p.map.value(p_lat, p_lon)
            resid = p.meas[k] - (m_val + p_bias)
            logw = -0.5 * (resid / self.sigma_meas)**2
            logw[~np.isfinite(m_val)] = -1e9
            logw -= np.max(logw)
            w = w * np.exp(logw)
            s = w.sum()
            w = np.full(N, 1.0 / N) if (s <= 0 or not np.isfinite(s)) else w / s

            lat_est[k] = np.sum(w * p_lat)
            lon_est[k] = np.sum(w * p_lon)
            bias_est[k] = np.sum(w * p_bias)

            if 1.0 / np.sum(w**2) < N / 2:            # systematic resampling
                pos = (rng.random() + np.arange(N)) / N
                cum = np.cumsum(w); cum[-1] = 1.0
                idx = np.searchsorted(cum, pos)
                p_lat, p_lon = p_lat[idx], p_lon[idx]
                p_evN, p_evE, p_bias = p_evN[idx], p_evE[idx], p_bias[idx]
                w = np.full(N, 1.0 / N)

        return NavResult(lat=lat_est, lon=lon_est, extras={"bias": bias_est})
