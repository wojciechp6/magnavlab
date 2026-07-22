"""38-state, tightly coupled EKF with online calibration (Canciani 2022).

Modes:
  - ``mode="tightly"`` - T-L coefficients estimated online (Q_TL>0),
  - ``mode="loosely"`` - T-L frozen at batch values (baseline), FOGM bias only.

The measurement model is injected (by default :class:`TLAugmentedMeasurement`), so
it can easily be swapped for another variant without changing the filter engine.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..interfaces import NavFilter, NavProblem, NavResult
from ..models.measurement import TLAugmentedMeasurement
from ..models.pinson import (
    NX, N_TL, I_TL, I_S, I_V, augmented_F, augmented_Qd, pinson_F,
)

_DEFAULT_QP = dict(vrw=0.02, arw=1e-6, ba_rw=1e-6, bg_rw=1e-9)
_DEFAULT_QTL = dict(perm=0.05, ind=1e-4, eddy=0.0, offset=0.0)
_ZERO_QTL = dict(perm=0.0, ind=0.0, eddy=0.0, offset=0.0)
_DEFAULT_P0 = dict(pos=30.0, vel=1.0, tilt=1e-3, ba=1e-3, bg=1e-7)


class Canciani38EKF(NavFilter):
    """Extended Kalman filter with 38 states (INS Pinson + T-L online + FOGM + vector).

    ``Qf`` is the assumed variance of the fluxgate (vector-magnetometer) states and is the filter's
    main tuning knob. The paper uses (3000 nT)^2 for the F-16's ~10,000 nT aircraft field; the
    default here is (200 nT)^2 for the magnetically-clean SGL Cessna, whose fluxgate is far
    cleaner. On this data the result is robust across that whole range (~56 m at 200^2 vs ~57 m at
    the paper's 3000^2; best ~51 m near 500^2) — see the comparison in notebook 04.
    """

    def __init__(self, mode: str = "tightly", R: float = 60.0**2, Qf: float = 200.0**2,
                 tau_wf: float = 600.0, sigma_wf: float = 20.0,
                 qP: dict | None = None, qTL: dict | None = None, P0: dict | None = None,
                 measurement_cls=TLAugmentedMeasurement):
        if mode not in ("tightly", "loosely"):
            raise ValueError("mode must be 'tightly' or 'loosely'")
        self.mode = mode
        self.R, self.Qf = R, Qf
        self.tau_wf, self.sigma_wf = tau_wf, sigma_wf
        self.qP = qP or _DEFAULT_QP
        self.qTL = _ZERO_QTL if mode == "loosely" else (qTL or _DEFAULT_QTL)
        self.P0 = P0 or _DEFAULT_P0
        self.measurement_cls = measurement_cls

    def _init_cov(self, lat0: float) -> np.ndarray:
        from ..geo import meridian_radii
        P = np.zeros((NX, NX))
        Rn0, Re0 = meridian_radii(lat0)
        P[0, 0] = (self.P0["pos"] / Rn0)**2
        P[1, 1] = (self.P0["pos"] / (Re0 * np.cos(lat0)))**2
        P[2, 2] = self.P0["pos"]**2
        P[3:6, 3:6] = np.eye(3) * self.P0["vel"]**2
        P[6:9, 6:9] = np.eye(3) * self.P0["tilt"]**2
        P[9:12, 9:12] = np.eye(3) * self.P0["ba"]**2
        P[12:15, 12:15] = np.eye(3) * self.P0["bg"]**2
        P[I_S, I_S] = self.sigma_wf**2
        P[35:38, 35:38] = np.eye(3) * self.Qf
        return P

    def run(self, problem: NavProblem) -> NavResult:
        p = problem
        n, dt = p.n, p.dt
        if p.flux is None or p.core is None or p.tl0 is None:
            raise ValueError("EKF38 requires NavProblem with flux, core, cos_dot, tl0, P_tl0.")
        meas = self.measurement_cls(p.map, p.core, p.lat, p.lon, p.meas, p.cos_dot)
        Bx, By, Bz = p.flux.x, p.flux.y, p.flux.z

        x = np.zeros(NX)
        x[I_TL] = p.tl0
        P = self._init_cov(p.lat[0])
        if self.mode == "loosely":
            P[15:34, 15:34] = 0.0                     # T-L frozen
        else:
            P[15:34, 15:34] = p.P_tl0                 # from batch (eq. 23)

        Qd = augmented_Qd(dt, self.qP, self.qTL, self.sigma_wf, self.tau_wf)
        lat_est = np.empty(n); lon_est = np.empty(n)
        tl_hist = np.empty((N_TL, n)); S_hist = np.empty(n)

        for k in range(n):
            if k > 0:
                F15 = pinson_F(p.lat[k-1], p.alt[k-1], p.vN[k-1], p.vE[k-1], p.vD[k-1],
                               p.fn[k-1], p.fe[k-1], p.fd[k-1], p.Cnb[k-1])
                Phi = expm(augmented_F(F15, self.tau_wf) * dt)
                x[:35] = Phi @ x[:35]
                P[:35, :35] = Phi @ P[:35, :35] @ Phi.T + Qd
            # vector states: overwrite with fluxgate measurement, covariance Qf
            x[I_V] = [Bx[k], By[k], Bz[k]]
            P[35:38, :] = 0.0; P[:, 35:38] = 0.0
            P[35:38, 35:38] = np.eye(3) * self.Qf

            h = meas.h(x, k)
            if not np.isfinite(h):
                lat_est[k], lon_est[k] = p.lat[k] + x[0], p.lon[k] + x[1]
                tl_hist[:, k], S_hist[k] = x[I_TL], x[I_S]
                continue
            H = meas.H(x, k)
            S = float(H @ P @ H.T + self.R)
            K = (P @ H) / S
            x = x + K * (p.meas[k] - h)
            P = (np.eye(NX) - np.outer(K, H)) @ P

            lat_est[k], lon_est[k] = p.lat[k] + x[0], p.lon[k] + x[1]
            tl_hist[:, k], S_hist[k] = x[I_TL], x[I_S]

        return NavResult(lat=lat_est, lon=lon_est,
                         extras={"tl": tl_hist, "S": S_hist})
