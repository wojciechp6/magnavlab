"""Tolles-Lawson aeromagnetic compensation - interchangeable variants.

The T-L model decomposes the aircraft field into 3 components (permanent/induced/eddy),
18 coefficients (+1 map offset in the map-based variant). Implemented:

  - :class:`BuiltinTL`          - classic map-less (band-pass) with ridge regression,
  - :class:`MapBasedModifiedTL` - "modified" map-based (Bt from the scalar, target = map+core).

All implement the common :class:`~magnavlab.interfaces.Calibrator` protocol.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from ..interfaces import Calibrator, MagV


# ---------------------------------------------------------------------------
# T-L design matrix
# ---------------------------------------------------------------------------
def tl_design_matrix(flux: MagV, ref: np.ndarray, dt: float,
                     cos_dot: tuple | None = None,
                     add_offset: bool = False) -> np.ndarray:
    """Build the T-L matrix (N x 18 or N x 19) in the order from Canciani's paper.

    ``ref`` is the reference quantity for the cosines and the induced/eddy terms:
    |B| for classic T-L, or the (atomic) scalar for the "modified" variant.
    """
    ref = np.where(ref == 0, 1e-9, ref)
    cX, cY, cZ = flux.x / ref, flux.y / ref, flux.z / ref
    if cos_dot is None:
        cXd, cYd, cZd = np.gradient(cX, dt), np.gradient(cY, dt), np.gradient(cZ, dt)
    else:
        cXd, cYd, cZd = cos_dot
    Bt = ref
    cols = [
        cX, cY, cZ,                                                     # permanent (3)
        Bt * cX * cX, Bt * cX * cY, Bt * cX * cZ,                       # induced (6)
        Bt * cY * cY, Bt * cY * cZ, Bt * cZ * cZ,
        Bt * cX * cXd, Bt * cX * cYd, Bt * cX * cZd,                    # eddy (9)
        Bt * cY * cXd, Bt * cY * cYd, Bt * cY * cZd,
        Bt * cZ * cXd, Bt * cZ * cYd, Bt * cZ * cZd,
    ]
    if add_offset:
        cols.append(np.ones_like(cX))
    return np.column_stack(cols)


def bandpass(x: np.ndarray, fs: float, lo: float = 0.1, hi: float = 0.9,
             order: int = 4) -> np.ndarray:
    """Butterworth band-pass filter (zero-phase) along axis 0."""
    nyq = 0.5 * fs
    b, a = butter(order, [max(lo / nyq, 1e-4), min(hi / nyq, 0.999)], btype="band")
    return filtfilt(b, a, x, axis=0)


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------
class BuiltinTL(Calibrator):
    """Classic map-less Tolles-Lawson (band-pass + ridge regression)."""

    def __init__(self, lo: float = 0.1, hi: float = 0.9, ridge: float = 1e-3):
        self.lo, self.hi, self.ridge = lo, hi, ridge
        self.coef: np.ndarray | None = None

    def fit(self, flux, scalar, dt, target=None):
        ref = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
        A = tl_design_matrix(flux, ref, dt)
        fs = 1.0 / dt
        A_f, y_f = bandpass(A, fs, self.lo, self.hi), bandpass(scalar, fs, self.lo, self.hi)
        AtA = A_f.T @ A_f
        lam = self.ridge * np.trace(AtA) / AtA.shape[0]
        self.coef = np.linalg.solve(AtA + lam * np.eye(AtA.shape[0]), A_f.T @ y_f)
        return self

    def compensate(self, flux, scalar, dt):
        ref = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
        aircraft = tl_design_matrix(flux, ref, dt) @ self.coef
        comp = scalar - aircraft
        return comp - comp.mean() + scalar.mean()


class MapBasedModifiedTL(Calibrator):
    """Modified map-based T-L (Bt = scalar; target = Earth field = map + core).

    After fitting it exposes ``coef`` (19) and ``P_cov`` (19x19) for initializing
    the T-L states in the EKF38 filter (covariance = (AᵀA)⁻¹·var(residuals))."""

    def __init__(self):
        self.coef: np.ndarray | None = None      # 19 (18 + offset)
        self.P_cov: np.ndarray | None = None
        self.resid_std: float = np.nan

    def fit(self, flux, scalar, dt, target=None):
        if target is None:
            raise ValueError("MapBasedModifiedTL requires 'target' (Earth field = map+core).")
        A = tl_design_matrix(flux, scalar, dt, add_offset=True)   # N x 19, modified
        d = scalar - target                                       # disturbance to be modeled
        self.coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        resid = d - A @ self.coef
        self.resid_std = float(np.std(resid))
        self.P_cov = np.linalg.pinv(A.T @ A) * float(np.var(resid))
        return self

    def compensate(self, flux, scalar, dt):
        # aircraft field without the constant offset (cols 0..17)
        A = tl_design_matrix(flux, scalar, dt)
        aircraft = A @ self.coef[:18]
        comp = scalar - aircraft
        return comp - comp.mean() + scalar.mean()


# Registry for easy selection in experiments.
CALIBRATORS: dict[str, type[Calibrator]] = {
    "builtin": BuiltinTL,
    "mapbased": MapBasedModifiedTL,
}
