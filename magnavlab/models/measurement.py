"""Filter measurement models (interchangeable via the MeasurementModel protocol).

Contains the modified Tolles-Lawson model plugged directly into the measurement equation
(tightly-coupled per Canciani 2022):

    z = map(lat_c, lon_c) + core_field + C(TL, V) + S
"""
from __future__ import annotations

import numpy as np

from ..interfaces import MapLike
from .pinson import I_TL, I_S, NX


def tl_row(Bx, By, Bz, z, cXd, cYd, cZd):
    """Row C(TL,V): 19 terms of modified T-L (cosines from the scalar z). Scalar or arrays."""
    cX, cY, cZ = Bx / z, By / z, Bz / z
    Bt = z
    cols = [cX, cY, cZ,
            Bt * cX * cX, Bt * cX * cY, Bt * cX * cZ,
            Bt * cY * cY, Bt * cY * cZ, Bt * cZ * cZ,
            Bt * cX * cXd, Bt * cX * cYd, Bt * cX * cZd,
            Bt * cY * cXd, Bt * cY * cYd, Bt * cY * cZd,
            Bt * cZ * cXd, Bt * cZ * cYd, Bt * cZ * cZd,
            np.ones_like(cX) if np.ndim(cX) else 1.0]
    return np.stack(cols, axis=-1) if np.ndim(cX) else np.array(cols)


def tl_jacobian_V(coef, Bx, By, Bz, z, cXd, cYd, cZd):
    """Measurement derivatives w.r.t. the vector states [Bx,By,Bz] (eqs. 28-30 of the paper)."""
    c, Bt = coef, z
    dBx = c[0] / z + Bt * (2 * Bx / z**2 * c[3] + By / z**2 * c[4] + Bz / z**2 * c[5]
                           + cXd / z * c[9] + cYd / z * c[10] + cZd / z * c[11])
    dBy = c[1] / z + Bt * (Bx / z**2 * c[4] + 2 * By / z**2 * c[6] + Bz / z**2 * c[7]
                           + cXd / z * c[12] + cYd / z * c[13] + cZd / z * c[14])
    dBz = c[2] / z + Bt * (Bx / z**2 * c[5] + By / z**2 * c[7] + 2 * Bz / z**2 * c[8]
                           + cXd / z * c[15] + cYd / z * c[16] + cZd / z * c[17])
    return dBx, dBy, dBz


class TLAugmentedMeasurement:
    """Measurement model for EKF38 (implements the MeasurementModel protocol).

    The vector states [Bx,By,Bz] (indices 35:38) are overwritten with the fluxgate
    measurement by the filter; here we use them to compute h and H at each step.
    """

    def __init__(self, mag_map: MapLike, core: np.ndarray,
                 nom_lat: np.ndarray, nom_lon: np.ndarray,
                 z: np.ndarray, cos_dot: tuple):
        self.map = mag_map
        self.core = core
        self.nom_lat, self.nom_lon = nom_lat, nom_lon
        self.z = z
        self.cXd, self.cYd, self.cZd = cos_dot

    def _corrected_pos(self, state, k):
        return self.nom_lat[k] + state[0], self.nom_lon[k] + state[1]

    def h(self, state: np.ndarray, k: int) -> float:
        lat_c, lon_c = self._corrected_pos(state, k)
        m_val = float(self.map.value(lat_c, lon_c)[0])
        if not np.isfinite(m_val):
            return np.nan
        row = tl_row(state[35], state[36], state[37], self.z[k],
                     self.cXd[k], self.cYd[k], self.cZd[k])
        return m_val + self.core[k] + float(row @ state[I_TL]) + state[I_S]

    def H(self, state: np.ndarray, k: int) -> np.ndarray:
        lat_c, lon_c = self._corrected_pos(state, k)
        d_lat, d_lon = self.map.gradient(lat_c, lon_c)
        H = np.zeros(NX)
        H[0], H[1] = float(d_lat[0]), float(d_lon[0])
        H[I_TL] = tl_row(state[35], state[36], state[37], self.z[k],
                         self.cXd[k], self.cYd[k], self.cZd[k])
        H[I_S] = 1.0
        H[35], H[36], H[37] = tl_jacobian_V(state[I_TL], state[35], state[36], state[37],
                                            self.z[k], self.cXd[k], self.cYd[k], self.cZd[k])
        return H
