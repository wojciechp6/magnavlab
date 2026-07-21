"""Pinson INS error model and augmented-state dynamics (EKF38).

38-state layout (as in Canciani 2022):
  [ δρ(3) δv(3) ε(3) b_a(3) b_g(3) | TL(19) | S(1) | V(3) ]
The Pinson block contains the dominant coupling terms (position<-velocity,
velocity<-tilts and accel. bias, tilts<-Earth angular rate and gyro bias).
Minor transport-rate/Schuler terms are omitted (negligible over a flight of tens of minutes).
"""
from __future__ import annotations

import numpy as np

from ..geo import OMEGA_IE, meridian_radii, skew

# --- state indices ---
I_POS = slice(0, 3)
I_VEL = slice(3, 6)
I_TILT = slice(6, 9)
I_BA = slice(9, 12)
I_BG = slice(12, 15)
I_TL = slice(15, 34)      # 19 coefficients; offset = 33
I_OFFSET = 33
I_S = 34
I_V = slice(35, 38)
NX = 38
N_TL = 19
N_CORE = 35               # Pinson+TL+S block (without vector states)


def pinson_F(lat: float, alt: float, vN: float, vE: float, vD: float,
             fn: float, fe: float, fd: float, Cnb: np.ndarray) -> np.ndarray:
    """Continuous INS error matrix (15x15) in the NED frame."""
    F = np.zeros((15, 15))
    Rn, Re = meridian_radii(lat)
    # position <- velocity
    F[0, 3] = 1.0 / (Rn + alt)
    F[1, 4] = 1.0 / ((Re + alt) * np.cos(lat))
    F[2, 5] = -1.0
    # velocity <- tilts (-[f×]) and accelerometer bias (Cnb)
    F[3:6, 6:9] = -skew(np.array([fn, fe, fd]))
    F[3:6, 9:12] = Cnb
    # tilts <- Earth angular rate and gyroscope bias (-Cnb)
    w_ie = OMEGA_IE * np.array([np.cos(lat), 0.0, -np.sin(lat)])
    F[6:9, 6:9] = -skew(w_ie)
    F[6:9, 12:15] = -Cnb
    return F


def augmented_F(F15: np.ndarray, tau_wf: float) -> np.ndarray:
    """F matrix of the 35x35 block (Pinson + TL[Brownian motion: 0] + S[FOGM])."""
    F = np.zeros((N_CORE, N_CORE))
    F[:15, :15] = F15
    F[I_S, I_S] = -1.0 / tau_wf
    return F


def augmented_Qd(dt: float, qP: dict, qTL: dict, sigma_wf: float, tau_wf: float) -> np.ndarray:
    """Discrete process noise of the 35-block (Q·dt)."""
    Q = np.zeros((N_CORE, N_CORE))
    Q[3:6, 3:6] = np.eye(3) * qP["vrw"]**2       # velocity random walk
    Q[6:9, 6:9] = np.eye(3) * qP["arw"]**2       # angular random walk
    Q[9:12, 9:12] = np.eye(3) * qP["ba_rw"]**2   # accel. bias
    Q[12:15, 12:15] = np.eye(3) * qP["bg_rw"]**2 # gyro bias
    tl = np.zeros(N_TL)
    tl[0:3] = qTL["perm"]**2
    tl[3:9] = qTL["ind"]**2
    tl[9:18] = qTL["eddy"]**2
    tl[18] = qTL["offset"]**2
    Q[15:34, 15:34] = np.diag(tl)
    Q[I_S, I_S] = 2.0 * sigma_wf**2 / tau_wf     # FOGM
    return Q * dt
