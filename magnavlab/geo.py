"""Geodesy and helper transforms (WGS-84, NED frame).

All functions are pure (stateless), operate on NumPy arrays, and are suitable
for direct unit testing.
"""
from __future__ import annotations

import numpy as np

# --- WGS-84 ellipsoid ---
WGS84_A: float = 6378137.0                       # semi-major axis [m]
WGS84_F: float = 1.0 / 298.257223563             # flattening
WGS84_E2: float = WGS84_F * (2.0 - WGS84_F)      # eccentricity squared
OMEGA_IE: float = 7.292115e-5                    # Earth angular rate [rad/s]
G0: float = 9.80665                              # standard gravity [m/s^2]


def meridian_radii(lat_rad: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """Ellipsoid radii of curvature for a given latitude.

    Returns (Rn, Re): meridian and prime vertical [m].
    """
    s = np.sin(lat_rad)
    denom = 1.0 - WGS84_E2 * s * s
    Rn = WGS84_A * (1.0 - WGS84_E2) / denom**1.5
    Re = WGS84_A / np.sqrt(denom)
    return Rn, Re


def velocity_from_track(lat_rad: np.ndarray, lon_rad: np.ndarray,
                        alt: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """North/east velocities [m/s] from differentiating the geographic trajectory."""
    Rn, Re = meridian_radii(lat_rad)
    vN = np.gradient(lat_rad, dt) * (Rn + alt)
    vE = np.gradient(lon_rad, dt) * (Re + alt) * np.cos(lat_rad)
    return vN, vE


def horizontal_error(lat1: np.ndarray, lon1: np.ndarray,
                     lat2: np.ndarray, lon2: np.ndarray,
                     lat_ref: np.ndarray | float) -> np.ndarray:
    """Horizontal error [m] between two trajectories (local flat approximation)."""
    Rn, Re = meridian_radii(lat_ref)
    dN = (lat1 - lat2) * Rn
    dE = (lon1 - lon2) * Re * np.cos(lat_ref)
    return np.hypot(dN, dE)


def ned_offset(lat: np.ndarray, lon: np.ndarray,
               lat_ref: np.ndarray, lon_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """North/east offset [m] of point (lat,lon) relative to (lat_ref,lon_ref)."""
    Rn, Re = meridian_radii(lat_ref)
    dN = (lat - lat_ref) * Rn
    dE = (lon - lon_ref) * Re * np.cos(lat_ref)
    return dN, dE


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix [v×] such that skew(v) @ w == np.cross(v, w)."""
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def dcm_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Direction cosine matrix body->NED from Euler angles (ZYX) [rad]."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        [-sp,     sr * cp,                cr * cp],
    ])
