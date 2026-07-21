"""Protocols and data structures that tie the architecture together.

They define *interchangeable* extension points:
  - :class:`MapLike`        - anomaly map (value + gradient),
  - :class:`Calibrator`     - aeromagnetic compensation method,
  - :class:`NavFilter`      - navigation algorithm (EKF, PF, EKF38, ...),
  - :class:`MeasurementModel` / :class:`DynamicsModel` - filter components.

Thanks to uniform interfaces, experiments can swap individual building blocks
(e.g. a different calibrator or a different filter) without changing the rest of the pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class MagV:
    """Vector magnetometer (fluxgate) measurement [nT]."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


@runtime_checkable
class MapLike(Protocol):
    """Magnetic anomaly map with value and gradient interpolation."""

    alt: float

    def value(self, lat_rad, lon_rad) -> np.ndarray:
        """Map field value [nT] at point(s) (lat, lon) [rad]."""
        ...

    def gradient(self, lat_rad, lon_rad) -> tuple[np.ndarray, np.ndarray]:
        """Gradient (∂/∂lat, ∂/∂lon) [nT/rad]."""
        ...

    def extent_deg(self) -> tuple[float, float, float, float]:
        """(lon_min, lon_max, lat_min, lat_max) in degrees - for plots."""
        ...


@dataclass
class NavProblem:
    """Complete set of inputs for the navigation filter.

    The nominal trajectory is usually a drifting INS; the filter estimates its error
    using the magnetic measurement matched to ``map``.
    """
    dt: float
    map: MapLike
    # nominal trajectory (INS)
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    vN: np.ndarray
    vE: np.ndarray
    vD: np.ndarray
    # scalar measurement fed to the filter
    #   - simple filters: field in anomaly scale (offset removed),
    #   - EKF38: raw scalar (total field).
    meas: np.ndarray
    # --- optional (for filters with online calibration / Pinson model) ---
    core: np.ndarray | None = None            # core field [nT] along the track
    flux: MagV | None = None                  # vector magnetometer
    cos_dot: tuple | None = None              # (cXd, cYd, cZd) cosine derivatives
    fn: np.ndarray | None = None              # specific force N [m/s^2]
    fe: np.ndarray | None = None
    fd: np.ndarray | None = None
    Cnb: np.ndarray | None = None             # DCM body->NED, shape (N,3,3)
    tl0: np.ndarray | None = None             # initial T-L coefficients (19)
    P_tl0: np.ndarray | None = None           # initial T-L covariance (19x19)

    @property
    def n(self) -> int:
        return len(self.lat)


@dataclass
class NavResult:
    """Filter result: estimated trajectory + additional time series."""
    lat: np.ndarray
    lon: np.ndarray
    extras: dict = field(default_factory=dict)


class Calibrator(ABC):
    """Aeromagnetic compensation method (removing the aircraft field).

    Contract: :meth:`fit` learns coefficients on a calibration segment,
    :meth:`compensate` returns the compensated scalar signal for the flight.
    """

    @abstractmethod
    def fit(self, flux: MagV, scalar: np.ndarray, dt: float,
            target: np.ndarray | None = None) -> "Calibrator":
        """Fit the coefficients. ``target`` = Earth field (map-based) or None (map-less)."""

    @abstractmethod
    def compensate(self, flux: MagV, scalar: np.ndarray, dt: float) -> np.ndarray:
        """Return the compensated scalar signal [nT]."""


class NavFilter(ABC):
    """Magnetic navigation algorithm. Takes a :class:`NavProblem`."""

    @abstractmethod
    def run(self, problem: NavProblem) -> NavResult:
        ...


class MeasurementModel(Protocol):
    """Filter measurement model: prediction h(x) and Jacobian H(x) at step k."""

    def h(self, state: np.ndarray, k: int) -> float: ...
    def H(self, state: np.ndarray, k: int) -> np.ndarray: ...


class DynamicsModel(Protocol):
    """Error dynamics model: continuous matrix F(k) for discretization."""

    def F(self, k: int) -> np.ndarray: ...
