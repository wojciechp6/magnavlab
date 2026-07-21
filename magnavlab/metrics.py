"""Navigation accuracy metrics (horizontal error in meters)."""
from __future__ import annotations

import numpy as np

from .geo import horizontal_error
from .interfaces import NavResult


def error_series(result: NavResult, lat_t: np.ndarray, lon_t: np.ndarray) -> np.ndarray:
    """Horizontal error series [m] of the estimate relative to the true trajectory."""
    return horizontal_error(result.lat, result.lon, lat_t, lon_t, lat_t)


def drms(result: NavResult, lat_t: np.ndarray, lon_t: np.ndarray) -> float:
    """DRMS = sqrt(mean(dN^2)+mean(dE^2)) = root mean square of the position error."""
    return float(np.sqrt(np.mean(error_series(result, lat_t, lon_t) ** 2)))


def summary(result: NavResult, lat_t: np.ndarray, lon_t: np.ndarray) -> dict:
    """Metrics dictionary: DRMS/RMSE, CEP50, CEP95, max, final value [m]."""
    e = error_series(result, lat_t, lon_t)
    return {
        "drms": float(np.sqrt(np.mean(e**2))),
        "cep50": float(np.percentile(e, 50)),
        "cep95": float(np.percentile(e, 95)),
        "max": float(np.max(e)),
        "final": float(e[-1]),
    }
