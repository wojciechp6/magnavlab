"""Core (main) geomagnetic field from the IGRF model.

Anomaly maps have the core field removed, so the scalar measurement model must add it back
(Canciani eq. 25 uses the WMM; IGRF plays the same role and agrees with it to within tens of nT).
This lets the pipeline get the core field from a geophysical model rather than from a clean
reference magnetometer.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

# SGL 2020 survey epoch; the core field changes negligibly over a single campaign.
DEFAULT_EPOCH = datetime(2020, 6, 29)


def core_field(lat_rad, lon_rad, alt_m, when: datetime = DEFAULT_EPOCH) -> np.ndarray:
    """IGRF core-field total intensity [nT] along a trajectory.

    ``lat_rad``/``lon_rad`` are in radians and ``alt_m`` in metres. Requires the optional
    ``ppigrf`` package (installed with the ``dev`` extra)."""
    try:
        import ppigrf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("core_field needs 'ppigrf' - install with: pip install -e '.[dev]'") from exc
    lat = np.degrees(np.atleast_1d(np.asarray(lat_rad, dtype=float)))
    lon = np.degrees(np.atleast_1d(np.asarray(lon_rad, dtype=float)))
    h_km = np.atleast_1d(np.asarray(alt_m, dtype=float)) / 1000.0
    Be, Bn, Bu = ppigrf.igrf(lon, lat, h_km, when)
    return np.sqrt(np.asarray(Be) ** 2 + np.asarray(Bn) ** 2 + np.asarray(Bu) ** 2).ravel()
