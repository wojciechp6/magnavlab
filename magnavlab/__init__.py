"""magnavlab - modular library for magnetic navigation experiments.

Layers (each with interchangeable components):
    data        - dataset registry + download/prepare helpers (Zenodo, maps)
    io          - loading flights (HDF5) and anomaly maps
    geo         - WGS-84 geodesy, NED frame
    calibration - Tolles-Lawson compensation (interchangeable backends)
    models      - INS error dynamics (Pinson) and measurement models
    ins         - drifting INS simulation
    filters     - EKF, particle filter, Canciani's 38-state EKF
    metrics     - DRMS/CEP
    viz         - plots

The experiment pipelines live in the notebooks (notebooks/), assembled from these primitives.

Example (the building blocks):
    >>> from magnavlab.io import load_flight, load_map, segment_indices
    >>> from magnavlab.filters import Canciani38EKF   # any NavFilter takes a NavProblem
"""
from .interfaces import Calibrator, MagV, MapLike, NavFilter, NavProblem, NavResult
from .io import FlightData, MagMap, inspect_h5, load_flight, load_map, segment_indices

__version__ = "0.1.0"

__all__ = [
    "Calibrator", "MagV", "MapLike", "NavFilter", "NavProblem", "NavResult",
    "FlightData", "MagMap", "load_flight", "load_map", "segment_indices", "inspect_h5",
]
