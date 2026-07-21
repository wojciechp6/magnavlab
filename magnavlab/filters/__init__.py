"""Navigation filters - interchangeable implementations of the NavFilter protocol."""
from .ekf import EKFNav
from .pf import ParticleFilterNav
from .ekf38 import Canciani38EKF

# Registry for selection in experiments.
FILTERS = {
    "ekf": EKFNav,
    "pf": ParticleFilterNav,
    "ekf38": Canciani38EKF,
}

__all__ = ["EKFNav", "ParticleFilterNav", "Canciani38EKF", "FILTERS"]
