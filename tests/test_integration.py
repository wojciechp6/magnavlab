"""Integration tests on real SGL data (skipped when files are missing)."""
import os

import pytest

DATA = ["data/Flt1003_train.h5", "data/Flt1002_train.h5", "data/maps/Eastern_395.h5"]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in DATA),
    reason="Missing SGL/map data in data/ - download per the README.")


def test_canciani_tightly_beats_loosely():
    from magnavlab.experiments import CancianiConfig, run_canciani
    out = run_canciani(CancianiConfig(make_plots=False))
    m = out["metrics"]
    assert m["INS"]["drms"] > 1000.0                     # INS drifts by km
    assert m["loosely"]["drms"] < m["INS"]["drms"]
    assert m["tightly"]["drms"] < m["loosely"]["drms"]   # online calibration helps
    assert m["tightly"]["drms"] < 120.0                  # consistent with ~57 m


def test_simple_nav_ekf_beats_ins():
    from magnavlab.experiments import SimpleNavConfig, run_simple_nav
    out = run_simple_nav(SimpleNavConfig(make_plots=False))
    m = out["metrics"]
    assert m["EKF"]["drms"] < m["INS"]["drms"]
    assert m["EKF"]["drms"] < 200.0
