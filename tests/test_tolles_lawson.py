import numpy as np
import pytest

from magnavlab.calibration import BuiltinTL, MapBasedModifiedTL, tl_design_matrix


def test_design_matrix_shape(synthetic_flux):
    flux = synthetic_flux(n=400)
    ref = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
    assert tl_design_matrix(flux, ref, 0.1).shape == (400, 18)
    assert tl_design_matrix(flux, ref, 0.1, add_offset=True).shape == (400, 19)


def test_mapbased_recovers_known_coefficients(synthetic_flux):
    """Map-based: if target = z − A·coef_true, then the fit recovers coef_true."""
    flux = synthetic_flux(n=600, seed=3)
    dt = 0.1
    z = 53000.0 + 20.0 * np.sin(np.linspace(0, 8, 600))       # scalar (Bt)
    A = tl_design_matrix(flux, z, dt, add_offset=True)         # modified (ref=z)
    rng = np.random.default_rng(1)
    coef_true = rng.normal(0, 1.0, 19)
    disturbance = A @ coef_true
    target = z - disturbance                                   # Earth's field

    cal = MapBasedModifiedTL().fit(flux, z, dt, target=target)
    assert np.allclose(cal.coef, coef_true, atol=1e-6)
    assert cal.P_cov.shape == (19, 19)
    # compensation removes the aircraft field (returns to the target, up to a constant)
    comp = cal.compensate(flux, z, dt)
    resid = (comp - comp.mean()) - (target - target.mean())
    assert np.std(resid) < 0.05 * np.std(target - target.mean()) + 1e-6


def test_mapbased_requires_target(synthetic_flux):
    with pytest.raises(ValueError):
        MapBasedModifiedTL().fit(synthetic_flux(n=100), np.ones(100), 0.1)


def test_builtin_reduces_aircraft_field(synthetic_flux):
    """Map-less BPF: compensation reduces the aircraft field correlated with maneuvers."""
    flux = synthetic_flux(n=1500, seed=5)
    dt = 0.1
    ref = np.sqrt(flux.x**2 + flux.y**2 + flux.z**2)
    A = tl_design_matrix(flux, ref, dt)
    rng = np.random.default_rng(2)
    coef = rng.normal(0, 0.5, 18)
    earth = 53000.0 + np.cumsum(rng.normal(0, 0.01, 1500))     # slow Earth's field
    aircraft = A @ coef
    scalar = earth + aircraft
    comp = BuiltinTL().fit(flux, scalar, dt).compensate(flux, scalar, dt)
    # after compensation, closer to Earth's field than the raw measurement (in the maneuver band)
    from magnavlab.calibration import bandpass
    fs = 1 / dt
    assert np.std(bandpass(comp - earth, fs)) < np.std(bandpass(scalar - earth, fs))
