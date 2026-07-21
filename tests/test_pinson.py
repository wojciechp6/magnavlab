import numpy as np

from magnavlab.geo import dcm_from_euler, meridian_radii
from magnavlab.models import pinson


def _sample_F():
    lat, alt = np.radians(45.0), 400.0
    Cnb = dcm_from_euler(0.0, 0.0, 0.5)
    return pinson.pinson_F(lat, alt, 60.0, -30.0, 0.0, 0.1, 0.0, -9.81, Cnb), lat, alt


def test_pinson_F_structure():
    F, lat, alt = _sample_F()
    Rn, Re = meridian_radii(lat)
    assert F.shape == (15, 15)
    assert F[0, 3] == np.float64(1.0 / (Rn + alt))       # δlat <- δvN
    assert F[2, 5] == -1.0                                # δh  <- -δvD
    # velocity depends on tilts and accelerometer bias
    assert not np.allclose(F[3:6, 6:9], 0.0)
    assert np.allclose(F[3:6, 9:12], dcm_from_euler(0.0, 0.0, 0.5))


def test_augmented_shapes():
    F15, *_ = _sample_F()
    F = pinson.augmented_F(F15, tau_wf=600.0)
    assert F.shape == (35, 35)
    assert F[pinson.I_S, pinson.I_S] == -1.0 / 600.0
    qP = dict(vrw=0.02, arw=1e-6, ba_rw=1e-6, bg_rw=1e-9)
    qTL = dict(perm=0.05, ind=1e-4, eddy=0.0, offset=0.0)
    Qd = pinson.augmented_Qd(0.5, qP, qTL, sigma_wf=20.0, tau_wf=600.0)
    assert Qd.shape == (35, 35)
    assert np.all(np.diag(Qd) >= 0)
    assert Qd[15, 15] > 0 and Qd[24, 24] == 0.0          # perm > 0, eddy = 0


def test_covariance_grows_without_measurement():
    from scipy.linalg import expm
    F15, *_ = _sample_F()
    F = pinson.augmented_F(F15, 600.0)
    dt = 0.5
    Phi = expm(F * dt)
    qP = dict(vrw=0.05, arw=1e-5, ba_rw=1e-5, bg_rw=1e-8)
    qTL = dict(perm=0.1, ind=1e-3, eddy=0.0, offset=0.0)
    Qd = pinson.augmented_Qd(dt, qP, qTL, 20.0, 600.0)
    P = np.eye(35) * 1e-6
    tr0 = np.trace(P)
    for _ in range(50):
        P = Phi @ P @ Phi.T + Qd
    assert np.trace(P) > tr0                              # uncertainty grows
