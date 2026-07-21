import numpy as np

from magnavlab.models import measurement as ms
from magnavlab.models.pinson import NX


def test_tl_row_length():
    r = ms.tl_row(1.0, 2.0, 3.0, 53000.0, 0.1, -0.2, 0.05)
    assert r.shape == (19,)
    arr = ms.tl_row(np.ones(5), np.ones(5), np.ones(5), np.full(5, 53000.0),
                    np.zeros(5), np.zeros(5), np.zeros(5))
    assert arr.shape == (5, 19)


def test_jacobian_V_matches_finite_difference():
    coef = np.linspace(-1, 1, 19)
    Bx, By, Bz, z = 1200.0, -800.0, 400.0, 53000.0
    cXd, cYd, cZd = 0.02, -0.03, 0.01
    dBx, dBy, dBz = ms.tl_jacobian_V(coef, Bx, By, Bz, z, cXd, cYd, cZd)
    h = 1e-2

    def val(bx, by, bz):
        return ms.tl_row(bx, by, bz, z, cXd, cYd, cZd) @ coef

    num_x = (val(Bx + h, By, Bz) - val(Bx - h, By, Bz)) / (2 * h)
    num_y = (val(Bx, By + h, Bz) - val(Bx, By - h, Bz)) / (2 * h)
    num_z = (val(Bx, By, Bz + h) - val(Bx, By, Bz - h)) / (2 * h)
    assert num_x == np.float64(num_x) and abs(dBx - num_x) < 1e-6
    assert abs(dBy - num_y) < 1e-6
    assert abs(dBz - num_z) < 1e-6


def test_augmented_measurement_h_and_H(wavy_map, straight_traj):
    t = straight_traj
    n = t["n"]
    core = np.zeros(n)
    z = np.full(n, 53000.0)
    cos_dot = (np.zeros(n), np.zeros(n), np.zeros(n))
    meas = ms.TLAugmentedMeasurement(wavy_map, core, t["lat"], t["lon"], z, cos_dot)
    x = np.zeros(NX)
    x[35:38] = [1000.0, -500.0, 300.0]
    k = 100
    h = meas.h(x, k)
    expected = float(wavy_map.value(t["lat"][k], t["lon"][k])[0])  # tl0=0, S=0, core=0
    assert h == np.float64(h) and abs(h - expected) < 1e-9
    H = meas.H(x, k)
    assert H.shape == (NX,)
    d_lat, _ = wavy_map.gradient(t["lat"][k], t["lon"][k])
    assert H[0] == float(d_lat[0])
    assert H[34] == 1.0                                    # ∂/∂S
