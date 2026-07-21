#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for Canciani's 38-state EKF (loosely vs tightly), on SGL data.

Thin wrapper around ``magnavlab`` (logic in magnavlab/experiments.py). Run:

    python magnav_ekf38.py                    # default: with injected body-field drift
    python magnav_ekf38.py --inject-drift 0   # without injected drift
"""
from __future__ import annotations

import argparse

from magnavlab.experiments import CancianiConfig, run_canciani
from magnavlab.geo import horizontal_error


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="38-state tightly-coupled MagNav EKF (Canciani 2022) on SGL data.")
    p.add_argument("--nav-file", default="data/Flt1003_train.h5")
    p.add_argument("--tstart", type=float, default=50713.0)
    p.add_argument("--tend", type=float, default=54497.0)
    p.add_argument("--map-file", default="data/maps/Eastern_395.h5")
    p.add_argument("--scalar-mag", default="mag_4_uc")
    p.add_argument("--decimate", type=int, default=5)
    p.add_argument("--inject-drift", type=float, default=0.6,
                   help="σ of body-field drift [nT/√s] (F-16 emulation; 0 = disable).")
    p.add_argument("--R", type=float, default=60.0)
    p.add_argument("--Qf", type=float, default=200.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="outputs")
    a = p.parse_args(argv)

    cfg = CancianiConfig(nav_file=a.nav_file, tstart=a.tstart, tend=a.tend,
                         map_file=a.map_file, scalar_mag=a.scalar_mag, decimate=a.decimate,
                         inject_drift=a.inject_drift, R=a.R, Qf=a.Qf, seed=a.seed, outdir=a.outdir)
    out = run_canciani(cfg)
    seg = out["seg"]
    e_ins = horizontal_error(out["nominal"]["lat"], out["nominal"]["lon"],
                             seg.lat, seg.lon, seg.lat)
    print("=" * 72)
    print(" Canciani 2022 - 38-state tightly-coupled MagNav EKF (SGL data)")
    print("=" * 72)
    print(f" Flight window [{a.tstart},{a.tend}] -> {seg.n} samples @ {1/seg.dt:.1f} Hz "
          f"({seg.n*seg.dt/60:.1f} min); {out['drift_info']}")
    print(f" INS drift: end={e_ins[-1]:.0f} m, max={e_ins.max():.0f} m; "
          f"batch T-L residual = {out['tl_resid']:.1f} nT")
    m = out["metrics"]
    print(f" INS (unaided)     DRMS = {m['INS']['drms']:8.1f} m")
    print(f" Loosely-coupled   DRMS = {m['loosely']['drms']:8.1f} m")
    print(f" Tightly-coupled   DRMS = {m['tightly']['drms']:8.1f} m")
    if m["loosely"]["drms"] > 0:
        print(f" Improvement tightly vs loosely: {100*(1-m['tightly']['drms']/m['loosely']['drms']):.0f}%")
    print(f" Plots -> {a.outdir}/ekf38_*.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
