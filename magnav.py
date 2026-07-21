#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simple MagNav demo CLI (T-L compensation + EKF/PF).

Thin wrapper around the ``magnavlab`` package - all logic lives in the library
(see magnavlab/experiments.py). Run:

    python magnav.py demo                 # full demo (EKF + PF + plots)
    python magnav.py demo --calibrator mapbased --nav-signal stinger
    python magnav.py inspect data/Flt1003_train.h5
"""
from __future__ import annotations

import argparse

from magnavlab.experiments import SimpleNavConfig, run_simple_nav
from magnavlab.io import inspect_h5


def _demo(args) -> None:
    cfg = SimpleNavConfig(
        nav_file=args.nav_file, tstart=args.tstart, tend=args.tend,
        map_file=args.map_file, cal_file=args.cal_file, decimate=args.decimate,
        calibrator=args.calibrator, comp_mag=args.comp_mag, nav_signal=args.nav_signal,
        n_particles=args.n_particles, seed=args.seed, outdir=args.outdir)
    out = run_simple_nav(cfg)
    print("=" * 66)
    print(f" MagNav demo | compensation: {out['cal_info']} | sigma_meas={out['sigma_meas']:.1f} nT")
    print("=" * 66)
    for name, m in out["metrics"].items():
        print(f"  {name:4s}  DRMS={m['drms']:8.1f} m  CEP50={m['cep50']:7.1f} m  "
              f"CEP95={m['cep95']:7.1f} m  max={m['max']:7.1f} m")
    print(f"  Plots -> {args.outdir}/simple_*.png")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Simple magnetic navigation demo (EKF + PF).")
    sub = p.add_subparsers(dest="cmd")
    pi = sub.add_parser("inspect", help="Print the fields of an HDF5 file.")
    pi.add_argument("path")
    d = sub.add_parser("demo", help="Run the navigation demo.")
    d.add_argument("--nav-file", default="data/Flt1003_train.h5")
    d.add_argument("--tstart", type=float, default=50713.0)
    d.add_argument("--tend", type=float, default=54497.0)
    d.add_argument("--map-file", default="data/maps/Eastern_395.h5")
    d.add_argument("--cal-file", default="data/Flt1002_train.h5")
    d.add_argument("--decimate", type=int, default=5)
    d.add_argument("--calibrator", default="builtin", choices=["builtin", "mapbased"])
    d.add_argument("--comp-mag", default="mag_4_uc")
    d.add_argument("--nav-signal", default="compensated", choices=["compensated", "stinger"])
    d.add_argument("--n-particles", type=int, default=4000)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--outdir", default="outputs")

    args = p.parse_args(argv)
    if args.cmd == "inspect":
        inspect_h5(args.path)
    else:
        if args.cmd is None:
            args = p.parse_args(["demo"])
        _demo(args)


if __name__ == "__main__":
    main()
