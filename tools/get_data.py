#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download and prepare MagNav data (Zenodo flights + Ottawa anomaly maps).

Runs from anywhere; files land in <repo>/data by default. Idempotent - existing
files are skipped. Thin wrapper around :mod:`magnavlab.data`.

Examples:
    python tools/get_data.py                       # default: demo set (Flt1002, Flt1003, Eastern_395)
    python tools/get_data.py --list                # list flights available per Zenodo record
    python tools/get_data.py --flights Flt1006 Flt1007 --maps Eastern_395 Renfrew_395
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                         # importable without installation
from magnavlab import data  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Download & prepare MagNav data (Zenodo + maps).")
    p.add_argument("--list", action="store_true",
                   help="List flights available per Zenodo record and exit.")
    p.add_argument("--demo", action="store_true",
                   help="Fetch the default demo set (Flt1002, Flt1003, Eastern_395).")
    p.add_argument("--flights", nargs="*", default=[], metavar="FltXXXX",
                   help="Flights to download, e.g. Flt1006 Flt1007 (Flt10xx/Flt20xx).")
    p.add_argument("--maps", nargs="*", default=[], metavar="MAP",
                   help=f"Maps to extract. Available: {', '.join(data.AVAILABLE_MAPS)}.")
    p.add_argument("--dest", default=os.path.join(_ROOT, "data"),
                   help="Destination data directory (default: <repo>/data).")
    a = p.parse_args(argv)

    if a.list:
        data.list_records()
        return

    if not (a.demo or a.flights or a.maps):
        a.demo = True                              # default action when called with no arguments

    maps_dir = os.path.join(a.dest, "maps")
    if a.demo:
        print("Fetching demo dataset (Flt1002, Flt1003, Eastern_395)...")
        data.fetch_flight("Flt1002", a.dest)
        data.fetch_flight("Flt1003", a.dest)
        data.fetch_maps(["Eastern_395"], maps_dir)
    for flight in a.flights:
        data.fetch_flight(flight, a.dest)
    if a.maps:
        data.fetch_maps(a.maps, maps_dir)

    print(f"Done. Data directory: {a.dest}")


if __name__ == "__main__":
    main()
