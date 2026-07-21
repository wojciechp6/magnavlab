# magnavlab — magnetic navigation (MagNav)

A modular library for experiments with **magnetic navigation**: determining position
by matching the measured magnetic field to anomaly maps — a passive, jamming-resistant
alternative to GNSS/GPS.

The project's target goal: **reproducing the method from the paper**
A. J. Canciani, *"Magnetic Navigation on an F-16 Aircraft using Online Calibration"*,
IEEE TAES 2022 (38-state, tightly-coupled EKF with online Tolles-Lawson calibration).

## Architecture

Each layer has a clear interface and **interchangeable** implementations (see `magnavlab/interfaces.py`):

```
magnavlab/
  interfaces.py      protocols: MapLike, Calibrator, NavFilter, MeasurementModel
                     + structures: NavProblem, NavResult, MagV
  geo.py             WGS-84 geodesy, NED frame (pure functions)
  io/
    flight.py        HDF5 flight loading (FlightData, segment_indices)
    maps.py          map loading/interpolation (MagMap)
  calibration/
    tolles_lawson.py BuiltinTL · MapBasedModifiedTL  (CALIBRATORS registry)
  models/
    pinson.py        Pinson INS error model + augmented dynamics (38 states)
    measurement.py   TLAugmentedMeasurement (h, H) — interchangeable measurement model
  ins.py             drifting-INS simulation (velocity-level and Pinson)
  filters/
    ekf.py           EKFNav (5 states)
    pf.py            ParticleFilterNav
    ekf38.py         Canciani38EKF (loosely/tightly)          (FILTERS registry)
  metrics.py         DRMS / CEP
  viz.py             plots
  experiments.py     ready-made pipelines: run_simple_nav, run_canciani
```

Swapping a component = one line, e.g. a different calibrator or filter:

```python
from magnavlab.experiments import run_simple_nav, SimpleNavConfig
run_simple_nav(SimpleNavConfig(calibrator="mapbased", nav_signal="stinger"))
```

## Installation

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .            # package + core dependencies
./.venv/bin/pip install -e '.[dev]'     # + pytest, jupyter (tests and notebooks)
```

## Data (public, ~1 GB)

| Resource | Source |
|-------|--------|
| Flights Flt1002–1005 (HDF5) | [Zenodo 4271804](https://zenodo.org/records/4271804) — Cessna SGL 2020 |
| Anomaly maps (Eastern_395 …) | `ottawa_area_maps` artifact from [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) |

```bash
mkdir -p data data/maps
curl -L "https://zenodo.org/records/4271804/files/Flt1002-train.h5?download=1" -o data/Flt1002_train.h5
curl -L "https://zenodo.org/records/4271804/files/Flt1003-train.h5?download=1" -o data/Flt1003_train.h5
# maps: unpack ottawa_area_maps_v3.tar.gz and keep data/maps/Eastern_395.h5
```

> **F-16:** the F-16 flight data from Canciani's paper is **non-public** (AFIT tests, Edwards AFB).
> We demonstrate the method on public Cessna data (SGL). The code has clean inputs
> (`--nav-file/--map-file/--scalar-mag`) — the F-16 data just needs to be substituted.

## Running

Everything runs through the notebooks in `notebooks/` (below), or programmatically via
the `magnavlab` API:

```python
# reproducing Canciani 2022 (EKF38: loosely vs tightly)
from magnavlab.experiments import run_canciani, CancianiConfig
res = run_canciani(CancianiConfig(make_plots=True))
print(res["metrics"]["tightly"]["drms"])

# simple demo (T-L compensation + EKF + PF)
from magnavlab.experiments import run_simple_nav, SimpleNavConfig
run_simple_nav(SimpleNavConfig())
```

Launch the notebooks with:

```bash
./.venv/bin/jupyter notebook notebooks/
```

## Notebooks (`notebooks/`)

1. `01_data_and_map` — loading flight and map, correlation of the measurement with the map
2. `02_tolles_lawson_calibration` — comparison of calibrators (map-based vs map-less)
3. `03_ekf_pf_navigation` — EKF navigation vs particle filter
4. `04_canciani_ekf38` — 38-state EKF, loosely vs tightly

Regenerate: `python tools/make_notebooks.py`.

## Tests

```bash
./.venv/bin/pytest            # unit (synthetic, fast) + integration
```
Unit tests do not require data; the integration tests (`test_integration.py`) are skipped
when files are missing from `data/`.

## Results — reproducing Canciani 2022

On public SGL data (line 1003.02, Eastern_395 map, cabin magnetometer `mag_4_uc`):

| Method | DRMS (ours, SGL) | Paper (F-16, 300 m AGL) |
|---|---|---|
| INS unaided | ~2470 m (5 km drift) | ">1 km" |
| Loosely-coupled (static calibration) | **~104 m** | 111 m |
| Tightly-coupled (online calibration) | **~57 m** | 59 m |
| Online improvement | **~45%** | ~47% |

Plots in `outputs/`. In line with the paper, the advantage of online calibration stems from
the non-stationarity of the aircraft's field, which static calibration cannot remove quickly enough.

### Deliberate simplifications (no F-16 data)
- INS simulated at the error level (Pinson integration with IMU biases), not full mechanization.
- Core field from data (`mag_1_c − igrf`) instead of a WMM model (core gradient negligible here).
- Pinson block: dominant terms (without the minor transport-rate/Schuler terms).
- F-16 environment emulation via an optional injected field drift in the body frame.
