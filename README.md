# magnavlab — magnetic navigation (MagNav)

A modular library for experiments with **magnetic navigation**: determining position
by matching the measured magnetic field to anomaly maps — a passive, jamming-resistant
alternative to GNSS/GPS.

The project's target goal: **reproducing the method from the paper**
A. J. Canciani, *"Magnetic Navigation on an F-16 Aircraft using Online Calibration"*,
IEEE TAES 2022 (38-state, tightly-coupled EKF with online Tolles-Lawson calibration).

## Installation

```bash
python3 -m venv .venv
./.venv/bin/pip install -e .            # package + core dependencies
./.venv/bin/pip install -e '.[dev]'     # + pytest, jupyter (tests and notebooks)
```

## Data (public)

Two Zenodo records are supported (SGL, Cessna Grand Caravan), plus the Ottawa-area
anomaly maps (a [MagNav.jl](https://github.com/MIT-AI-Accelerator/MagNav.jl) artifact):

| Record | Contents |
|--------|----------|
| [4271804](https://zenodo.org/records/4271804) | SGL 2020 challenge — Flt1002–1005 |
| [12723700](https://zenodo.org/records/12723700) | DAF-MIT AIA open flight — adds **Flt1006** (calibration flight), **Flt1007**, and 2021 flights (Flt20xx) |

Download and prepare with the helper script (idempotent; existing files are skipped):

```bash
python tools/get_data.py                 # default demo set: Flt1002, Flt1003, Eastern_395
python tools/get_data.py --list          # list flights available per record
python tools/get_data.py --flights Flt1006 Flt1007 --maps Eastern_395 Renfrew_395
```

…or from a notebook / Python:

```python
from magnavlab import data
data.ensure_demo_data()                  # Flt1002 + Flt1003 + Eastern_395
data.fetch_flight("Flt1006")             # proper calibration flight (record 12723700)
data.fetch_maps(["Eastern_395", "Renfrew_395"])
```

Calibration segments and known navigation windows (from MagNav.jl `df_cal`/`df_nav`) live in
`magnavlab.data` (`CAL_SEGMENTS`, `NAV_WINDOWS`) — including the Flt1006 calibration box.

> **F-16:** the F-16 flight data from Canciani's paper is **non-public** (AFIT tests, Edwards AFB).
> We demonstrate the method on public Cessna data (SGL). The notebooks take the flight/map paths
> as plain variables — the F-16 data just needs to be substituted.

## Notebooks (`notebooks/`)

The experiment **pipelines live in the notebooks** — each one shows the full sequence: loading,
calibration, INS simulation, filtering, metrics and plots — built from the `magnavlab`
primitives. Launch with:

```bash
./.venv/bin/jupyter notebook notebooks/
```

1. `01_data_and_map` — loading flight and map, correlation of the measurement with the map
2. `02_tolles_lawson_calibration` — comparison of calibrators (map-based vs map-less)
3. `03_ekf_pf_navigation` — EKF navigation vs particle filter (full pipeline)
4. `04_canciani_ekf38` — 38-state EKF, loosely vs tightly (full pipeline)
5. `05_dataset_12723700` — validating the DAF-MIT AIA dataset (Flt1006/1007, 2021 flights)

## Results — reproducing Canciani 2022

On public SGL data (line 1003.02, Eastern_395 map, cabin magnetometer `mag_4_uc`):

| Method | DRMS (ours, SGL) | Paper (F-16, 300 m AGL) |
|---|---|---|
| INS unaided | ~490 m (~1.1 km peak drift) | ">1 km" |
| Loosely-coupled (static calibration) | **~100 m** | 111 m |
| Tightly-coupled (online calibration) | **~56 m** | 59 m |
| Online improvement | **~44%** | ~47% |

Notebook `05` also runs the tightly-coupled filter on a flight unique to record 12723700
(Flt1007 line 1007.06 over `Renfrew_395`): loosely ~134 m → tightly ~66 m.

Plots render inline in notebook `04`. In line with the paper, the advantage of online
calibration stems from the non-stationarity of the aircraft's field, which static calibration
cannot remove quickly enough.

### Deliberate simplifications (no F-16 data)
- INS simulated at the error level (Pinson integration with IMU biases), not full mechanization.
- Core field from the IGRF model (`magnavlab.geomag`, via `ppigrf`) — the same role as the paper's WMM.
- Pinson block: dominant terms (without the minor transport-rate/Schuler terms).
- Cessna data, not F-16: the online-calibration advantage is present but modest here; the F-16's
  much larger, non-stationary field is what drives the paper's dramatic gains.

## Tests

```bash
./.venv/bin/pytest            # unit (synthetic, fast) + integration
```
Unit tests do not require data; the integration tests (`test_integration.py`) are skipped
when files are missing from `data/`.

## Architecture

Each layer has a clear interface and **interchangeable** implementations (see `magnavlab/interfaces.py`):

```
magnavlab/
  interfaces.py      protocols: MapLike, Calibrator, NavFilter, MeasurementModel
                     + structures: NavProblem, NavResult, MagV
  data.py            dataset registry + download/prepare helpers (Zenodo, maps)
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
```

The library provides **primitives**; the experiment *pipelines* live in the notebooks
(`notebooks/`), assembled from these primitives. Components are interchangeable, e.g. swapping
the navigation filter is one line:

```python
from magnavlab.filters import FILTERS        # {'ekf', 'pf', 'ekf38'}
result = FILTERS["pf"](sigma_meas=60.0).run(problem)   # any NavFilter takes a NavProblem
```
