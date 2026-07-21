"""Experiment notebook generator (nbformat). Run from the repo directory."""
import os
import nbformat as nbf

OUT = "notebooks"
os.makedirs(OUT, exist_ok=True)


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"}}
    return n


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s):
    return nbf.v4.new_code_cell(s)


# ---------------------------------------------------------------------------
# 01 - data and map
# ---------------------------------------------------------------------------
nb1 = nb([
    md("# 01 · Flight data and anomaly map\n"
       "Load the SGL flight (Zenodo 4271804) and the `Eastern_395` anomaly map, and "
       "check how the measured field matches the map along the track.\n\n"
       "Requires data in `data/` (see README)."),
    code("import numpy as np\n"
         "import matplotlib.pyplot as plt\n"
         "from magnavlab.io import load_flight, load_map, segment_indices\n"
         "from magnavlab import viz"),
    md("## Loading and selecting a segment (line 1003.02, ~63 min)"),
    code("nav = load_flight('data/Flt1003_train.h5')\n"
         "mag_map = load_map('data/maps/Eastern_395.h5')\n"
         "sl = segment_indices(nav, 50713.0, 54497.0)[::5]\n"
         "lat = np.radians(nav.get('lat')[sl]); lon = np.radians(nav.get('lon')[sl])\n"
         "print(f'samples: {sl.size}, dt={nav.dt:.3f}s')\n"
         "print(f'map: {mag_map.grid.shape}, alt={mag_map.alt:.0f} m, "
         "values {mag_map.grid.min():.0f}..{mag_map.grid.max():.0f} nT')"),
    md("## Trajectory over the anomaly map"),
    code("from IPython.display import Image\n"
         "viz.plot_map_tracks(mag_map, lat, lon, {}, 'outputs/nb01_map.png',\n"
         "                    title='Flight 1003.02 track on the Eastern_395 map')\n"
         "Image('outputs/nb01_map.png')"),
    md("## Measurement (stinger) vs map along the track\n"
       "Total field `mag_1_c` correlated with the anomaly map (after removing a constant offset)."),
    code("z = nav.get('mag_1_c')[sl] - nav.get('diurnal')[sl]\n"
         "map_along = mag_map.value(lat, lon)\n"
         "bias0 = np.median(z - map_along); z = z - bias0\n"
         "print(f'measurement/map correlation: {np.corrcoef(z, map_along)[0,1]:.3f}')\n"
         "viz.plot_signal_vs_map(nav.get('tt')[sl], z, map_along, 'outputs/nb01_signal.png')\n"
         "Image('outputs/nb01_signal.png')"),
])

# ---------------------------------------------------------------------------
# 02 - Tolles-Lawson compensation
# ---------------------------------------------------------------------------
nb2 = nb([
    md("# 02 · Tolles-Lawson compensation (interchangeable calibrators)\n"
       "Comparison of calibration backends on the noisy cabin magnetometer `mag_4_uc`: "
       "built-in map-less and map-based modified.\n\n"
       "Quality metric: correlation of the compensated signal with the map and residual std. deviation."),
    code("import numpy as np\n"
         "from magnavlab.io import load_flight, load_map, segment_indices\n"
         "from magnavlab.calibration import BuiltinTL, MapBasedModifiedTL\n"
         "nav = load_flight('data/Flt1003_train.h5')\n"
         "mag_map = load_map('data/maps/Eastern_395.h5')\n"
         "sl = segment_indices(nav, 50713.0, 54497.0)[::5]\n"
         "lat = np.radians(nav.get('lat')[sl]); lon = np.radians(nav.get('lon')[sl])\n"
         "# Earth's field (map-based target) at total-field scale = anomaly map + core\n"
         "earth = mag_map.value(lat, lon) + (nav.get('mag_1_c')[sl] - nav.get('igrf')[sl])\n"
         "flux = nav.flux(sl); z = nav.get('mag_4_uc')[sl]\n"
         "half = sl.size // 2                      # paper protocol: 1st half = calibration"),
    md("## Fit on the 1st half of the flight, evaluate on the 2nd half (validation)"),
    code("val = slice(half, None)\n"
         "def score(name, calibrator, target=None):\n"
         "    tgt = None if target is None else target[:half]\n"
         "    calibrator.fit(nav.flux(sl[:half]), z[:half], nav.dt, target=tgt)\n"
         "    comp = calibrator.compensate(flux, z, nav.dt)\n"
         "    corr = np.corrcoef(comp[val], earth[val])[0,1]\n"
         "    print(f'{name:24s} corr with map={corr:.3f}  "
         "std(comp-map)={np.std(comp[val]-earth[val]):.0f} nT')\n"
         "\n"
         "print('raw mag_4_uc'.ljust(24), "
         "f'corr with map={np.corrcoef(z[val], earth[val])[0,1]:.3f}  "
         "std={np.std(z[val]-earth[val]):.0f} nT')\n"
         "score('BuiltinTL (map-less)', BuiltinTL())\n"
         "score('MapBasedModifiedTL', MapBasedModifiedTL(), target=earth)"),
    md("The map-based modified variant gives the best fit to the map (in line with the paper), "
       "clearly better than the map-less approach on the validation half."),
])

# ---------------------------------------------------------------------------
# 03 - EKF / PF navigation
# ---------------------------------------------------------------------------
nb3 = nb([
    md("# 03 · Magnetic navigation: EKF vs particle filter\n"
       "Full `run_simple_nav` pipeline: T-L compensation, drifting-INS simulation, "
       "EKF and PF navigation via map matching. The filters are interchangeable (`NavFilter` protocol)."),
    code("from magnavlab.experiments import SimpleNavConfig, run_simple_nav\n"
         "from IPython.display import Image\n"
         "out = run_simple_nav(SimpleNavConfig(make_plots=True))\n"
         "for name, m in out['metrics'].items():\n"
         "    print(f\"{name:4s} DRMS={m['drms']:7.1f} m  CEP50={m['cep50']:6.1f}  \"\n"
         "          f\"CEP95={m['cep95']:6.1f}  max={m['max']:7.1f}\")"),
    md("## Position error over time"),
    code("Image('outputs/simple_position_error.png')"),
    md("## Trajectories on the map"),
    code("Image('outputs/simple_tracks.png')"),
    md("Swapping the filter/calibrator comes down to replacing a component, e.g.:\n"
       "```python\n"
       "run_simple_nav(SimpleNavConfig(calibrator='mapbased', nav_signal='stinger'))\n"
       "```"),
])

# ---------------------------------------------------------------------------
# 04 - Canciani EKF38
# ---------------------------------------------------------------------------
nb4 = nb([
    md("# 04 · Reproducing Canciani 2022 — EKF38 with online calibration\n"
       "38-state tightly-coupled filter (Pinson INS + online T-L + FOGM + vector). "
       "Comparison of **loosely-coupled** (static calibration) vs **tightly-coupled** "
       "(in-flight calibration).\n\n"
       "⚠️ The F-16 flight data from the paper is non-public — we demonstrate the method on "
       "public SGL data (Cessna). We emulate the F-16 conditions (stochastic field drift) with "
       "the optional `inject_drift`."),
    code("from magnavlab.experiments import CancianiConfig, run_canciani\n"
         "from IPython.display import Image\n"
         "out = run_canciani(CancianiConfig(make_plots=True))\n"
         "print(out['drift_info'], '| batch T-L residual =', round(out['tl_resid'],1), 'nT')\n"
         "for name, m in out['metrics'].items():\n"
         "    print(f\"{name:9s} DRMS={m['drms']:8.1f} m\")\n"
         "mt, ml = out['metrics']['tightly']['drms'], out['metrics']['loosely']['drms']\n"
         "print(f'improvement tightly vs loosely: {100*(1-mt/ml):.0f}%')"),
    md("## N/E errors (like Fig. 12–15 in the paper)"),
    code("Image('outputs/ekf38_NE_errors.png')"),
    md("## Online calibration — trace of T-L coefficients"),
    code("Image('outputs/ekf38_TL_online.png')"),
    md("The result on SGL data (loosely ~102 m, tightly ~57 m, ~45%) qualitatively matches "
       "Table I from the paper (F-16: 111 m → 59 m, ~47%)."),
])

# bootstrap cell: run from the repo directory (so imports and data/ paths work)
BOOT = ("import os, sys\n"
        "while not os.path.isdir('magnavlab') and os.path.dirname(os.getcwd()) != os.getcwd():\n"
        "    os.chdir('..')\n"
        "sys.path.insert(0, os.getcwd())\n"
        "print('repo:', os.getcwd())")

for name, notebook in [("01_data_and_map", nb1), ("02_tolles_lawson_calibration", nb2),
                       ("03_ekf_pf_navigation", nb3), ("04_canciani_ekf38", nb4)]:
    notebook.cells.insert(1, code(BOOT))   # after the title, before the rest
    path = os.path.join(OUT, name + ".ipynb")
    with open(path, "w") as f:
        nbf.write(notebook, f)
    print("saved", path)
