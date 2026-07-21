"""Experiment orchestration layer.

Runners tie interchangeable components (calibrator, filter, map) into a repeatable pipeline.
A new experiment = new configuration or component swap, without changes to the engines.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import viz
from .calibration import CALIBRATORS, MapBasedModifiedTL
from .filters import Canciani38EKF, EKFNav, ParticleFilterNav
from .metrics import summary
from .interfaces import NavProblem
from .ins import build_kinematics, simulate_ins_pinson, simulate_ins_velocity
from .io import load_flight, load_map, segment_indices

# default Flt1002 calibration segments (from df_cal.csv MagNav.jl)
DEFAULT_CAL_SEGMENTS = [
    (46390.9, 46964.5), (47027.1, 47546.3),
    (66571.7, 67131.8), (67276.8, 67839.2),
]


def _robust_std(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


# ===========================================================================
#  Common loading and extraction of a flight segment
# ===========================================================================
@dataclass
class Segment:
    """Extracted, decimated flight segment + true trajectory."""
    flight: object
    sl: np.ndarray
    dt: float
    lat: np.ndarray
    lon: np.ndarray
    alt: np.ndarray
    tt: np.ndarray

    @property
    def n(self) -> int:
        return len(self.lat)


def load_segment(nav_file: str, tstart: float, tend: float, decimate: int) -> Segment:
    """Loads a flight, selects a contiguous block within the time window and decimates."""
    nav = load_flight(nav_file)
    idx = segment_indices(nav, tstart, tend)
    if idx.size < 100:
        raise RuntimeError(f"Window [{tstart},{tend}] yielded only {idx.size} samples.")
    sl = idx[::decimate]
    return Segment(flight=nav, sl=sl, dt=nav.dt * decimate,
                   lat=np.radians(nav.get("lat")[sl]),
                   lon=np.radians(nav.get("lon")[sl]),
                   alt=nav.get("alt")[sl],
                   tt=nav.get("tt")[sl])


# ===========================================================================
#  Experiment 1: simple MagNav (T-L compensation + EKF/PF)
# ===========================================================================
@dataclass
class SimpleNavConfig:
    nav_file: str = "data/Flt1003_train.h5"
    tstart: float = 50713.0
    tend: float = 54497.0
    map_file: str = "data/maps/Eastern_395.h5"
    cal_file: str | None = "data/Flt1002_train.h5"
    cal_segments: list = field(default_factory=lambda: list(DEFAULT_CAL_SEGMENTS))
    decimate: int = 5
    calibrator: str = "builtin"          # 'builtin' | 'mapbased'
    comp_mag: str = "mag_4_uc"
    nav_signal: str = "compensated"      # 'compensated' | 'stinger'
    vel_bias: tuple = (0.25, -0.20)
    rw_sigma: float = 1e-3
    n_particles: int = 4000
    seed: int = 0
    outdir: str = "outputs"
    make_plots: bool = True


def run_simple_nav(cfg: SimpleNavConfig) -> dict:
    """T-L compensation (interchangeable calibrator) + EKF and PF navigation on SGL data."""
    seg = load_segment(cfg.nav_file, cfg.tstart, cfg.tend, cfg.decimate)
    nav, sl, dt = seg.flight, seg.sl, seg.dt
    mag_map = load_map(cfg.map_file)

    # --- core field and navigation signal ---
    diurnal = nav.get("diurnal")[sl] if nav.has("diurnal") else 0.0
    if cfg.nav_signal == "stinger" and nav.has("mag_1_c"):
        nav_field = nav.get("mag_1_c")[sl]
        cal_info = "stinger (mag_1_c)"
    else:
        cal = CALIBRATORS[cfg.calibrator]()
        target_needed = isinstance(cal, MapBasedModifiedTL)
        # fitting on the calibration segment (separate flight) or on the nav flight
        if cfg.cal_file:
            calf = load_flight(cfg.cal_file)
            ci = np.concatenate([segment_indices(calf, a, b) for a, b in cfg.cal_segments])
            flux_c = calf.flux(ci); scal_c = calf.get(cfg.comp_mag)[ci]
            tgt = None
            if target_needed:
                # the map-based target requires the calibration-segment position (GPS) and the map
                latc = np.radians(calf.get("lat")[ci]); lonc = np.radians(calf.get("lon")[ci])
                corec = (calf.get("mag_1_c")[ci] - calf.get("igrf")[ci])
                tgt = mag_map.value(latc, lonc) + corec
            cal.fit(flux_c, scal_c, calf.dt, target=tgt)
        nav_field = cal.compensate(nav.flux(sl), nav.get(cfg.comp_mag)[sl], dt)
        cal_info = f"{cfg.calibrator} T-L on {cfg.comp_mag}"

    meas = nav_field - diurnal
    map_along = mag_map.value(seg.lat, seg.lon)
    valid = np.isfinite(map_along)
    bias0 = np.median(meas[valid] - map_along[valid])
    meas = meas - bias0
    sigma_meas = max(_robust_std((meas - map_along)[valid]), 5.0)

    # --- drifting INS (velocity level) ---
    ins_lat, ins_lon, vN, vE = simulate_ins_velocity(
        seg.lat, seg.lon, seg.alt, dt, vel_bias=tuple(cfg.vel_bias),
        rw_sigma=cfg.rw_sigma, seed=cfg.seed)

    problem = NavProblem(dt=dt, map=mag_map, lat=ins_lat, lon=ins_lon, alt=seg.alt,
                         vN=vN, vE=vE, vD=np.zeros(seg.n), meas=meas)

    ekf = EKFNav(sigma_meas=sigma_meas).run(problem)
    pf = ParticleFilterNav(sigma_meas=sigma_meas, n_particles=cfg.n_particles).run(problem)

    from .interfaces import NavResult
    ins_res = NavResult(lat=ins_lat, lon=ins_lon)
    results = {"INS": ins_res, "EKF": ekf, "PF": pf}
    metrics = {k: summary(v, seg.lat, seg.lon) for k, v in results.items()}

    out = dict(segment=seg, map=mag_map, meas=meas, map_along=map_along,
               sigma_meas=sigma_meas, cal_info=cal_info, results=results, metrics=metrics)
    if cfg.make_plots:
        viz.plot_map_tracks(mag_map, seg.lat, seg.lon,
                            {"INS (drift)": (ins_lat, ins_lon, "w--"),
                             "EKF": (ekf.lat, ekf.lon, "r-"),
                             "PF": (pf.lat, pf.lon, "g-")},
                            f"{cfg.outdir}/simple_tracks.png")
        viz.plot_error_time(seg.tt, seg.lat, seg.lon,
                            {"INS": (ins_res, "tab:blue"), "EKF": (ekf, "tab:red"),
                             "PF": (pf, "tab:green")},
                            f"{cfg.outdir}/simple_position_error.png")
        viz.plot_signal_vs_map(seg.tt, meas, map_along, f"{cfg.outdir}/simple_signal_vs_map.png")
    return out


# ===========================================================================
#  Experiment 2: Canciani 2022 - EKF38 loosely vs tightly
# ===========================================================================
@dataclass
class CancianiConfig:
    nav_file: str = "data/Flt1003_train.h5"
    tstart: float = 50713.0
    tend: float = 54497.0
    map_file: str = "data/maps/Eastern_395.h5"
    scalar_mag: str = "mag_4_uc"
    decimate: int = 5
    inject_drift: float = 0.6            # body-field drift σ [nT/√s]; 0 = disable
    accel_bias: tuple = (3e-4, -2e-4, 1e-4)
    gyro_bias: tuple = (3e-7, -2e-7, 1e-7)
    R: float = 60.0
    Qf: float = 200.0
    seed: int = 0
    outdir: str = "outputs"
    make_plots: bool = True


def prepare_canciani(cfg: CancianiConfig) -> dict:
    """Builds all EKF38 inputs (kinematics, INS, batch T-L, measurement with body-field drift)."""
    seg = load_segment(cfg.nav_file, cfg.tstart, cfg.tend, cfg.decimate)
    nav, sl, dt, n = seg.flight, seg.sl, seg.dt, seg.n
    mag_map = load_map(cfg.map_file)

    roll = np.radians(nav.get("roll")[sl]) if nav.has("roll") else None
    pitch = np.radians(nav.get("pitch")[sl]) if nav.has("pitch") else None
    kin = build_kinematics(seg.lat, seg.lon, seg.alt, dt, roll, pitch)

    flux = nav.flux(sl)
    z = nav.get(cfg.scalar_mag)[sl].astype(float)
    core = nav.get("mag_1_c")[sl] - nav.get("igrf")[sl]
    if nav.has("diurnal"):
        core = core - nav.get("diurnal")[sl]

    cX, cY, cZ = flux.x / z, flux.y / z, flux.z / z
    cos_dot = (np.gradient(cX, dt), np.gradient(cY, dt), np.gradient(cZ, dt))

    drift_info = "no injected drift"
    if cfg.inject_drift > 0:
        rng = np.random.default_rng(cfg.seed + 7)
        sd = cfg.inject_drift
        ex = np.cumsum(rng.normal(0, sd, n)) * np.sqrt(dt)
        ey = np.cumsum(rng.normal(0, sd, n)) * np.sqrt(dt)
        ez = np.cumsum(rng.normal(0, sd, n)) * np.sqrt(dt)
        z = z + ex * cX + ey * cY + ez * cZ
        drift_info = f"body-field drift σ={sd} nT/√s (~{np.max(np.abs(np.c_[ex,ey,ez])):.0f} nT)"

    nominal, e_true = simulate_ins_pinson(seg.lat, seg.lon, seg.alt, kin, dt,
                                          accel_bias=tuple(cfg.accel_bias),
                                          gyro_bias=tuple(cfg.gyro_bias), seed=cfg.seed)

    # batch modified map-based T-L on the first 50% of the flight
    half = n // 2
    be = mag_map.value(seg.lat[:half], seg.lon[:half]) + core[:half]
    flux_half = nav.flux(sl[:half])
    tl = MapBasedModifiedTL().fit(flux_half, z[:half], dt, target=be)

    problem = NavProblem(
        dt=dt, map=mag_map, lat=nominal["lat"], lon=nominal["lon"], alt=nominal["alt"],
        vN=nominal["vN"], vE=nominal["vE"], vD=nominal["vD"], meas=z,
        core=core, flux=flux, cos_dot=cos_dot,
        fn=kin["fn"], fe=kin["fe"], fd=kin["fd"], Cnb=kin["Cnb"],
        tl0=tl.coef, P_tl0=tl.P_cov)
    return dict(seg=seg, map=mag_map, problem=problem, nominal=nominal, e_true=e_true,
                drift_info=drift_info, tl_resid=tl.resid_std)


def run_canciani(cfg: CancianiConfig) -> dict:
    """Runs EKF38 in loosely and tightly modes; returns metrics and (optionally) plots."""
    prep = prepare_canciani(cfg)
    seg, problem = prep["seg"], prep["problem"]
    from .interfaces import NavResult

    loose = Canciani38EKF(mode="loosely", R=cfg.R**2, Qf=cfg.Qf**2).run(problem)
    tight = Canciani38EKF(mode="tightly", R=cfg.R**2, Qf=cfg.Qf**2).run(problem)
    ins_res = NavResult(lat=prep["nominal"]["lat"], lon=prep["nominal"]["lon"])

    results = {"INS": ins_res, "loosely": loose, "tightly": tight}
    metrics = {k: summary(v, seg.lat, seg.lon) for k, v in results.items()}
    prep.update(results=results, metrics=metrics)

    if cfg.make_plots:
        viz.plot_ne_errors(seg.tt, seg.lat, seg.lon,
                           {"INS": (ins_res, "tab:blue"), "loosely": (loose, "tab:orange"),
                            "tightly": (tight, "tab:green")},
                           f"{cfg.outdir}/ekf38_NE_errors.png",
                           title=f"N/E errors: loosely (DRMS={metrics['loosely']['drms']:.0f} m) "
                                 f"vs tightly (DRMS={metrics['tightly']['drms']:.0f} m)")
        viz.plot_map_tracks(prep["map"], seg.lat, seg.lon,
                            {"INS": (ins_res.lat, ins_res.lon, "b--"),
                             "loosely": (loose.lat, loose.lon, "-"),
                             "tightly": (tight.lat, tight.lon, "-")},
                            f"{cfg.outdir}/ekf38_tracks.png")
        viz.plot_tl_online(seg.tt, tight.extras["tl"], f"{cfg.outdir}/ekf38_TL_online.png")
    return prep
