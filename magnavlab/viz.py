"""Navigation-result plots.

Each function builds a matplotlib ``Figure`` and returns it, so it renders inline in a
notebook (with ``%matplotlib inline``). Nothing is written to disk - to save a figure,
call ``fig.savefig(...)`` on the returned object.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .geo import ned_offset
from .interfaces import MapLike

_DEG = np.degrees


def plot_map_tracks(mag_map: MapLike, lat_t, lon_t, tracks: dict,
                    title: str = "Trajectories on the anomaly map"):
    """Anomaly map with overlaid trajectories. ``tracks`` = {label: (lat, lon, style)}."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ext = mag_map.extent_deg()
    sy = max(1, mag_map.grid.shape[0] // 1200)
    sx = max(1, mag_map.grid.shape[1] // 1200)
    im = ax.imshow(mag_map.grid[::sy, ::sx], origin="lower", extent=ext,
                   aspect="auto", cmap="turbo", alpha=0.85)
    fig.colorbar(im, ax=ax, label="Map magnetic field [nT]")
    ax.plot(_DEG(lon_t), _DEG(lat_t), "k-", lw=2.2, label="Truth (GPS)")
    for lab, (lat, lon, style) in tracks.items():
        ax.plot(_DEG(lon), _DEG(lat), style, lw=1.4, label=lab)
    ax.set_xlim(_DEG(lon_t).min() - 0.02, _DEG(lon_t).max() + 0.02)
    ax.set_ylim(_DEG(lat_t).min() - 0.02, _DEG(lat_t).max() + 0.02)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title(title); ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_error_time(tt, lat_t, lon_t, series: dict, title: str = "Navigation error over time"):
    """Horizontal error [m] over time. ``series`` = {label: (result, color)}."""
    from .metrics import error_series
    tm = (tt - tt[0]) / 60.0
    fig, ax = plt.subplots(figsize=(10, 5))
    for lab, (res, color) in series.items():
        e = error_series(res, lat_t, lon_t)
        ax.plot(tm, e, color=color, lw=1.1,
                label=f"{lab}, DRMS={np.sqrt(np.mean(e**2)):.0f} m")
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Horizontal position error [m]")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def plot_ne_errors(tt, lat_t, lon_t, results: dict, title: str = "N/E errors", ylim: float = 400.0):
    """North/east errors [m] over time (like Fig. 12-15 of Canciani)."""
    tm = (tt - tt[0]) / 60.0
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for lab, (res, color) in results.items():
        dN, dE = ned_offset(res.lat, res.lon, lat_t, lon_t)
        ax[0].plot(tm, dN, color=color, lw=1.0, label=lab)
        ax[1].plot(tm, dE, color=color, lw=1.0, label=lab)
    ax[0].set_ylabel("N error [m]"); ax[1].set_ylabel("E error [m]")
    ax[1].set_xlabel("Time [min]"); ax[0].set_title(title)
    for a in ax:
        a.grid(alpha=0.3); a.legend(ncol=3); a.set_ylim(-ylim, ylim)
    fig.tight_layout()
    return fig


def plot_signal_vs_map(tt, meas, map_along, title: str = "Measurement vs map along the route"):
    """Compare the measurement signal with the map profile along the true route."""
    tm = (tt - tt[0]) / 60.0
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tm, meas, "k-", lw=1.0, label="Measurement (offset removed)")
    ax.plot(tm, map_along, "-", color="orange", lw=1.0, label="Map along the route")
    ax.set_xlabel("Time [min]"); ax.set_ylabel("Magnetic field [nT]")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig


def plot_tl_online(tt, tl_hist, idx=(0, 1, 2), labels=("perm X", "perm Y", "perm Z")):
    """Trace of estimated T-L coefficients (observability of online calibration)."""
    tm = (tt - tt[0]) / 60.0
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, lab in zip(idx, labels):
        ax.plot(tm, tl_hist[i], label=lab)
    ax.set_xlabel("Time [min]"); ax.set_ylabel("T-L coef. [nT]")
    ax.set_title("Online calibration: trace of T-L coefficients")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    return fig
