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


def plot_tl_parameters(x, tl, s_cb=None, x_label: str = "Distance [km]",
                       title: str = "Online Tolles-Lawson parameters"):
    """Evolution of the T-L states over the run, grouped as in Canciani 2022 Fig. (c).

    ``tl`` is the (19, N) history from a tightly-coupled filter (``result.extras['tl']``):
    rows 0:3 permanent, 3:9 induced, 9:18 eddy-current, 18 the constant offset. ``s_cb`` is the
    optional FOGM measurement-bias history (``result.extras['S']``) drawn in its own bottom panel.
    ``x`` is the shared horizontal axis (e.g. cumulative distance [km] or time [min]).
    """
    tl = np.asarray(tl)
    groups = [(r"$\beta_{TL,perm}$", tl[0:3]),
              (r"$\beta_{TL,ind}$",  tl[3:9]),
              (r"$\beta_{TL,eddy}$", tl[9:18])]
    nrows = len(groups) + (s_cb is not None)
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.1 * nrows), sharex=True)
    for ax, (lab, block) in zip(axes, groups):
        for row in block:
            ax.plot(x, row, lw=0.9)
        ax.set_ylabel(lab); ax.grid(alpha=0.3, ls="--")
        ax.axhline(0.0, color="0.6", lw=0.6)
    if s_cb is not None:
        axes[-1].plot(x, s_cb, color="tab:blue", lw=1.0)
        axes[-1].set_ylabel(r"$S_{CB}$"); axes[-1].grid(alpha=0.3, ls="--")
        axes[-1].axhline(0.0, color="0.6", lw=0.6)
    axes[-1].set_xlabel(x_label); axes[0].set_title(title)
    fig.tight_layout()
    return fig


def plot_tracks_vs_truth(mag_map: MapLike, lat_t, lon_t, estimates: dict,
                         title: str = "MagNav position vs truth (on the anomaly map)"):
    """MagNav-estimated position(s) against the true (GPS) track, overlaid on the anomaly map.

    ``estimates`` = {label: (result, color)}. The map is cropped and the axes are tight to the
    plotted tracks so the estimate-vs-truth difference is visible against the local anomaly
    field; the free-INS solution is intentionally omitted (its km-scale drift would flatten it).
    """
    lons, lats = [_DEG(lon_t)], [_DEG(lat_t)]
    for _lab, (res, _c) in estimates.items():
        lons.append(_DEG(res.lon)); lats.append(_DEG(res.lat))
    alllon, alllat = np.concatenate(lons), np.concatenate(lats)
    mx, my = 0.05 * (np.ptp(alllon) + 1e-9), 0.05 * (np.ptp(alllat) + 1e-9)
    x0, x1 = alllon.min() - mx, alllon.max() + mx
    y0, y1 = alllat.min() - my, alllat.max() + my

    # crop the map to the track's bounding box (keeps full resolution, small image)
    lon_deg, lat_deg = _DEG(mag_map.lon), _DEG(mag_map.lat)
    j0 = max(int(np.searchsorted(lon_deg, x0)) - 1, 0)
    j1 = min(int(np.searchsorted(lon_deg, x1)) + 1, lon_deg.size)
    i0 = max(int(np.searchsorted(lat_deg, y0)) - 1, 0)
    i1 = min(int(np.searchsorted(lat_deg, y1)) + 1, lat_deg.size)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(mag_map.grid[i0:i1, j0:j1], origin="lower", cmap="turbo", alpha=0.85,
                   aspect="auto", extent=(lon_deg[j0], lon_deg[j1 - 1], lat_deg[i0], lat_deg[i1 - 1]))
    fig.colorbar(im, ax=ax, label="Map magnetic field [nT]")
    ax.plot(_DEG(lon_t), _DEG(lat_t), "k-", lw=2.6, label="Truth (GPS)")
    for lab, (res, color) in estimates.items():
        ax.plot(_DEG(res.lon), _DEG(res.lat), color=color, lw=1.3, label=lab)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("Longitude [deg]"); ax.set_ylabel("Latitude [deg]")
    ax.set_title(title); ax.legend(loc="best")
    fig.tight_layout()
    return fig
