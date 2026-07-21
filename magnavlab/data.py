"""Dataset registry and download/preparation helpers.

Fetches the public SGL flight data (Zenodo) and the Ottawa-area anomaly maps, so
the notebooks can run without manual setup. Uses only the standard library
(urllib, tarfile) - no extra dependencies.

Supported Zenodo records:
  - 4271804  : "Signal Enhancement for MagNav Challenge Problem" (SGL 2020, Flt1002-1005)
  - 12723700 : "DAF-MIT AIA Open Flight Data" (SGL 2020+2021, adds Flt1006/1007, Flt20xx)

Programmatic use (e.g. from a notebook):
    >>> from magnavlab import data
    >>> data.ensure_demo_data()                 # Flt1002 + Flt1003 + Eastern_395
    >>> data.fetch_flight("Flt1006")            # proper calibration flight (12723700)
    >>> data.fetch_maps(["Eastern_395", "Renfrew_395"])
"""
from __future__ import annotations

import json
import os
import tarfile
import urllib.request

# --- Zenodo records (resolved to file URLs at runtime via the API) ---
ZENODO_RECORDS: dict[str, dict] = {
    "4271804": {"title": "Signal Enhancement for MagNav Challenge Problem",
                "aircraft": "Cessna Grand Caravan (SGL 2020)",
                "flights": ["Flt1002", "Flt1003", "Flt1004", "Flt1005"]},
    "12723700": {"title": "DAF-MIT AIA Open Flight Data for Magnetic Navigation Research",
                 "aircraft": "Cessna Grand Caravan (SGL 2020 + 2021)",
                 "flights": ["Flt1002", "Flt1003", "Flt1004", "Flt1005", "Flt1006", "Flt1007",
                             "Flt2001", "Flt2002", "Flt2004", "Flt2005", "Flt2006", "Flt2007",
                             "Flt2008", "Flt2015", "Flt2016", "Flt2017"]},
}
# search order for a flight name (12723700 is a superset; 4271804 is the canonical challenge set)
_DEFAULT_RECORD_ORDER = ("12723700", "4271804")

# --- Anomaly maps: single tarball artifact (from MagNav.jl Artifacts.toml, hosted on Dropbox) ---
MAP_ARTIFACT_URL = ("https://www.dropbox.com/scl/fi/ttd2ru1cgl5l7vrngcl2j/"
                    "ottawa_area_maps_v3.tar.gz?rlkey=5f365x4afackkhqhbbhkc87be&dl=1")
MAP_ARTIFACT_SIZE = 710_503_436   # bytes (for progress; not a strict check)
AVAILABLE_MAPS = ["Eastern_395", "Eastern_drape", "Renfrew_395", "Renfrew_555",
                  "Renfrew_drape", "HighAlt_5181", "Perth_800"]

# --- Calibration segments (time windows [s], from MagNav.jl df_cal.csv) ---
# Dedicated pitch/roll/yaw calibration maneuvers used to fit Tolles-Lawson coefficients.
CAL_SEGMENTS: dict[str, list[tuple[float, float]]] = {
    "Flt1002": [(46390.9, 46964.5), (47027.1, 47546.3),
                (66571.7, 67131.8), (67276.8, 67839.2)],
    # Flt1006 is the proper calibration flight (box maneuvers); available in record 12723700.
    "Flt1006": [(47222.0, 48213.0), (49165.3, 49798.5), (49940.1, 50318.5),
                (50340.7, 50804.2), (50829.7, 51301.7), (51377.5, 52013.8),
                (52408.4, 52843.1), (52861.8, 53286.0), (53855.0, 54510.0)],
}

# --- Known navigation windows (from MagNav.jl df_nav.csv): (flight, tstart, tend, map) ---
NAV_WINDOWS: dict[str, dict] = {
    "Flt1003_1003.02": {"flight": "Flt1003", "tstart": 50713.0, "tend": 54497.0, "map": "Eastern_395"},
    "Flt1006_1006.08": {"flight": "Flt1006", "tstart": 55770.0, "tend": 56609.0, "map": "Eastern_395"},
    "Flt1007_1007.02": {"flight": "Flt1007", "tstart": 48024.0, "tend": 51880.0, "map": "Eastern_395"},
    "Flt1007_1007.06": {"flight": "Flt1007", "tstart": 57770.0, "tend": 63010.0, "map": "Renfrew_395"},
    "Flt1002_1002.17": {"flight": "Flt1002", "tstart": 63935.0, "tend": 65812.0, "map": "Renfrew_555"},
}


# ---------------------------------------------------------------------------
# Download helpers (stdlib only)
# ---------------------------------------------------------------------------
def _stream_download(url: str, dest: str, expected: int | None = None, retries: int = 3) -> None:
    """Stream a URL to ``dest`` with a progress readout; retries on incomplete transfer."""
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "magnavlab-getdata"})
            with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", expected or 0))
                done = 0
                last_bucket = -1
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    f.write(block)
                    done += len(block)
                    # print progress in ~20% steps only (keeps logs/notebooks tidy)
                    if total and (bucket := int(100 * done / total) // 20) > last_bucket:
                        last_bucket = bucket
                        print(f"  {os.path.basename(dest)}: {100*done/total:3.0f}% "
                              f"({done/1e6:.0f}/{total/1e6:.0f} MB)")
            if total and done < 0.999 * total:
                raise IOError(f"incomplete: {done}/{total} bytes")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == retries:
                raise
            print(f"  retry {attempt}/{retries-1} after error: {str(e)[:80]}")


def resolve_zenodo(record_id: str) -> dict[str, dict]:
    """Return {filename: {'url', 'size'}} for a Zenodo record (via its public API)."""
    api = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "magnavlab"})) as r:
        meta = json.load(r)
    files = {}
    for f in meta.get("files", []):
        key = f["key"]
        files[key] = {"url": f"https://zenodo.org/records/{record_id}/files/{key}?download=1",
                      "size": f.get("size")}
    return files


def list_records() -> None:
    """Print available flights per Zenodo record (queries the API)."""
    for rec, info in ZENODO_RECORDS.items():
        print(f"[{rec}] {info['title']} - {info['aircraft']}")
        try:
            for key, m in sorted(resolve_zenodo(rec).items()):
                if key.endswith(".h5"):
                    print(f"    {key:22s} {(m['size'] or 0)/1e6:7.1f} MB")
        except Exception as e:  # noqa: BLE001
            print(f"    (could not query API: {str(e)[:60]})")


def fetch_flight(name: str, dest_dir: str = "data",
                 records: tuple[str, ...] = _DEFAULT_RECORD_ORDER) -> str:
    """Download a flight HDF5 (e.g. 'Flt1006') into ``dest_dir`` as ``{name}_train.h5``.

    Searches the given Zenodo records (handles both 'Flt1005-train.h5' and
    'Flt1006_train.h5' naming). Skips download if the file already exists.
    """
    dest = os.path.join(dest_dir, f"{name}_train.h5")
    if os.path.exists(dest):
        print(f"  {name}: already present, skip")
        return dest
    for rec in records:
        for key, m in resolve_zenodo(rec).items():
            base = key.lower().replace("-", "_")
            if base.startswith(name.lower()) and base.endswith(".h5"):
                os.makedirs(dest_dir, exist_ok=True)
                print(f"  {name}: downloading from Zenodo {rec} ({key})")
                _stream_download(m["url"], dest, m["size"])
                return dest
    raise FileNotFoundError(f"{name} not found in Zenodo records {records}")


def fetch_maps(names=("Eastern_395",), dest_dir: str = "data/maps",
               keep_tar: bool = False) -> list[str]:
    """Download the Ottawa-area map tarball and extract the requested maps.

    Only the requested maps are extracted. Skips maps that already exist.
    """
    os.makedirs(dest_dir, exist_ok=True)
    unknown = [n for n in names if n not in AVAILABLE_MAPS]
    if unknown:
        raise ValueError(f"Unknown maps {unknown}. Available: {AVAILABLE_MAPS}")
    needed = [n for n in names if not os.path.exists(os.path.join(dest_dir, f"{n}.h5"))]
    if not needed:
        print("  maps: already present, skip")
        return [os.path.join(dest_dir, f"{n}.h5") for n in names]

    tar_dir = os.path.dirname(os.path.abspath(dest_dir))   # e.g. .../data (parent of data/maps)
    tar_path = os.path.join(tar_dir, "ottawa_area_maps_v3.tar.gz")
    if not os.path.exists(tar_path):
        os.makedirs(tar_dir, exist_ok=True)
        print(f"  maps: downloading tarball (~{MAP_ARTIFACT_SIZE/1e6:.0f} MB) for {needed}")
        _stream_download(MAP_ARTIFACT_URL, tar_path, MAP_ARTIFACT_SIZE)

    out = []
    with tarfile.open(tar_path) as tar:
        wanted = {f"{n}.h5" for n in needed}
        for member in tar.getmembers():
            bn = os.path.basename(member.name)
            if bn in wanted:
                member.name = bn                       # strip the archive's top directory
                tar.extract(member, dest_dir)          # noqa: S202 (name sanitized to basename)
                print(f"  extracted {bn}")
                out.append(os.path.join(dest_dir, bn))
    if not keep_tar and os.path.exists(tar_path):
        os.remove(tar_path)
    return out


def ensure_demo_data() -> None:
    """Download the minimal set for the default experiments: Flt1002, Flt1003, Eastern_395."""
    fetch_flight("Flt1002")
    fetch_flight("Flt1003")
    fetch_maps(["Eastern_395"])
