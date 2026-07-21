"""Dataset-registry tests (no network - metadata and idempotency only)."""
from magnavlab import data


def test_records_present():
    assert "4271804" in data.ZENODO_RECORDS
    assert "12723700" in data.ZENODO_RECORDS
    # 12723700 is a superset that adds the Flt1006 calibration flight
    assert "Flt1006" in data.ZENODO_RECORDS["12723700"]["flights"]
    assert "Flt1006" not in data.ZENODO_RECORDS["4271804"]["flights"]


def test_cal_segments_valid():
    for flight, segs in data.CAL_SEGMENTS.items():
        assert flight.startswith("Flt")
        assert segs, f"{flight} has no segments"
        for t0, t1 in segs:
            assert t1 > t0                       # each window is a positive interval


def test_nav_windows_valid():
    for key, w in data.NAV_WINDOWS.items():
        assert w["tend"] > w["tstart"]
        assert w["map"] in data.AVAILABLE_MAPS
        assert w["flight"].startswith("Flt")


def test_fetch_flight_is_idempotent(tmp_path):
    # a pre-existing file must be returned without any network access
    existing = tmp_path / "Flt9999_train.h5"
    existing.write_bytes(b"stub")
    got = data.fetch_flight("Flt9999", str(tmp_path))
    assert got == str(existing)


def test_fetch_maps_rejects_unknown(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        data.fetch_maps(["NoSuchMap"], str(tmp_path))
