import hashlib
import io
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from voyage_sim import enso, fetch, fields, geo


@pytest.fixture
def no_network(monkeypatch):
    """Fail any download or client import; tests that need one patch it back in."""
    def refuse(*args, **kwargs):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", refuse)
    monkeypatch.setitem(sys.modules, "cdsapi", None)  # importing a None entry raises ImportError
    monkeypatch.setitem(sys.modules, "copernicusmarine", None)


def _leftovers(path):
    return sorted(p.name for p in path.rglob("*.part*"))


# --- ERA5 -------------------------------------------------------------------------

def test_era5_request_is_one_month_over_the_domain():
    req = fetch.era5_request(2000, 2)
    assert req["product_type"] == ["reanalysis"]
    assert req["variable"] == ["10m_u_component_of_wind", "10m_v_component_of_wind"]
    assert (req["year"], req["month"]) == (["2000"], ["02"])
    assert req["day"] == [f"{d:02d}" for d in range(1, 30)]  # leap year
    assert len(fetch.era5_request(1900, 2)["day"]) == 28
    assert req["time"] == ["00:00", "06:00", "12:00", "18:00"]
    assert (req["data_format"], req["download_format"]) == ("netcdf", "unarchived")
    assert req["area"] == [30, -165, -50, -65]  # N, W, S, E
    assert fetch.era5_request(2010, 12, hours=range(24))["time"][-1] == "23:00"


def test_era5_dry_run_lists_missing_months_only(tmp_path, no_network):
    files = fetch.era5_files(tmp_path, [1997, 1998], [12, 1])
    assert [f.name for f in files] == [
        "era5_uv10_1997-12.nc", "era5_uv10_1997-01.nc", "era5_uv10_1998-12.nc", "era5_uv10_1998-01.nc",
    ]
    files[0].write_bytes(b"done")
    plan = fetch.fetch_era5([1997, 1998], [12, 1], tmp_path, dry_run=True)
    assert [p["target"] for p in plan] == files[1:]
    assert all(p["dataset"] == "reanalysis-era5-single-levels" for p in plan)
    assert plan[0]["request"] == fetch.era5_request(1997, 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["era5_uv10_1997-12.nc"]
    # Hours given as a one-shot iterator still reach every month's request.
    hourly = fetch.fetch_era5([1999], [1, 2], tmp_path, hours=iter(range(0, 24, 3)), dry_run=True)
    assert [p["request"]["time"] for p in hourly] == [fetch.era5_request(1999, 1, range(0, 24, 3))["time"]] * 2


def test_era5_download_goes_through_a_part_file(tmp_path, monkeypatch):
    calls = []

    class Client:
        def retrieve(self, name, request, target):
            calls.append((name, request, target))
            with open(target, "wb") as f:
                f.write(request["month"][0].encode())

    monkeypatch.setitem(sys.modules, "cdsapi", SimpleNamespace(Client=Client))
    out = tmp_path / "era5"
    plan = fetch.fetch_era5([2001], [3, 4], out)
    assert [c[0] for c in calls] == ["reanalysis-era5-single-levels"] * 2
    assert [c[1] for c in calls] == [fetch.era5_request(2001, 3), fetch.era5_request(2001, 4)]
    assert calls[0][2] == str(out / "era5_uv10_2001-03.part.nc")
    assert [p["target"].read_text() for p in plan] == ["03", "04"]
    assert _leftovers(out) == []
    # Present files are skipped without a client.
    monkeypatch.setitem(sys.modules, "cdsapi", None)
    assert fetch.fetch_era5([2001], [3, 4], out) == []


# --- GLORYS -------------------------------------------------------------------------

def _fake_marine(calls, write=False):
    def subset(**kw):
        calls.append(kw)
        if write and not kw["dry_run"]:
            path = fetch.Path(kw["output_directory"]) / kw["output_filename"]
            path.write_text(kw["dataset_id"])

    return SimpleNamespace(subset=subset)


def test_glorys_files_are_calendar_years_split_at_the_interim_product(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "copernicusmarine", _fake_marine(calls))
    plan = fetch.fetch_glorys("2020-12-01", "2022-01-31", tmp_path, dry_run=True)
    spans = [(p["request"]["dataset_id"], p["request"]["start_datetime"][:10], p["request"]["end_datetime"])
             for p in plan]
    my, myint = "cmems_mod_glo_phy_my_0.083deg_P1D-m", "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
    assert spans == [
        (my, "2020-12-01", "2020-12-31T23:59:59"),
        (my, "2021-01-01", "2021-06-30T23:59:59"),
        (myint, "2021-07-01", "2021-12-31T23:59:59"),
        (myint, "2022-01-01", "2022-01-31T23:59:59"),
    ]
    assert plan[1]["target"] == tmp_path / "glorys_uv_2021-01-01_2021-06-30.nc"
    # The toolbox's own dry run sees each request; nothing is written.
    assert [c.pop("dry_run") for c in calls] == [True] * 4
    assert calls == [p["request"] for p in plan]
    assert list(tmp_path.iterdir()) == []
    req = plan[0]["request"]
    assert req["variables"] == ["uo", "vo"]
    assert (req["minimum_depth"], req["maximum_depth"]) == (0.0, 1.0)
    assert (req["minimum_longitude"], req["maximum_longitude"]) == (-165, -65)
    assert (req["minimum_latitude"], req["maximum_latitude"]) == (-50, 30)
    with pytest.raises(ValueError, match="1993"):
        fetch.fetch_glorys("1992-12-31", "1993-02-01", tmp_path, dry_run=True)
    with pytest.raises(ValueError):
        fetch.fetch_glorys("1999-02-01", "1999-01-31", tmp_path, dry_run=True)


def test_glorys_download_renames_and_skips_present_years(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "copernicusmarine", _fake_marine(calls, write=True))
    (tmp_path / "glorys_uv_1993-01-01_1993-12-31.nc").write_text("old")
    plan = fetch.fetch_glorys("1993-01-01", "1994-12-31", tmp_path)
    assert [p["target"].name for p in plan] == ["glorys_uv_1994-01-01_1994-12-31.nc"]
    assert calls[0]["output_filename"] == "glorys_uv_1994-01-01_1994-12-31.part.nc"
    assert calls[0]["overwrite"] is True and calls[0]["dry_run"] is False
    assert plan[0]["target"].read_text() == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    assert _leftovers(tmp_path) == []


def _currents(time, lat, lon, seed=0):
    rng = np.random.default_rng(seed)
    shape = (len(time), 1, len(lat), len(lon))
    return xr.Dataset(
        {
            "uo": (("time", "depth", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32),
                   {"units": "m s-1"}),
            "vo": (("time", "depth", "latitude", "longitude"), rng.normal(size=shape).astype(np.float32),
                   {"units": "m s-1"}),
        },
        coords={"time": time, "depth": [0.494], "latitude": lat, "longitude": lon},
    )


def test_coarsen_takes_nan_aware_block_means(tmp_path):
    lat = -10 + np.arange(7) / 12  # 7 x 7 cells: the last row and column are trimmed
    lon = -100 + np.arange(7) / 12
    ds = _currents(pd.date_range("2000-01-01", periods=40), lat, lon)
    ds["uo"][:, :, :3, :3] = np.nan                    # an all-land block
    ds["uo"][:, :, 3, 3] = np.nan                      # one land cell in another
    ds.to_netcdf(tmp_path / "in.nc")
    fetch.coarsen_currents(tmp_path / "in.nc", tmp_path / "out.nc")
    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert dict(out.sizes) == {"time": 40, "depth": 1, "latitude": 2, "longitude": 2}
        np.testing.assert_allclose(out.latitude, lat[[1, 4]])
        np.testing.assert_allclose(out.longitude, lon[[1, 4]])
        u = ds.uo.values
        assert np.isnan(out.uo.values[:, 0, 0, 0]).all()
        tol = dict(rtol=1e-6, atol=1e-6)  # float32 data
        np.testing.assert_allclose(out.uo.values[:, 0, 1, 1], np.nanmean(u[:, 0, 3:6, 3:6], axis=(1, 2)), **tol)
        np.testing.assert_allclose(out.vo.values[:, 0, 0, 1], ds.vo.values[:, 0, 0:3, 3:6].mean(axis=(1, 2)), **tol)
        assert out.uo.attrs["units"] == "m s-1" and out.uo.dtype == np.float32
        # The simulator reads the result directly; the depth level is squeezed there.
        field = fields.GriddedField.from_dataset(out, "uo", "vo")
        assert field.u.shape == (40, 2, 2)
    assert _leftovers(tmp_path) == []


def test_climatology_matches_monthly_means_over_all_files(tmp_path):
    lat, lon = np.array([-1.0, -0.75]), np.array([-90.0, -89.75, -89.5])
    years = [_currents(pd.date_range(f"{y}-01-01", f"{y}-12-31"), lat, lon, seed=y) for y in (1993, 1994)]
    years[0]["uo"].values[:31, :, 0, 0] = np.nan  # no data in January 1993 in one cell
    # Another cell has no data in 1994, nor in March 1993: NaN in March only.
    years[1]["vo"].values[:, :, 1, 2] = np.nan
    years[0]["vo"].values[(years[0].time.dt.month == 3).values, :, 1, 2] = np.nan
    paths = []
    for ds in years:
        paths.append(tmp_path / f"{int(ds.time.dt.year[0])}.nc")
        ds.to_netcdf(paths[-1])
    fetch.build_current_climatology(paths, tmp_path / "clim.nc")

    both = xr.concat(years, dim="time")
    u12, v12 = fields.monthly_climatology(
        geo.to_epoch_hours(both.time.values), both.uo.values[:, 0], both.vo.values[:, 0]
    )
    with xr.open_dataset(tmp_path / "clim.nc") as clim:
        assert dict(clim.sizes) == {"month": 12, "latitude": 2, "longitude": 3}
        assert list(clim.month.values) == list(range(1, 13))
        np.testing.assert_allclose(clim.uo.values, u12, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(clim.vo.values, v12, rtol=1e-5, atol=1e-6)
        assert np.isnan(clim.vo.values[2, 1, 2]) and not np.isnan(clim.vo.values[3, 1, 2])
        assert clim.attrs["time_coverage_start"].startswith("1993-01-01")
        assert clim.attrs["time_coverage_end"].startswith("1994-12-31")

    shifted = years[1].assign_coords(longitude=lon + 0.25)
    shifted.to_netcdf(tmp_path / "shifted.nc")
    with pytest.raises(ValueError):
        fetch.build_current_climatology([paths[0], tmp_path / "shifted.nc"], tmp_path / "bad.nc")
    with pytest.raises(ValueError):
        fetch.build_current_climatology([], tmp_path / "none.nc")


# --- GSHHG ------------------------------------------------------------------------

def _gshhg_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in (*fetch.GSHHG_MEMBERS, "SHAPEFILES.TXT", "GSHHS_shp/f/GSHHS_f_L2.shp",
                     "GSHHS_shp/h/GSHHS_h_L1.shp", "WDBII_shp/f/WDBII_border_f_L1.shp"):
            zf.writestr(name, f"contents of {name}")
    return buf.getvalue()


def _serve(monkeypatch, payloads):
    """Patch urlopen to serve ``payloads[url]`` and record the URLs asked for."""
    asked = []

    def urlopen(url, timeout=None):
        asked.append(url)
        return io.BytesIO(payloads[url])

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    return asked


def test_gshhg_unpacks_only_the_full_resolution_level_1_shoreline(tmp_path, monkeypatch, no_network):
    out = tmp_path / "gshhg"
    plan = fetch.fetch_gshhg(out, dry_run=True)
    assert plan[0]["url"].endswith("/2.3.7/gshhg-shp-2.3.7.zip")
    assert plan[0]["sha256"] == "8dbbe7e071e77e9e75f2d639239099ebca8d5c16d6a07df8169729d49f15cf41"
    assert not out.exists()

    data = _gshhg_zip()
    monkeypatch.setattr(fetch, "GSHHG_SHA256", hashlib.sha256(data).hexdigest())
    asked = _serve(monkeypatch, {fetch.GSHHG_URL: data})
    done = fetch.fetch_gshhg(out)
    assert [(d["url"], d["target"], d["members"]) for d in done] == [(fetch.GSHHG_URL, out, list(fetch.GSHHG_MEMBERS))]
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == sorted(fetch.GSHHG_MEMBERS)
    assert (out / "GSHHS_shp/f/GSHHS_f_L1.shp").read_text() == "contents of GSHHS_shp/f/GSHHS_f_L1.shp"
    assert asked == [fetch.GSHHG_URL]
    assert fetch.fetch_gshhg(out) == [] and asked == [fetch.GSHHG_URL]


def test_gshhg_hash_mismatch_keeps_nothing(tmp_path, monkeypatch):
    _serve(monkeypatch, {fetch.GSHHG_URL: _gshhg_zip()})
    with pytest.raises(ValueError, match="SHA-256"):
        fetch.fetch_gshhg(tmp_path)
    assert list(tmp_path.iterdir()) == []


# --- ENSO indices -------------------------------------------------------------------

RONI = " SEAS   YR  ANOM\n  DJF 1998  1.98\n  JFM 1998  1.51\n"
ONI = " SEAS  YR   TOTAL   ANOM\n  DJF 1998  28.71  2.24\n  JFM 1998  28.40  1.92\n"
PSL = " 1997 1997\n 1997" + " 0.50" * 12 + "\n  -99.99\n  Nino 3.4 anomaly\n"


def test_enso_indices_download_and_parse(tmp_path, monkeypatch, no_network):
    urls = {name: url for name, (url, _) in fetch.ENSO_FILES.items()}
    assert urls == {
        "RONI.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt",
        "oni.ascii.txt": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        "nino34.long.anom.data": "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data",
    }
    (tmp_path / "oni.ascii.txt").write_text(ONI)
    plan = fetch.fetch_enso_indices(tmp_path, dry_run=True)
    assert [p["target"].name for p in plan] == ["RONI.ascii.txt", "nino34.long.anom.data"]

    asked = _serve(monkeypatch, {urls["RONI.ascii.txt"]: RONI.encode(), urls["nino34.long.anom.data"]: PSL.encode()})
    assert fetch.fetch_enso_indices(tmp_path) == plan
    assert asked == [p["url"] for p in plan]
    roni = enso.parse_cpc_seasonal((tmp_path / "RONI.ascii.txt").read_text())
    assert roni[pd.Period("1998-02", "M")] == pytest.approx(1.51)
    assert enso.parse_psl_monthly((tmp_path / "nino34.long.anom.data").read_text()).eq(0.5).all()
    assert _leftovers(tmp_path) == []
    assert fetch.fetch_enso_indices(tmp_path) == []


def test_enso_error_page_is_not_saved(tmp_path, monkeypatch):
    page = b"<html><body>Service unavailable</body></html>"
    _serve(monkeypatch, {url: page for url, _ in fetch.ENSO_FILES.values()})
    with pytest.raises(ValueError, match="no index values"):
        fetch.fetch_enso_indices(tmp_path)
    assert list(tmp_path.iterdir()) == []
