"""Data access: ERA5 winds, GLORYS currents, GSHHG shorelines and ENSO indices.

Run once to fill a data directory; the simulator only reads the files, and no
test touches the network. Every fetch function takes ``dry_run`` and returns
the downloads it made, or with ``dry_run`` would make. Files already present
are skipped and not listed, so an interrupted fetch resumes where it stopped.
Each file is written under a ``.part`` name and renamed once complete, so a
file that exists is whole.

The Copernicus clients (``cdsapi``, ``copernicusmarine``: the ``fetch`` extra)
are imported only when a download needs them; they read their credentials
from their own configuration (``~/.cdsapirc``, ``copernicusmarine login``).

- **ERA5**: 6-hourly 10 m u and v over ``DOMAIN``, one month per request to
  stay under the Climate Data Store's request limit.
- **GLORYS12**: daily surface currents (top level, 0.494 m), one file per
  year, from the multi-year product to June 2021 and the interim product after.
  ``coarsen_currents`` block-averages them to 0.25 deg (a canoe covering
  200 km a day gains nothing from 8 km), and ``build_current_climatology``
  makes the monthly means that pair with pre-1993 wind years.
- **GSHHG 2.3.7**: the full-resolution level-1 shorelines from the GitHub
  release (NOAA no longer hosts it), checked against a pinned SHA-256.
- **ENSO indices**: CPC's RONI and ONI, and PSL's HadISST Nino 3.4 anomaly
  for the 1940s.
"""

from __future__ import annotations

import calendar
import hashlib
import itertools
import os
import shutil
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import enso

DOMAIN = {"north": 30, "south": -50, "west": -165, "east": -65}

ERA5_DATASET = "reanalysis-era5-single-levels"
ERA5_VARIABLES = ("10m_u_component_of_wind", "10m_v_component_of_wind")

GLORYS_MY = "cmems_mod_glo_phy_my_0.083deg_P1D-m"        # 1993-01-01 to 2021-06-30
GLORYS_MYINT = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"  # 2021-07-01 onwards
GLORYS_START = date(1993, 1, 1)
GLORYS_MYINT_START = date(2021, 7, 1)

GSHHG_URL = "https://github.com/GenericMappingTools/gshhg-gmt/releases/download/2.3.7/gshhg-shp-2.3.7.zip"
GSHHG_SHA256 = "8dbbe7e071e77e9e75f2d639239099ebca8d5c16d6a07df8169729d49f15cf41"
GSHHG_MEMBERS = (  # unpacked under out_dir with these relative paths
    "README.TXT",
    "LICENSE.TXT",
    "COPYING.LESSERv3",  # the LGPL text LICENSE.TXT refers to
    "GSHHS_shp/f/GSHHS_f_L1.dbf",
    "GSHHS_shp/f/GSHHS_f_L1.prj",
    "GSHHS_shp/f/GSHHS_f_L1.shx",
    "GSHHS_shp/f/GSHHS_f_L1.shp",
)

# File name -> (URL, parser that must find values in it).
ENSO_FILES = {
    "RONI.ascii.txt": ("https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt", enso.parse_cpc_seasonal),
    "oni.ascii.txt": ("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", enso.parse_cpc_seasonal),
    "nino34.long.anom.data": (
        "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data",
        enso.parse_psl_monthly,
    ),
}

_LON_NAMES = ("longitude", "lon", "x")
_LAT_NAMES = ("latitude", "lat", "y")
_TIME_NAMES = ("time", "valid_time")
_SLAB_STEPS = 31  # time steps coarsened at once: about a month of daily GLORYS in memory
_TIMEOUT_S = 60
_CHUNK_BYTES = 1 << 20


# --- ERA5 -------------------------------------------------------------------------

def era5_request(year, month, hours=(0, 6, 12, 18), area=DOMAIN) -> dict:
    """CDS request for one month of 10 m u and v at ``hours`` (UTC) over ``area``, as NetCDF."""
    n_days = calendar.monthrange(int(year), int(month))[1]
    return {
        "product_type": ["reanalysis"],
        "variable": list(ERA5_VARIABLES),
        "year": [f"{int(year):d}"],
        "month": [f"{int(month):02d}"],
        "day": [f"{d:02d}" for d in range(1, n_days + 1)],
        "time": [f"{int(h):02d}:00" for h in hours],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [area["north"], area["west"], area["south"], area["east"]],
    }


def era5_files(out_dir, years, months) -> list[Path]:
    """The files ``fetch_era5`` writes, ``era5_uv10_YYYY-MM.nc``, year by year and month by month."""
    return [Path(out_dir) / f"era5_uv10_{int(y):d}-{int(m):02d}.nc" for y, m in itertools.product(years, months)]


def fetch_era5(years, months, out_dir, hours=(0, 6, 12, 18), dry_run=False) -> list[dict]:
    """Download ERA5 10 m winds for every month of ``months`` in every year of ``years``.

    Returns ``{"dataset", "request", "target"}`` per month fetched. File names
    do not record ``hours``: keep hourly downloads (the time-step test) in a
    directory of their own.
    """
    years, months, hours = list(years), list(months), list(hours)
    plan = [
        {"dataset": ERA5_DATASET, "request": era5_request(y, m, hours), "target": path}
        for (y, m), path in zip(itertools.product(years, months), era5_files(out_dir, years, months))
        if not path.exists()
    ]
    if dry_run or not plan:
        return plan
    import cdsapi

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for item in plan:
        part = _part(item["target"])
        client.retrieve(item["dataset"], item["request"], str(part))
        os.replace(part, item["target"])
    return plan


# --- GLORYS -------------------------------------------------------------------------

def _glorys_chunks(start: date, end: date) -> list[tuple[str, date, date]]:
    """(dataset id, first day, last day) per file: calendar years, split where the interim product starts."""
    if start < GLORYS_START:
        raise ValueError(f"GLORYS starts on {GLORYS_START}; got {start}")
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    chunks = []
    for year in range(start.year, end.year + 1):
        first, last = max(start, date(year, 1, 1)), min(end, date(year, 12, 31))
        if first < GLORYS_MYINT_START <= last:
            chunks.append((GLORYS_MY, first, GLORYS_MYINT_START - timedelta(days=1)))
            first = GLORYS_MYINT_START
        chunks.append((GLORYS_MY if last < GLORYS_MYINT_START else GLORYS_MYINT, first, last))
    return chunks


def fetch_glorys(start, end, out_dir, dry_run=False) -> list[dict]:
    """Download daily GLORYS12 surface currents (uo, vo) over ``DOMAIN`` from ``start`` to ``end`` (inclusive).

    Dates may be ISO strings. One file per calendar year, named
    ``glorys_uv_<first day>_<last day>.nc``; 2021 comes in two, one from each
    product. Returns ``{"request", "target"}`` per file, the request being the
    ``copernicusmarine.subset`` keyword arguments. With ``dry_run`` each
    request goes to the toolbox's own dry run, which checks it against the
    catalogue (this needs credentials and the network) but downloads nothing.
    """
    out_dir = Path(out_dir)
    plan = []
    for dataset_id, first, last in _glorys_chunks(pd.Timestamp(start).date(), pd.Timestamp(end).date()):
        target = out_dir / f"glorys_uv_{first.isoformat()}_{last.isoformat()}.nc"
        if target.exists():
            continue
        request = {
            "dataset_id": dataset_id,
            "variables": ["uo", "vo"],
            "minimum_longitude": DOMAIN["west"],
            "maximum_longitude": DOMAIN["east"],
            "minimum_latitude": DOMAIN["south"],
            "maximum_latitude": DOMAIN["north"],
            "minimum_depth": 0.0,  # the top level only, at 0.494 m
            "maximum_depth": 1.0,
            "start_datetime": f"{first.isoformat()}T00:00:00",
            "end_datetime": f"{last.isoformat()}T23:59:59",
            "output_directory": str(out_dir),
            "output_filename": _part(target).name,
            # Replace a .part left by an interrupted run rather than writing a numbered copy beside it.
            "overwrite": True,
        }
        plan.append({"request": request, "target": target})
    if not plan:
        return plan
    import copernicusmarine

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        copernicusmarine.subset(**item["request"], dry_run=dry_run)
        if not dry_run:
            os.replace(out_dir / item["request"]["output_filename"], item["target"])
    return plan


def coarsen_currents(path_in, path_out, factor=3):
    """Block-average a current file onto a grid ``factor`` times coarser (1/12 deg -> 0.25 deg for 3).

    Each output cell is the mean of the ocean cells in its ``factor`` x
    ``factor`` block: land (NaN) is skipped, so a coastal block keeps the
    current of its sea part, and an all-land block stays NaN. Coordinates
    become block centres; cells left over at the grid's edges are dropped.
    Time is processed in slabs to bound memory.
    """
    with xr.open_dataset(path_in) as ds:
        x, y, t = _dim(ds, _LON_NAMES), _dim(ds, _LAT_NAMES), _dim(ds, _TIME_NAMES)
        slabs = [
            ds.isel({t: slice(i, i + _SLAB_STEPS)})
            .coarsen({x: factor, y: factor}, boundary="trim")
            .mean(keep_attrs=True)
            for i in range(0, ds.sizes[t], _SLAB_STEPS)
        ]
        out = xr.concat(slabs, dim=t, data_vars="minimal", coords="minimal", compat="override", join="exact")
    step = f"{factor}x{factor} block mean of ocean cells (voyage_sim.fetch.coarsen_currents)"
    out.attrs["history"] = f"{out.attrs['history']}\n{step}" if "history" in out.attrs else step
    _write_netcdf(out, path_out)


def build_current_climatology(paths, path_out):
    """Monthly means of current files (e.g. coarsened GLORYS years) -> NetCDF with a ``month`` dimension (1-12).

    As ``fields.monthly_climatology``: NaN-aware, every time step weighted
    equally, a cell without valid data in a month NaN there; but accumulated
    one file at a time, so decades of data never sit in memory together. All
    files must share one grid. Length-1 dimensions such as GLORYS's depth are
    dropped, leaving each variable (month, latitude, longitude) for
    ``fields.MonthlyClimatologyField``.
    """
    paths = list(paths)
    if not paths:
        raise ValueError("build_current_climatology needs at least one file")
    total = count = None
    times = []
    months = {"month": np.arange(1, 13)}
    with xr.set_options(keep_attrs=True, arithmetic_join="exact"):
        for path in paths:
            with xr.open_dataset(path) as ds:
                t = _dim(ds, _TIME_NAMES)
                data = ds[[v for v in ds.data_vars if t in ds[v].dims]]
                by_month = data.groupby(data[t].dt.month)
                s = by_month.sum(skipna=True).reindex(months, fill_value=0.0)
                c = by_month.count().reindex(months, fill_value=0)
                times += [ds[t].values.min(), ds[t].values.max()]
            total = s if total is None else total + s
            count = c if count is None else count + c
        clim = total / count.where(count > 0)
    x, y = _dim(clim, _LON_NAMES), _dim(clim, _LAT_NAMES)
    clim = clim.isel({d: 0 for d in clim.dims if d not in ("month", x, y) and clim.sizes[d] == 1})
    clim.attrs = {
        "title": "Monthly climatology of surface currents",
        "time_coverage_start": str(min(times)),
        "time_coverage_end": str(max(times)),
        "history": f"monthly means of {len(paths)} files (voyage_sim.fetch.build_current_climatology)",
    }
    _write_netcdf(clim, path_out)


# --- GSHHG and ENSO indices -------------------------------------------------------

def fetch_gshhg(out_dir, dry_run=False) -> list[dict]:
    """Download the GSHHG 2.3.7 shapefile zip, check its SHA-256, and unpack ``GSHHG_MEMBERS`` into ``out_dir``.

    That is the full-resolution level-1 shoreline (``GSHHS_shp/f/GSHHS_f_L1.*``)
    with the README and the LGPL licence; the zip (149 MB) is deleted after.
    Returns ``[{"url", "sha256", "target", "members"}]``, or ``[]`` when every
    member is already there.
    """
    out_dir = Path(out_dir)
    if all((out_dir / name).exists() for name in GSHHG_MEMBERS):
        return []
    plan = [{"url": GSHHG_URL, "sha256": GSHHG_SHA256, "target": out_dir, "members": list(GSHHG_MEMBERS)}]
    if dry_run:
        return plan
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = _part(out_dir / GSHHG_URL.rsplit("/", 1)[-1])
    try:
        digest = _download(GSHHG_URL, archive)
        if digest != GSHHG_SHA256:
            raise ValueError(f"{GSHHG_URL}: SHA-256 is {digest}, expected {GSHHG_SHA256}")
        with zipfile.ZipFile(archive) as zf:
            for name in GSHHG_MEMBERS:
                target = out_dir / name
                target.parent.mkdir(parents=True, exist_ok=True)
                part = _part(target)
                with zf.open(name) as src, open(part, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(part, target)
    finally:
        archive.unlink(missing_ok=True)
    return plan


def fetch_enso_indices(out_dir, dry_run=False) -> list[dict]:
    """Download the ``ENSO_FILES`` (RONI, ONI, HadISST Nino 3.4) into ``out_dir``.

    A download is kept only if its parser finds index values in it, so an
    error page is never saved as data. NOAA revises the indices every month;
    delete a file to fetch it again. Returns ``{"url", "target"}`` per file.
    """
    out_dir = Path(out_dir)
    plan = [
        {"url": url, "target": out_dir / name}
        for name, (url, _) in ENSO_FILES.items()
        if not (out_dir / name).exists()
    ]
    if dry_run or not plan:
        return plan
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in plan:
        parse = ENSO_FILES[item["target"].name][1]
        part = _part(item["target"])
        try:
            _download(item["url"], part)
            if not parse(part.read_text(encoding="latin-1")).notna().any():
                raise ValueError(f"{item['url']}: no index values in the download")
            os.replace(part, item["target"])
        finally:
            part.unlink(missing_ok=True)
    return plan


# --- helpers ------------------------------------------------------------------------

def _part(path: Path) -> Path:
    """Where ``path`` is written before it is complete; the suffix is kept for tools that check it."""
    return path.with_name(f"{path.stem}.part{path.suffix}")


def _download(url: str, path: Path) -> str:
    """Stream ``url`` into ``path``; return the SHA-256 hex digest of the bytes written."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=_TIMEOUT_S) as response, open(path, "wb") as out:
        while chunk := response.read(_CHUNK_BYTES):
            digest.update(chunk)
            out.write(chunk)
    return digest.hexdigest()


def _dim(ds, names):
    """The dimension of ``ds`` whose lower-cased name is one of ``names``."""
    for d in ds.dims:
        if str(d).lower() in names:
            return d
    raise ValueError(f"no dimension named any of {names}; dimensions are {tuple(ds.dims)}")


def _write_netcdf(ds, path):
    """Write float32, compressed, to a ``.part`` file and rename it into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = _part(path)
    ds.to_netcdf(part, encoding={v: {"dtype": "float32", "zlib": True, "complevel": 4} for v in ds.data_vars})
    os.replace(part, path)
