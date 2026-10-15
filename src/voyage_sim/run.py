"""One command per scenario grid: a config in, every cell's voyages and their summary out.

``voyage-sim run configs/<grid>.yaml`` builds the land catalog, one sighting
raster per rule (cached), the vessels, strategies, departures and targets
named by the config, sails every cell of the grid and writes
``outputs/<name>/``:

- ``<cell>.npz``: the cell's ``VoyageResults``; ``meta["run"]`` holds every
  setting the voyages depend on;
- ``summary.csv``: success odds with year-bootstrap intervals and days at
  sea, per cell, sighting rule, supply limit and label (launch month; ENSO
  phase when an index file is configured, with ``enso_source`` saying which
  index labelled it: ``cpc`` for RONI or ONI, ``hadisst`` for the HadISST
  Nino 3.4 anomaly that labels the months before them, the 1940s);
- ``resolved_config.yaml``: the config with every default filled in.

Files are written under a temporary name and then renamed, so an
interrupted run leaves no half-written file. The config schema is documented
at the top of ``configs/synthetic_demo.yaml``.

Every cell launches the same schedule. A voyage's id is a 32-bit key from its
cell's name (``cell_key``) times ``ID_STRIDE`` plus its launch's index in the
schedule, so ids never collide across cells (two names with one key are
refused). Every random draw is keyed by id, so a voyage's track does not
depend on how launches are batched, which cells run with it or where its cell
sits in the config, and reruns give byte-identical files.

Launches sail in time-ordered batches. File-based winds and currents are read
one batch window at a time (a launch month, plus the longest wait in port and
horizon of the cells), and every cell sails the batch before the next window
is read. Synthetic fields need no window, so their launches sail in one batch.
Launches before ``climatology_before`` (GLORYS starts in 1993) pair the ERA5
winds with the monthly current climatology instead: their results are flagged
``current_is_climatology`` and their summary label ``currents`` reads
``climatology``.

``summary.csv`` covers every cell of the config whose ``.npz`` is in the
output directory, readable, and made with the current settings, land
included (a digest of the catalog's landmasses), so a grid can be run a few
cells at a time (``--cells``). Everything a run needs (data files, the GSHHG
shorelines when ``gshhg_dir`` names them, ENSO index files, a writable output
directory) is checked before the slow steps.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import glob
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from voyage_sim import __version__, analyze, catalog, enso, fetch, geo, landfall
from voyage_sim.fields import (
    Environment,
    FunctionField,
    GriddedField,
    MonthlyClimatologyField,
)
from voyage_sim.landfall import LandfallDetector, SightingRaster, SightingRule
from voyage_sim.simulate import (
    STATUS_NAMES,
    Launches,
    SimConfig,
    VoyageResults,
    simulate,
)
from voyage_sim.strategies import Strategy, make_strategy
from voyage_sim.vessels import Vessel, load_vessels

log = logging.getLogger(__name__)

ID_STRIDE = 1_000_000_000      # voyage id = cell_key(name) * ID_STRIDE + launch index (fits int64)
LABELS = ("month", "enso", "enso_source", "year", "currents")   # label columns analysis.by may group by
CELL_COLUMNS = ("cell", "departure", "target", "vessel", "strategy")   # lead columns of summary.csv
_TARGETED = ("hold_course", "wait_for_wind", "latitude")   # strategies that steer for a target
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")      # cell and rule names: safe as file names
_LON_NAMES, _LAT_NAMES, _TIME_NAMES = ("longitude", "lon", "x"), ("latitude", "lat", "y"), ("time", "valid_time")
_KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)


# --- synthetic fields ---------------------------------------------------------

# Pacific coast of South America, (lat, lon), south to north: where the Humboldt Current runs.
_COAST_LAT = np.array([-50.0, -40.0, -30.0, -20.0, -18.4, -15.0, -10.0, -6.0, -4.5, -2.0, 0.0, 2.0])
_COAST_LON = np.array([-75.0, -73.8, -71.6, -70.2, -70.3, -75.3, -78.2, -80.8, -81.3, -81.0, -80.1, -78.7])
_HOURS_PER_YEAR = 365.2425 * geo.HOURS_PER_DAY
_AUGUST_15 = 226.5 / 365.2425   # fraction of the year: the belts' northernmost, strongest trades


def _rise(x):
    """Smooth step from 0 to 1 around x = 0."""
    return 0.5 * (1.0 + np.tanh(x))


def _in_domain(lon, lat):
    return ((lon >= fetch.DOMAIN["west"]) & (lon <= fetch.DOMAIN["east"])
            & (lat >= fetch.DOMAIN["south"]) & (lat <= fetch.DOMAIN["north"]))


def pacific_trades(seasonal: bool = True) -> tuple[FunctionField, FunctionField]:
    """Idealized eastern-Pacific (wind, current): physically plausible, not tuned to any data.

    Wind: easterly trades of 6.5 m/s from about 25 S to 20 N, with a 2.5 m/s
    southerly component south of the ITCZ at 6 N (the southeast trades) and a
    northerly one north of it (northeast trades), and weaker along it (the
    doldrums); westerlies of 8 m/s poleward of about 35 S, and light winds in
    the subtropical high between. With ``seasonal``, the belts sit 3 deg
    further north and the trades blow 15% harder in mid-August (austral
    winter) than on average, 3 deg south and 15% lighter in mid-February.

    Current: the South Equatorial Current, 0.3 m/s westward at 1 S and fading
    over about 7 deg of latitude; and the Humboldt Current, 0.15 m/s northward
    along the South American coast from 42 S to 5 S, fading over 300 km
    offshore. Both fields are NaN outside the study domain (``fetch.DOMAIN``),
    which ends a voyage there as real data would.
    """

    def wind(lon, lat, t):
        s = np.cos(2 * np.pi * (t / _HOURS_PER_YEAR - _AUGUST_15)) if seasonal else 0.0
        lat_s = lat - 3.0 * s
        trades = 6.5 * (1.0 + 0.15 * s) * _rise((lat_s + 25.0) / 3.0) * _rise((20.0 - lat_s) / 3.0)
        doldrums = 1.0 - 0.6 * np.exp(-(((lat_s - 6.0) / 3.0) ** 2))
        u = -trades * doldrums + 8.0 * _rise((-35.0 - lat_s) / 4.0)
        v = -trades * (2.5 / 6.5) * np.tanh((lat_s - 6.0) / 4.0)
        return _masked(lon, lat, u, v)

    def current(lon, lat, t):
        u = -0.3 * np.exp(-(((lat + 1.0) / 7.0) ** 2))
        offshore_km = np.maximum(np.interp(lat, _COAST_LAT, _COAST_LON) - lon, 0.0) * _KM_PER_DEG * np.cos(
            np.radians(lat))
        v = 0.15 * np.exp(-offshore_km / 300.0) * _rise((lat + 42.0) / 3.0) * _rise((-5.0 - lat) / 2.0)
        return _masked(lon, lat, u, v)

    return FunctionField(wind), FunctionField(current)


def _masked(lon, lat, u, v):
    inside = _in_domain(lon, lat)
    return np.where(inside, u, np.nan), np.where(inside, v, np.nan)


SYNTHETIC_FIELDS = {"pacific_trades": pacific_trades}


# --- config -------------------------------------------------------------------

def _keys(cfg, what, required=(), **defaults):
    """Check a config mapping's keys; return a copy with defaults filled in."""
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{what} must be a mapping")
    if unknown := set(cfg) - set(required) - set(defaults):
        raise ValueError(f"{what}: unknown keys {sorted(unknown)}; allowed: {sorted({*required, *defaults})}")
    if missing := set(required) - set(cfg):
        raise ValueError(f"{what}: missing keys {sorted(missing)}")
    return {**defaults, **cfg}


def _name(value, what):
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ValueError(f"{what} must be letters, digits, '_', '.' or '-' (it names files), got {value!r}")
    return value


def _names(value, what) -> list[str]:
    """A name or a list of names, as a list."""
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{what} must be a name or a list of names, got {value!r}")
    return list(names)


def _path(value, what, optional=False):
    if isinstance(value, str) or (optional and value is None):
        return value
    raise ValueError(f"{what} must be a path{' or null' if optional else ''}, got {value!r}")


def _number(value, what, kind=float):
    """A config value as a float, or as an int for ``kind=int`` (whole numbers only)."""
    if kind is int and isinstance(value, int):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number, got {value!r}") from None
    if kind is int:
        if not x.is_integer():
            raise ValueError(f"{what} must be a whole number, got {value!r}")
        return int(x)
    return x


def _date(value, unit, what):
    """A YAML date or ISO string, normalized to an ISO string at ``unit`` ("D" or "s")."""
    try:
        return str(np.datetime64(str(value), unit))
    except ValueError as err:
        raise ValueError(f"{what}: not a date: {value!r}") from err


def _float_list(values, what):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{what} must be a non-empty list")
    return [_number(x, f"{what}[{k}]") for k, x in enumerate(values)]


def _data_files(spec, what, path_key, u, v):
    """A ``{paths or path, u, v}`` section: where the NetCDF files are and their variable names."""
    spec = _keys(spec, what, (path_key,), u=u, v=v)
    _path(spec[path_key], f"{what}.{path_key}")
    return spec


def _environment(env):
    kind = env.get("type") if isinstance(env, Mapping) else None
    if kind == "synthetic":
        env = _keys(env, "environment", ("type", "field"), params={}, wind_cap_ms=50.0)
        if env["field"] not in SYNTHETIC_FIELDS:
            raise ValueError(f"environment.field must be one of {sorted(SYNTHETIC_FIELDS)}, got {env['field']!r}")
        if not isinstance(env["params"], Mapping):
            raise ValueError("environment.params must be a mapping")
        try:
            inspect.signature(SYNTHETIC_FIELDS[env["field"]]).bind(**env["params"])
        except TypeError as err:
            raise ValueError(f"environment.params: {err}") from err
        env["params"] = dict(env["params"])
    elif kind == "files":
        env = _keys(env, "environment", ("type", "wind", "current"), climatology=None,
                    climatology_before="1993-01-01", wind_cap_ms=50.0)
        env["wind"] = _data_files(env["wind"], "environment.wind", "paths", "u10", "v10")
        env["current"] = _data_files(env["current"], "environment.current", "paths", "uo", "vo")
        if env["climatology"] is not None:
            env["climatology"] = _data_files(env["climatology"], "environment.climatology", "path", "uo", "vo")
        env["climatology_before"] = _date(env["climatology_before"], "s", "environment.climatology_before")
    else:
        raise ValueError("environment.type must be synthetic or files")
    if env["wind_cap_ms"] is not None:
        env["wind_cap_ms"] = _number(env["wind_cap_ms"], "environment.wind_cap_ms")
    return env


def _climatology_before(env) -> float:
    """Epoch hours before which launches pair the winds with the current climatology (-inf: none do)."""
    if env["type"] != "files" or env["climatology"] is None:
        return -np.inf
    return float(geo.to_epoch_hours(env["climatology_before"]))


def _cell(cell, k):
    what = f"cells[{k}]"
    cell = _keys(cell, what, ("name", "departure", "vessel", "strategy", "destinations", "max_days_at_sea"),
                 target=None)
    _name(cell["name"], f"{what}.name")
    strategy = cell["strategy"]
    if isinstance(strategy, str):
        strategy = {"name": strategy}
    if not isinstance(strategy, Mapping) or "name" not in strategy:
        raise ValueError(f"{what}.strategy must be a name or a mapping with a name and parameters")
    try:
        make_strategy(**strategy)   # checks the name and parameters
    except TypeError as err:        # a parameter of the wrong type
        raise ValueError(f"{what}.strategy: {err}") from err
    cell["strategy"] = dict(strategy)
    dest = _keys(cell["destinations"], f"{what}.destinations", (), regions=[], groups=[])
    dest = {key: _names(dest[key], f"{what}.destinations.{key}") for key in ("regions", "groups")}
    if not dest["regions"] and not dest["groups"]:
        raise ValueError(f"{what}.destinations: give regions or groups")
    if unknown := set(dest["regions"]) - set(catalog.REGIONS):
        raise ValueError(f"{what}.destinations: unknown regions {sorted(unknown)}; choose from {catalog.REGIONS}")
    cell["destinations"] = dest
    cell["max_days_at_sea"] = _number(cell["max_days_at_sea"], f"{what}.max_days_at_sea")
    if not 0 < cell["max_days_at_sea"] < np.inf:
        raise ValueError(f"{what}.max_days_at_sea must be positive")
    return cell


def load_config(path) -> dict:
    """Read a grid config, check it and fill in its defaults (schema: top of ``configs/synthetic_demo.yaml``).

    Paths in the config stay as written, relative to the config file's
    directory; the result is plain YAML types, as ``resolved_config.yaml``
    records it.
    """
    with open(path, encoding="utf-8") as fh:
        try:
            doc = yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ValueError(f"{path} is not valid YAML: {err}") from err
    cfg = _keys(doc, str(path), ("name", "environment", "catalog", "rules", "vessels", "places", "launches",
                                 "cells"), seed=0, raster={}, sim={}, analysis={})
    _name(cfg["name"], "name")
    cfg["seed"] = _number(cfg["seed"], "seed", int)
    cfg["environment"] = _environment(cfg["environment"])
    cfg["catalog"] = _keys(cfg["catalog"], "catalog", ("config",), gshhg_dir=None, cache_dir=None)
    for key, value in cfg["catalog"].items():
        _path(value, f"catalog.{key}", optional=key != "config")
    _path(cfg["vessels"], "vessels")
    _path(cfg["places"], "places")

    if not isinstance(cfg["rules"], list) or not cfg["rules"]:
        raise ValueError("rules must be a non-empty list")
    rules = []
    for k, rule in enumerate(cfg["rules"]):
        rule = _keys(rule, f"rules[{k}]", ("name",), **{f.name: f.default for f in dataclasses.fields(SightingRule)
                                                       if f.name != "name"})
        _name(rule["name"], f"rules[{k}].name")
        rule = {key: value if key in ("name", "mode") else _number(value, f"rules[{k}].{key}")
                for key, value in rule.items()}
        SightingRule(**rule)   # checks the values
        rules.append(rule)
    if len({r["name"] for r in rules}) != len(rules):
        raise ValueError("rule names must be unique")
    cfg["rules"] = rules

    domain = [fetch.DOMAIN["west"], fetch.DOMAIN["east"], fetch.DOMAIN["south"], fetch.DOMAIN["north"]]
    raster = _keys(cfg["raster"], "raster", (), bounds=domain, res_deg=0.05, cap_km=250.0, densify_km=1.0,
                   substep_km=2.0)
    cfg["raster"] = {key: _float_list(value, "raster.bounds") if key == "bounds" else _number(value, f"raster.{key}")
                     for key, value in raster.items()}
    if len(cfg["raster"]["bounds"]) != 4:
        raise ValueError("raster.bounds must be [lon_min, lon_max, lat_min, lat_max]")

    launches = _keys(cfg["launches"], "launches", ("start", "end"), per_day=1, hour=0.0, every_days=1)
    launches = {"start": _date(launches["start"], "D", "launches.start"),
                "end": _date(launches["end"], "D", "launches.end"),
                "per_day": _number(launches["per_day"], "launches.per_day", int),
                "hour": _number(launches["hour"], "launches.hour"),
                "every_days": _number(launches["every_days"], "launches.every_days", int)}
    if not (launches["start"] <= launches["end"] and launches["per_day"] >= 1 and launches["every_days"] >= 1
            and 0 <= launches["hour"] < 24):
        raise ValueError("launches: need start <= end, per_day >= 1, every_days >= 1 and 0 <= hour < 24")
    cfg["launches"] = launches
    if launch_times(launches).size >= ID_STRIDE:
        raise ValueError(f"launches: at most {ID_STRIDE - 1} launches per cell")

    sim = _keys(cfg["sim"], "sim", (), dt_hours=6.0, integrator="midpoint", speed_sd=0.10, heading_sd_deg=10.0,
                diffusivity_m2s=30.0, track_every=4)
    cfg["sim"] = {key: value if key == "integrator"
                  else _number(value, f"sim.{key}", int if key == "track_every" else float)
                  for key, value in sim.items()}
    SimConfig(**cfg["sim"])   # checks the values

    analysis = _keys(cfg["analysis"], "analysis", (), supplies=[30, 60, 90, 120], by=["month"], n_boot=2000,
                     enso_index=None, enso_index_early=None)
    analysis["supplies"] = _float_list(analysis["supplies"], "analysis.supplies")
    analysis["by"] = [analysis["by"]] if isinstance(analysis["by"], str) else list(analysis["by"])
    if unknown := set(analysis["by"]) - set(LABELS):
        raise ValueError(f"analysis.by: unknown labels {sorted(unknown)}; choose from {LABELS}")
    for key in ("enso_index", "enso_index_early"):
        _path(analysis[key], f"analysis.{key}", optional=True)
    if {"enso", "enso_source"} & set(analysis["by"]) and analysis["enso_index"] is None:
        raise ValueError("analysis.by has enso but no enso_index file is configured")
    if analysis["enso_index_early"] is not None and analysis["enso_index"] is None:
        raise ValueError("analysis.enso_index_early labels the months before analysis.enso_index; configure both")
    analysis["n_boot"] = _number(analysis["n_boot"], "analysis.n_boot", int)
    if analysis["n_boot"] < 1:
        raise ValueError(f"analysis.n_boot must be at least 1, got {analysis['n_boot']}")
    cfg["analysis"] = analysis

    if not isinstance(cfg["cells"], list) or not cfg["cells"]:
        raise ValueError("cells must be a non-empty list")
    cfg["cells"] = [_cell(cell, k) for k, cell in enumerate(cfg["cells"])]
    names = [cell["name"] for cell in cfg["cells"]]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise ValueError(f"duplicate cell names {dupes}")
    for cell in cfg["cells"]:
        if not any(s <= cell["max_days_at_sea"] for s in analysis["supplies"]):
            raise ValueError(f"cell {cell['name']!r}: no supply limit within its max_days_at_sea")
    return cfg


def launch_times(spec) -> np.ndarray:
    """Launch times (epoch hours), ascending: ``per_day`` voyages at ``hour`` UTC every
    ``every_days`` days from ``start`` to ``end`` (inclusive)."""
    day = np.timedelta64(1, "D")
    days = np.arange(np.datetime64(spec["start"], "D"), np.datetime64(spec["end"], "D") + day,
                     spec["every_days"] * day)
    return np.repeat(geo.to_epoch_hours(days) + spec["hour"], spec["per_day"])


def _resolve(base, path):
    """A config path, relative to the config file's directory unless absolute."""
    path = Path(path)
    return path if path.is_absolute() else base / path


# --- environments -----------------------------------------------------------------

class SyntheticSource:
    """A named analytic field: one environment at every time, so all launches sail in one batch."""

    def __init__(self, env):
        wind, current = SYNTHETIC_FIELDS[env["field"]](**env["params"])
        self.env = Environment(wind, current, wind_cap_ms=env["wind_cap_ms"])

    def batches(self, times) -> list[np.ndarray]:
        return [np.arange(len(times))]

    def check(self, t0, t1):
        """Analytic fields cover every time."""

    def window(self, t0, t1) -> Environment:
        return self.env


class FileSource:
    """ERA5 winds and GLORYS currents from NetCDF files, read one batch window at a time.

    A window starting before ``climatology_before`` pairs the winds with the
    monthly current climatology (``current_is_climatology``); batches never
    straddle that time. Current files are indexed only once a window needs them,
    so a run wholly before GLORYS needs none.
    """

    def __init__(self, env, base):
        self.env, self.base = env, base
        self.wind = _FileSeries(base, env["wind"], "wind")
        self.climatology_before = _climatology_before(env)

    @cached_property
    def current(self):
        return _FileSeries(self.base, self.env["current"], "current")

    @cached_property
    def climatology(self):
        clim = self.env["climatology"]
        return _load_climatology(_resolve(self.base, clim["path"]), clim["u"], clim["v"])

    def batches(self, times) -> list[np.ndarray]:
        """Launch indices by launch month, split where the climatology ends."""
        month = geo.from_epoch_hours(times).astype("datetime64[M]").astype(np.int64)
        key = 2 * month + (times < self.climatology_before)
        return np.split(np.arange(len(times)), np.flatnonzero(np.diff(key)) + 1)

    def check(self, t0, t1):
        """Raise unless the files cover epoch hours [t0, t1] without gaps."""
        self.wind.span(t0, t1)
        if t0 < self.climatology_before:
            _ = self.climatology   # loading it checks the file
        else:
            self.current.span(t0, t1)

    def window(self, t0, t1) -> Environment:
        """Winds and currents covering epoch hours [t0, t1]."""
        cap = self.env["wind_cap_ms"]
        if t0 < self.climatology_before:
            return Environment(self.wind.window(t0, t1), self.climatology, current_is_climatology=True,
                               wind_cap_ms=cap)
        return Environment(self.wind.window(t0, t1), self.current.window(t0, t1), wind_cap_ms=cap)


def make_source(env, base) -> SyntheticSource | FileSource:
    """The environment source of a resolved ``environment`` config section."""
    return SyntheticSource(env) if env["type"] == "synthetic" else FileSource(env, base)


def _iso(hours):
    return str(geo.from_epoch_hours(hours))


def _dim(da, names):
    for d in da.dims:
        if str(d).lower() in names:
            return d
    raise ValueError(f"{da.name}: no dimension named any of {names}; it has {da.dims}")


class _FileSeries:
    """A (u, v) pair along time, spread over NetCDF files (one per month or year)."""

    def __init__(self, base, spec, what):
        self.u, self.v, self.what = spec["u"], spec["v"], what
        self.paths = sorted(glob.glob(str(_resolve(base, spec["paths"]))))
        if not self.paths:
            raise FileNotFoundError(f"{what}: no files match {spec['paths']} (relative to {base})")
        times, files, steps = [], [], []
        for k, path in enumerate(self.paths):
            with xr.open_dataset(path) as ds:
                if missing := [name for name in (self.u, self.v) if name not in ds.data_vars]:
                    raise ValueError(f"{what}: {path} has no variable {missing[0]!r}; it has {sorted(ds.data_vars)} "
                                     f"(set environment.{what}.u and .v)")
                t = geo.to_epoch_hours(ds[_dim(ds[self.u], _TIME_NAMES)].values)
            times.append(t)
            files.append(np.full(t.size, k))
            steps.append(np.arange(t.size))
        order = np.argsort(np.concatenate(times), kind="stable")
        self.times = np.concatenate(times)[order]
        self.file, self.step = np.concatenate(files)[order], np.concatenate(steps)[order]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError(f"{what}: files overlap in time")
        self.interval = float(np.median(np.diff(self.times))) if self.times.size > 1 else np.inf

    def span(self, t0, t1) -> np.ndarray:
        """Indices into ``times`` from the last time at or before t0 to the first at or after t1."""
        i0 = np.searchsorted(self.times, t0, side="right") - 1
        i1 = np.searchsorted(self.times, t1, side="left")
        if i0 < 0 or i1 >= self.times.size:
            raise ValueError(f"{self.what} files cover {_iso(self.times[0])} to {_iso(self.times[-1])}; "
                             f"a batch needs {_iso(t0)} to {_iso(t1)}")
        sel = np.arange(i0, i1 + 1)
        gaps = np.flatnonzero(np.diff(self.times[sel]) > 1.5 * self.interval)
        if gaps.size:
            g = sel[gaps[0]]
            raise ValueError(f"{self.what}: no data from {_iso(self.times[g])} to {_iso(self.times[g + 1])}; "
                             "is a file missing?")
        return sel

    def window(self, t0, t1) -> GriddedField:
        """The data over ``span(t0, t1)``.

        Each file's part is normalized (ascending latitude) and copied into one
        array per component, allocated once, so reading takes about the
        window's size plus one file's part (a window of hourly winds is GBs).
        """
        sel = self.span(t0, t1)
        u = v = lon = lat = None
        times, at = [], 0
        for k in dict.fromkeys(self.file[sel].tolist()):   # files in time order
            steps = self.step[sel][self.file[sel] == k]
            with xr.open_dataset(self.paths[k]) as ds:
                t_dim = _dim(ds[self.u], _TIME_NAMES)
                part = GriddedField.from_dataset(ds[[self.u, self.v]].isel({t_dim: steps}), self.u, self.v)
            if u is None:
                lon, lat = part.lon, part.lat
                u = np.empty((sel.size,) + part.u.shape[1:], dtype=part.u.dtype)
                v = np.empty_like(u)
            elif not (np.array_equal(part.lon, lon) and np.array_equal(part.lat, lat)):
                raise ValueError(f"{self.what}: {self.paths[k]} is on another grid than the files before it")
            n = part.u.shape[0]
            u[at:at + n], v[at:at + n] = part.u, part.v
            times.append(part.time)
            at += n
            del part
        return GriddedField(lon, lat, np.concatenate(times), u, v, name=self.what)


def _load_climatology(path, u, v) -> MonthlyClimatologyField:
    """A monthly climatology file (``fetch.build_current_climatology``: dims month 1-12, latitude, longitude)."""
    with xr.open_dataset(path) as ds:
        x, y = _dim(ds[u], _LON_NAMES), _dim(ds[u], _LAT_NAMES)
        if "month" not in ds[u].dims or not np.array_equal(ds["month"].values, np.arange(1, 13)):
            raise ValueError(f"{path}: {u} needs a month dimension holding 1-12 in order")

        def values(name):
            da = ds[name]
            extra = [d for d in da.dims if d not in ("month", y, x)]
            return (da.squeeze(extra, drop=True) if extra else da).transpose("month", y, x).values

        return MonthlyClimatologyField(ds[x].values, ds[y].values, values(u), values(v), name="climatology")


# --- land and cells -----------------------------------------------------------------

def landmass_digest(landmasses) -> str:
    """SHA-256 of all a voyage sees of the land: every landmass's name, group, kind, height and shoreline."""
    h = hashlib.sha256()
    for lm in landmasses:
        h.update(json.dumps([lm.name, lm.group, lm.kind, lm.height_m, [len(r) for r in lm.rings]]).encode())
        for ring in lm.rings:
            h.update(np.ascontiguousarray(ring, dtype=np.float64).tobytes())
    return h.hexdigest()


def _raster(landmasses, rule, raster_cfg, cache_dir, digest=None) -> SightingRaster:
    """The rule's sighting raster, from the cache when the landmasses (``digest``, computed if not
    given), rule, settings and code match."""
    build = {key: raster_cfg[key] for key in ("bounds", "res_deg", "densify_km", "cap_km")}
    path = None
    if cache_dir is not None:
        h = hashlib.sha256(Path(landfall.__file__).read_bytes())
        h.update(json.dumps([dataclasses.asdict(rule), build], sort_keys=True).encode())
        h.update((digest or landmass_digest(landmasses)).encode())
        path = cache_dir / f"raster-{rule.name}-{h.hexdigest()[:20]}.npz"
        if path.is_file():
            return SightingRaster.load(path)
    log.info("building the sighting raster for rule %s", rule.name)
    raster = SightingRaster.build(landmasses, rule, **build)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        raster.save(tmp)
        os.replace(tmp, path)
    return raster


def cell_key(name) -> int:
    """A cell's 32-bit key, from its name; part of every voyage id, so of every random draw."""
    return int.from_bytes(hashlib.sha256(str(name).encode("utf-8")).digest()[:4], "big")


@dataclass
class Cell:
    """One cell of the grid, resolved against the vessels, places and catalog."""

    key: int                   # cell_key(name); part of every voyage id
    name: str
    vessel: Vessel
    strategy: Strategy
    sim: SimConfig
    departure: dict            # name, lon, lat, group from the places file
    target: dict | None
    origin_group: int          # index into the detector's groups
    dest_groups: np.ndarray    # (n_groups,) bool
    spec: dict                 # every setting its voyages depend on, stored as meta["run"]

    def launches(self, idx, t) -> Launches:
        """Launches number ``idx`` of the schedule, at epoch hours ``t``."""
        target = self.target or {"lon": np.nan, "lat": np.nan}
        return Launches(self.key * ID_STRIDE + np.asarray(idx, dtype=np.int64), self.departure["lon"],
                        self.departure["lat"], t, self.origin_group, target["lon"], target["lat"],
                        dest_groups=self.dest_groups)


def _yaml(path):
    with open(path, encoding="utf-8") as fh:
        try:
            return yaml.safe_load(fh)
        except yaml.YAMLError as err:
            raise ValueError(f"{path} is not valid YAML: {err}") from err


def _place(entry, what):
    """A place's name, lon, lat and group, checked."""
    if entry is None:
        return None
    if missing := [key for key in ("name", "lon", "lat", "group") if key not in entry]:
        raise ValueError(f"{what}: no {', '.join(missing)}")
    return {"name": entry["name"], "lon": _number(entry["lon"], f"{what}: lon"),
            "lat": _number(entry["lat"], f"{what}: lat"), "group": entry["group"]}


def resolve_cells(cfg, base, cat, groups, digest=None) -> list[Cell]:
    """The config's cells with their vessels, strategies, places and destination groups.

    ``groups`` are the detector's group names, in its order. A cell's ``spec``
    holds its settings from the config, its places, its vessel's entry in the
    vessels file and a digest of the catalog's landmasses (``digest``,
    computed if not given), but not its position in the config.
    """
    vessels_path = _resolve(base, cfg["vessels"])
    vessels, vessel_entries = load_vessels(vessels_path), _yaml(vessels_path)["vessels"]
    places_path = _resolve(base, cfg["places"])
    places_doc = _yaml(places_path)
    places = {}
    for kind in ("departures", "targets"):
        entries = (places_doc or {}).get(kind) or []
        if not all(isinstance(p, Mapping) and "name" in p for p in entries):
            raise ValueError(f"{places_path}: every entry of {kind} needs a name")
        places[kind] = {p["name"]: p for p in entries}
    # Where the cache lives changes no voyage, so moving it leaves earlier results valid.
    shared = {key: cfg[key] for key in ("seed", "environment", "rules", "raster", "launches", "sim")}
    shared["catalog"] = {key: value for key, value in cfg["catalog"].items() if key != "cache_dir"}
    shared["catalog"]["digest"] = digest or landmass_digest(cat.landmasses)
    keys = {}
    for c in cfg["cells"]:
        if (other := keys.setdefault(cell_key(c["name"]), c["name"])) != c["name"]:
            raise ValueError(f"cells {other!r} and {c['name']!r} have the same id key; rename one")
    cells = []
    for c in cfg["cells"]:
        what = f"cell {c['name']!r}"
        lookups = (("departure", "departures", places["departures"]), ("target", "targets", places["targets"]),
                   ("vessel", "vessels", vessels))
        for key, kind, table in lookups:
            if c[key] is not None and c[key] not in table:
                raise ValueError(f"{what}: {key} {c[key]!r} is not among the {kind}: {sorted(table)}")
        vessel = vessels[c["vessel"]]
        strategy = make_strategy(**c["strategy"])
        if vessel.can_sail == (strategy.name == "drift"):
            raise ValueError(f"{what}: drift craft only drift, and sailing vessels never do")
        if strategy.name in _TARGETED and c["target"] is None:
            raise ValueError(f"{what}: strategy {strategy.name} needs a target")
        departure = _place(places["departures"][c["departure"]], f"{places_path}: departure {c['departure']!r}")
        target = _place(places["targets"].get(c["target"]), f"{places_path}: target {c['target']!r}")
        dest = list(dict.fromkeys(cat.groups_in(*c["destinations"]["regions"]) + c["destinations"]["groups"]))
        if not dest:
            raise ValueError(f"{what}: its destination regions hold no catalog group")
        for group in [departure["group"], *dest]:
            if group not in groups:
                raise ValueError(f"{what}: no group {group!r} in the catalog")
        spec = {"cell": c, "departure": departure, "target": target, "dest_groups": dest,
                "vessel": vessel_entries[c["vessel"]], **shared}
        cells.append(Cell(cell_key(c["name"]), c["name"], vessel, strategy,
                          SimConfig(**cfg["sim"], max_days_at_sea=c["max_days_at_sea"], seed=cfg["seed"]),
                          departure, target, groups.index(departure["group"]), np.isin(groups, dest),
                          json.loads(json.dumps(spec))))
    return cells


# --- running --------------------------------------------------------------------------

def _concat(parts, sim) -> VoyageResults:
    """Join batches run with the same settings into one ``VoyageResults``.

    Tracks are padded with each voyage's final position. A batch's last record
    is taken when its last voyage ended; it is filed under the next multiple of
    ``track_every`` steps, where the positions are the same (every voyage of the
    batch has ended), so all batches share one set of record times and the
    result does not depend on the batching.
    """
    out = {}
    for f in dataclasses.fields(VoyageResults):
        values = [getattr(p, f.name) for p in parts]
        if f.name in ("tracks", "track_days", "meta", "current_is_climatology"):
            continue
        if isinstance(values[0], np.ndarray):
            out[f.name] = np.concatenate(values, axis=-1)
        else:
            if any(v != values[0] for v in values):
                raise ValueError(f"batches disagree on {f.name}")
            out[f.name] = values[0]
    out["tracks"] = out["track_days"] = None
    if parts[0].tracks is not None:
        n_rec = max(p.tracks.shape[1] for p in parts)
        out["tracks"] = np.concatenate(
            [np.concatenate([p.tracks, np.repeat(p.tracks[:, -1:], n_rec - p.tracks.shape[1], axis=1)], axis=1)
             for p in parts])
        out["track_days"] = np.arange(n_rec) * sim.track_every * (sim.dt_hours / geo.HOURS_PER_DAY)
    out["current_is_climatology"] = any(p.current_is_climatology for p in parts)
    out["meta"] = {**parts[0].meta, "n_wind_capped": sum(p.meta["n_wind_capped"] for p in parts)}
    return VoyageResults(**out)


def batch_windows(cells, source, times) -> list[tuple[np.ndarray, float, float]]:
    """Each batch's launch indices and the epoch hours its voyages may need forcing for: from its
    first launch to its last plus the cells' longest wait in port and horizon (and two steps)."""
    reach_h = max((c.strategy.max_wait_days + c.sim.max_days_at_sea) * geo.HOURS_PER_DAY + 2 * c.sim.dt_hours
                  for c in cells)
    return [(idx, times[idx[0]], times[idx[-1]] + reach_h) for idx in source.batches(times)]


def sail(cells, source, times, detector) -> list[VoyageResults]:
    """Every cell's voyages from the launch schedule ``times``, sailed batch by batch; one result per cell."""
    batches = batch_windows(cells, source, times)
    parts = [[] for _ in cells]
    for b, (idx, t0, t1) in enumerate(batches):
        t = times[idx]
        log.info("batch %d/%d: launches %s to %s", b + 1, len(batches), _iso(t[0]), _iso(t[-1]))
        env = None   # let the last batch's fields go before the next are read
        env = source.window(t0, t1)
        for cell, acc in zip(cells, parts):
            acc.append(simulate(env, cell.vessel, cell.strategy, cell.launches(idx, t), detector, cell.sim))
    results = []
    for cell, acc in zip(cells, parts):
        res = _concat(acc, cell.sim)
        res.meta["run"] = cell.spec
        counts = np.bincount(res.status, minlength=len(STATUS_NAMES))
        log.info("cell %s: %d voyages; %s", cell.name, res.status.size,
                 ", ".join(f"{name} {n}" for name, n in zip(STATUS_NAMES, counts) if n))
        results.append(res)
    return results


def _currents(env, launch_t) -> np.ndarray:
    """Per-voyage current source label: synthetic, reanalysis, or climatology (pre-1993 pairing)."""
    if env["type"] == "synthetic":
        return np.full(len(launch_t), "synthetic", dtype=object)
    return np.where(np.asarray(launch_t) < _climatology_before(env), "climatology", "reanalysis").astype(object)


def enso_labels(cfg, base, times=None) -> tuple[pd.Series, pd.Series, pd.Series] | None:
    """ENSO ``(phases, index, source)`` by month from the analysis section's index files; None without one.

    ``analysis.enso_index`` is a CPC seasonal file (RONI or ONI); its months
    are labelled ``cpc``. ``analysis.enso_index_early``, a PSL HadISST Nino 3.4
    anomaly file, labels the months before it (SPEC: the 1940s), re-centred on
    1991-2020 and averaged over three months first; those are ``hadisst``.
    Raises if a file holds no index values, or none for the launch ``times``;
    logs how many launches no index labels.
    """
    analysis = cfg["analysis"]
    if analysis["enso_index"] is None:
        return None

    def read(key, parse):
        path = _resolve(base, analysis[key])
        series = parse(path.read_text(encoding="latin-1")).dropna()
        if series.empty:
            raise ValueError(f"analysis.{key}: no index values in {path}")
        return series

    index = read("enso_index", enso.parse_cpc_seasonal)
    phases = enso.enso_phases(index)
    source = pd.Series("cpc", index=phases.index, dtype=object)
    if analysis["enso_index_early"] is not None:
        early = enso.running_mean_3(enso.recentre(read("enso_index_early", enso.parse_psl_monthly))).dropna()
        early_phases = enso.enso_phases(early)
        before = early_phases.index < phases.index[0]
        phases = pd.concat([early_phases[before], phases])
        source = pd.concat([pd.Series("hadisst", index=early_phases.index[before], dtype=object), source])
        index = pd.concat([early[early.index < index.index[0]], index])
    if times is not None and len(times):
        labelled = enso.label_times(times, phases) != enso.UNKNOWN
        if not labelled.any():
            raise ValueError(f"analysis.enso_index: the index covers {phases.index[0]} to {phases.index[-1]}; "
                             f"the launches run {_iso(times[0])} to {_iso(times[-1])}")
        if not labelled.all():
            log.warning("no ENSO index labels %d of %d launches (from %s); their label is unknown",
                        np.count_nonzero(~labelled), labelled.size, _iso(times[~labelled][0]))
    return phases, index, source


def summary_table(cfg, base, cells, out, enso_months=None) -> pd.DataFrame:
    """Summary rows of every cell whose results in ``out`` are readable and were made with the current
    settings. ``enso_months`` is ``enso_labels``'s result (read from the config if not given)."""
    analysis = cfg["analysis"]
    phases = index = source = None
    if (enso_months := enso_months or enso_labels(cfg, base)) is not None:
        phases, index, source = enso_months
    by = list(analysis["by"])
    if "enso" in by and "enso_source" not in by:
        by.insert(by.index("enso") + 1, "enso_source")   # flags labels from a different index
    frames = []
    for cell in cells:
        path = out / f"{cell.name}.npz"
        if not path.is_file():
            continue
        try:
            results = VoyageResults.load(path)
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as err:
            log.warning("%s is unreadable (%s); rerun its cell to summarize it", path, err)
            continue
        if results.meta.get("run") != cell.spec:
            log.warning("%s was made with other settings; rerun its cell to summarize it", path)
            continue
        labels = analyze.launch_labels(results, phases, index, source)
        labels["currents"] = _currents(cfg["environment"], results.launch_t)
        supplies = [s for s in analysis["supplies"] if s <= cell.sim.max_days_at_sea]
        df = analyze.summarize(results, labels, results.rule_names, supplies, by=by,
                               n_boot=analysis["n_boot"], seed=cfg["seed"])
        lead = (cell.name, cell.departure["name"], cell.target["name"] if cell.target else None,
                cell.vessel.name, cell.strategy.name)
        for k, (column, value) in enumerate(zip(CELL_COLUMNS, lead)):
            df.insert(k, column, value)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run(config_path, out_dir=None, cells=None) -> Path:
    """Run the cells of a grid config whose names match the glob ``cells`` (all by default).

    Writes ``<cell>.npz``, ``summary.csv`` and ``resolved_config.yaml`` to
    ``out_dir`` (default ``outputs/<name>`` under the working directory) and
    returns that directory.
    """
    config_path = Path(config_path)
    base = config_path.parent
    cfg = load_config(config_path)
    out = Path("outputs", cfg["name"]) if out_dir is None else Path(out_dir)
    source = make_source(cfg["environment"], base)

    cat_cfg = cfg["catalog"]
    cache_dir = None if cat_cfg["cache_dir"] is None else _resolve(base, cat_cfg["cache_dir"])
    gshhg_dir = None
    if cat_cfg["gshhg_dir"] is not None:
        gshhg_dir = _resolve(base, cat_cfg["gshhg_dir"])
        if not (gshhg_dir / catalog.GSHHG_SHAPEFILE).is_file():
            # The library falls back to the placeholder catalog; a grid asked for real shorelines.
            raise ValueError(f"catalog.gshhg_dir: no GSHHG shapefile at {gshhg_dir / catalog.GSHHG_SHAPEFILE} "
                             "(voyage-sim fetch gshhg); set gshhg_dir: null for the placeholder catalog")
    cat = catalog.build_catalog(_resolve(base, cat_cfg["config"]), gshhg_dir, cache_dir)
    digest = landmass_digest(cat.landmasses)
    groups = list(dict.fromkeys(lm.group for lm in cat.landmasses))   # numbered as the detector numbers them
    every = resolve_cells(cfg, base, cat, groups, digest)
    chosen = [c for c in every if cells is None or fnmatch.fnmatchcase(c.name, cells)]
    if not chosen:
        raise ValueError(f"no cell name matches {cells!r}; cells are {[c.name for c in every]}")
    times = launch_times(cfg["launches"])
    # Everything the run needs is checked before the slow steps.
    for _, t0, t1 in batch_windows(chosen, source, times):
        source.check(t0, t1)
    enso_months = enso_labels(cfg, base, times)
    _writable(out)

    rasters = [_raster(cat.landmasses, SightingRule(**rule), cfg["raster"], cache_dir, digest) for rule in cfg["rules"]]
    detector = LandfallDetector(cat.landmasses, rasters, substep_km=cfg["raster"]["substep_km"])
    results = sail(chosen, source, times, detector)
    for cell, res in zip(chosen, results):
        res.save(out / f"{cell.name}.npz")
    _write_text(out / "resolved_config.yaml",
                yaml.safe_dump({**cfg, "version": __version__}, sort_keys=True, allow_unicode=True))
    summary = summary_table(cfg, base, every, out, enso_months)
    _write_text(out / "summary.csv", summary.to_csv(index=False, float_format="%.6g", lineterminator="\n"))
    return out


def _writable(out):
    """Create the output directory and check that files can be written in it."""
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / f".write-test.{os.getpid()}"
        probe.touch()
        probe.unlink()
    except OSError as err:
        raise ValueError(f"cannot write outputs to {out}: {err}") from err


def _write_text(path, text):
    """Write a text file under a temporary name, then rename it into place."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# --- command line ---------------------------------------------------------------------

def _ints(text) -> list[int]:
    """'2010-2012,2015' -> [2010, 2011, 2012, 2015]."""
    values = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        try:
            lo, hi = int(lo), int(hi or lo)
        except ValueError:
            raise ValueError(f"bad number or range {part!r} in {text!r}") from None
        if hi < lo:
            raise ValueError(f"bad range {part!r}: it ends before it starts")
        values += range(lo, hi + 1)
    return values


def _fetch(args):
    if args.source == "era5":
        plan = fetch.fetch_era5(_ints(args.years), _ints(args.months), args.out, hours=_ints(args.hours),
                                dry_run=args.dry_run)
    elif args.source == "glorys":
        plan = fetch.fetch_glorys(args.start, args.end, args.out, dry_run=args.dry_run)
    elif args.source == "gshhg":
        plan = fetch.fetch_gshhg(args.out, dry_run=args.dry_run)
    else:
        plan = fetch.fetch_enso_indices(args.out, dry_run=args.dry_run)
    verb = "would fetch" if args.dry_run else "fetched"
    for item in plan:
        print(f"{verb} {item['target']}")
    if not plan:
        print(f"nothing to fetch into {args.out}")


def main(argv=None) -> int:
    """``voyage-sim run CONFIG [--out DIR] [--cells GLOB]`` and ``voyage-sim fetch {era5,glorys,gshhg,enso}``."""
    parser = argparse.ArgumentParser(prog="voyage-sim", description="Simulated voyages between eastern Polynesia "
                                     "and South America.")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run", help="run a scenario grid config")
    p.add_argument("config", type=Path)
    p.add_argument("--out", type=Path, help="output directory (default: outputs/<name>)")
    p.add_argument("--cells", metavar="GLOB", help="run only the cells whose names match (default: all)")

    p = commands.add_parser("fetch", help="download input data (see voyage_sim.fetch)")
    sources = p.add_subparsers(dest="source", required=True)
    s = sources.add_parser("era5", help="ERA5 10 m winds, one file per month")
    s.add_argument("--years", required=True, help="e.g. 2010-2019 or 1947,1976")
    s.add_argument("--months", default="1-12")
    s.add_argument("--hours", default="0,6,12,18", help="UTC hours (default 6-hourly)")
    s.add_argument("--out", type=Path, default=Path("data/era5"))
    s = sources.add_parser("glorys", help="GLORYS12 daily surface currents, one file per year")
    s.add_argument("--start", required=True, help="first day, e.g. 2010-01-01")
    s.add_argument("--end", required=True, help="last day, e.g. 2019-12-31")
    s.add_argument("--out", type=Path, default=Path("data/glorys"))
    s = sources.add_parser("gshhg", help="GSHHG 2.3.7 full-resolution shorelines")
    s.add_argument("--out", type=Path, default=Path("data/gshhg"))
    s = sources.add_parser("enso", help="RONI, ONI and HadISST Nino 3.4 index files")
    s.add_argument("--out", type=Path, default=Path("data/enso"))
    for s in sources.choices.values():
        s.add_argument("--dry-run", action="store_true", help="list the downloads without making them")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "run":
            print(run(args.config, args.out, args.cells))
        else:
            _fetch(args)
    except (ValueError, OSError) as err:   # bad settings or files: a message, not a traceback
        print(f"voyage-sim: error: {err}", file=sys.stderr)
        return 1
    return 0
