import gc
import os
import subprocess
import sys
import tracemalloc
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

from voyage_sim import catalog, fetch, geo, run
from voyage_sim.fields import Environment, FunctionField, UniformField
from voyage_sim.simulate import END_HORIZON, VoyageResults

ROOT = Path(__file__).resolve().parents[1]

# A small world: one big high island 1,300 km west of a straight coast, under
# the pacific_trades trades and South Equatorial Current, both setting west.
WORLD = """
domain: [-100.0, -76.0, -10.0, 8.0]
pacific_mask: [[-101.0, -11.0], [-77.0, -11.0], [-77.0, 9.0], [-101.0, 9.0]]
groups:
  - name: Isla Grande
    region: polynesia
    height_m: 50
    boxes: [[-96.0, -90.0, -4.0, 4.0]]
    islands:
      - {name: Isla Grande, ref: [-93.0, 0.0], height_m: 500, radius_km: 100.0}
coast:
  segments:
    - {name: "Coast: East", region: americas, north: 8.0, south: -10.0, height_m: 300}
  placeholder:
    - [[-80.0, 8.0], [-80.0, -10.0]]
"""
PLACES = """
departures:
  - {name: Isla Grande, lon: -91.9, lat: 0.0, group: Isla Grande}
  - {name: Puerto, lon: -80.3, lat: -1.0, group: "Coast: East"}
targets:
  - {name: Isla Grande, lon: -91.9, lat: 0.0, group: Isla Grande}
  - {name: Puerto, lon: -80.3, lat: -1.0, group: "Coast: East"}
"""
CONFIG = {
    "name": "tiny",
    "seed": 7,
    "environment": {"type": "synthetic", "field": "pacific_trades", "params": {"seasonal": True}},
    "catalog": {"config": "world.yaml", "cache_dir": "cache"},
    "rules": [{"name": "main", "mode": "height", "bird_km": 37},
              {"name": "uniform40", "mode": "uniform", "radius_km": 40}],
    "raster": {"bounds": [-100, -76, -10, 8], "res_deg": 0.1, "cap_km": 100},
    "vessels": str(ROOT / "configs" / "vessels.yaml"),   # an absolute path
    "places": "places.yaml",
    # June and July: two July-June years, so the bootstrap has two clusters.
    "launches": {"start": "2001-06-01", "end": "2001-07-31", "every_days": 10, "per_day": 2},
    "sim": {"dt_hours": 12},
    "analysis": {"supplies": [20, 40, 60], "by": [], "n_boot": 200},
    "cells": [
        {"name": "island_canoe_hold", "departure": "Isla Grande", "target": "Puerto", "vessel": "canoe_75",
         "strategy": "hold_course", "destinations": {"regions": ["americas"]}, "max_days_at_sea": 40},
        {"name": "island_raft_drift", "departure": "Isla Grande", "vessel": "drift_raft", "strategy": "drift",
         "destinations": {"regions": ["americas"]}, "max_days_at_sea": 40},
        {"name": "puerto_canoe_hold", "departure": "Puerto", "target": "Isla Grande", "vessel": "canoe_75",
         "strategy": {"name": "hold_course", "tack_hysteresis_deg": 15},
         "destinations": {"regions": ["polynesia"]}, "max_days_at_sea": 40},
        {"name": "puerto_raft_drift", "departure": "Puerto", "vessel": "drift_raft", "strategy": "drift",
         "destinations": {"groups": ["Isla Grande"]}, "max_days_at_sea": 40},
    ],
}
N_LAUNCHES = 14   # 7 launch days x 2


def write_world(directory, config):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "world.yaml").write_text(WORLD)
    (directory / "places.yaml").write_text(PLACES)
    path = directory / f"{config['name']}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def files(out):
    return {p.name: p.read_bytes() for p in sorted(out.iterdir())}


def copy_outputs(out, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in files(out).items():
        (directory / name).write_bytes(data)


def shared_cache(tiny_config):
    """The tiny grid's catalog settings with its cache, so other runs reuse its sighting rasters."""
    return {**CONFIG["catalog"], "cache_dir": str(tiny_config.parent / "cache")}


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """The tiny grid run once through the command line: (config path, output directory)."""
    base = tmp_path_factory.mktemp("tiny")
    path = write_world(base, CONFIG)
    assert run.main(["run", str(path), "--out", str(base / "out")]) == 0
    return path, base / "out"


# --- synthetic field ----------------------------------------------------------------

def test_pacific_trades_is_plausible():
    wind, current = run.pacific_trades(seasonal=False)
    lon = np.array([-120.0, -120.0, -120.0, -120.0, -100.0, -76.0, -170.0])
    lat = np.array([-15.0, 12.0, -45.0, -30.0, -1.0, -15.0, -15.0])
    u, v = wind.sample(lon, lat, np.zeros(lon.size))
    speed = np.hypot(u, v)
    assert u[0] < 0 and v[0] > 0 and 6.0 < speed[0] < 7.5     # southeast trades
    assert u[1] < 0 and v[1] < 0 and 6.0 < speed[1] < 7.5     # northeast trades
    assert u[2] == pytest.approx(8.0, abs=0.3)                 # westerlies
    assert speed[3] < 2.0                                      # subtropical high
    cu, cv = current.sample(lon, lat, np.zeros(lon.size))
    assert cu[4] == pytest.approx(-0.3, abs=0.01)              # South Equatorial Current
    assert cv[5] > 0.1 and abs(cv[0]) < 1e-3                   # Humboldt, at the coast only
    assert np.isnan(u[6]) and np.isnan(cu[6])                  # outside the study domain

    # The belts follow the sun: in February the trades reach further south and the ITCZ lies
    # south of 5 N; in August it lies north of 5 N, so the southeast trades blow there.
    winds = run.pacific_trades(seasonal=True)[0]
    t = geo.to_epoch_hours(["2001-02-14", "2001-08-15", "2001-02-14", "2001-08-15"])
    u, v = winds.sample(np.full(4, -120.0), np.array([-24.0, -24.0, 5.0, 5.0]), t)
    assert u[0] < u[1] - 2.0 and v[2] < 0 < v[3]


# --- a synthetic grid end to end --------------------------------------------------

def test_tiny_grid_summary_is_sane(tiny):
    _, out = tiny
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"{c['name']}.npz" for c in CONFIG["cells"]] + ["summary.csv", "resolved_config.yaml"])
    df = pd.read_csv(out / "summary.csv")
    assert list(df.columns[:5]) == list(run.CELL_COLUMNS)
    assert (df["p"].between(0, 1)).all() and (df["n"] == N_LAUNCHES).all() and (df["n_years"] == 2).all()
    defined = df["lo"].notna()
    assert defined.any() and (df["lo"][defined] <= df["p"][defined]).all()
    assert (df["p"][defined] <= df["hi"][defined]).all()
    assert sorted(df["supply_days"].unique()) == [20, 40]   # 60 is beyond the horizon

    p = df[(df.rule == "main") & (df.supply_days == 40)].set_index("cell")["p"]
    assert p["puerto_raft_drift"] > 0.2 and p["island_raft_drift"] == 0   # drifting downwind, not upwind
    assert p["puerto_canoe_hold"] == 1 and p["puerto_canoe_hold"] > p["island_canoe_hold"]
    resolved = yaml.safe_load((out / "resolved_config.yaml").read_text())
    assert resolved["sim"]["diffusivity_m2s"] == 30.0 and resolved["version"]


def test_voyage_ids_are_unique_across_cells(tiny):
    _, out = tiny
    ids = []
    for cell in CONFIG["cells"]:
        res = VoyageResults.load(out / f"{cell['name']}.npz")
        np.testing.assert_array_equal(res.voyage_id, run.cell_key(cell["name"]) * run.ID_STRIDE + np.arange(N_LAUNCHES))
        assert res.meta["run"]["cell"]["name"] == cell["name"]
        ids.append(res.voyage_id)
    assert np.unique(np.concatenate(ids)).size == len(CONFIG["cells"]) * N_LAUNCHES
    assert (2**32 - 1) * run.ID_STRIDE + run.ID_STRIDE - 1 < np.iinfo(np.int64).max


def test_a_cell_sails_the_same_voyages_wherever_it_sits_in_the_grid(tiny, tmp_path):
    # Its voyages depend on the cell, not on the cells before it: reordered, or alone in a config of its
    # own, it gives the same file, so a grid can grow and a cell can be rerun on its own.
    path, out = tiny
    for name, cells in (("reordered", CONFIG["cells"][::-1]), ("alone", CONFIG["cells"][2:3])):
        config = write_world(tmp_path / name, {**CONFIG, "catalog": shared_cache(path), "cells": cells})
        result = run.run(config, tmp_path / name / "out", cells="puerto_canoe_hold") / "puerto_canoe_hold.npz"
        assert result.read_bytes() == (out / "puerto_canoe_hold.npz").read_bytes()


def test_cells_whose_names_share_an_id_key_are_refused(tiny, monkeypatch):
    path, _ = tiny
    monkeypatch.setattr(run, "cell_key", lambda name: 7)
    with pytest.raises(ValueError, match="same id key"):
        run.run(path, path.parent / "never")
    assert not (path.parent / "never").exists()


def test_rerun_is_byte_identical(tiny, tmp_path):
    path, out = tiny
    assert run.main(["run", str(path), "--out", str(tmp_path)]) == 0
    assert files(tmp_path) == files(out)


def test_results_do_not_depend_on_batching(tiny, tmp_path, monkeypatch):
    path, out = tiny
    monkeypatch.setattr(run.SyntheticSource, "batches", lambda self, times: np.array_split(np.arange(len(times)), 2))
    run.run(path, tmp_path)
    assert files(tmp_path) == files(out)


class Recorder:
    """A field that notes the times it is sampled at."""

    def __init__(self, field):
        self.field, self.times = field, []

    def sample(self, lon, lat, t):
        self.times.append(np.ravel(t))
        return self.field.sample(lon, lat, t)


def test_batch_windows_cover_the_longest_wait_and_horizon(tiny, tmp_path, monkeypatch):
    """A canoe waits in port as long as it may, then sails to a horizon that is not a whole number of
    steps: every wind and current it reads lies inside its batch's window, which spares at most two steps."""
    path, _ = tiny
    west_wind = geo.to_epoch_hours("2001-06-03")

    def westerly_later():
        wind = FunctionField(lambda lon, lat, t: (np.where(t < west_wind, -6.0, 6.0), np.zeros_like(t)))
        return wind, UniformField(0.0, 0.0)

    monkeypatch.setitem(run.SYNTHETIC_FIELDS, "westerly_later", westerly_later)
    windows = []

    def window(self, t0, t1):
        env = Environment(Recorder(self.env.wind), Recorder(self.env.current))
        windows.append((t0, t1, env))
        return env

    monkeypatch.setattr(run.SyntheticSource, "window", window)
    monkeypatch.setattr(run.SyntheticSource, "batches", lambda self, times: np.split(np.arange(len(times)), len(times)))
    cell = {"name": "wait", "departure": "Isla Grande", "target": "Puerto", "vessel": "canoe_75",
            "strategy": {"name": "wait_for_wind", "max_wait_days": 2}, "destinations": {"regions": ["americas"]},
            "max_days_at_sea": 3.1}
    config = {**CONFIG, "environment": {"type": "synthetic", "field": "westerly_later"},
              "catalog": shared_cache(path), "launches": {"start": "2001-06-01", "end": "2001-06-02"},
              "analysis": {"supplies": [3], "by": []}, "cells": [cell]}
    res = VoyageResults.load(run.run(write_world(tmp_path, config), tmp_path / "out") / "wait.npz")
    np.testing.assert_array_equal(res.depart_t - res.launch_t, [48.0, 24.0])   # both leave with the west wind
    assert (res.status == END_HORIZON).all()

    dt = CONFIG["sim"]["dt_hours"]
    assert len(windows) == 2
    for t0, t1, env in windows:
        t = np.concatenate(env.wind.times + env.current.times)
        assert t0 <= t.min() and t.max() <= t1
    t0, t1, env = windows[0]
    assert t1 - np.concatenate(env.current.times).max() <= 2 * dt


def test_batch_windows_cover_the_cell_that_reaches_furthest(tiny, tmp_path, monkeypatch):
    """The window is sized by the cell that sails longest, wherever it comes in the grid."""
    path, _ = tiny
    windows = []

    def window(self, t0, t1):
        env = Environment(Recorder(self.env.wind), Recorder(self.env.current))
        windows.append((t0, t1, env))
        return env

    monkeypatch.setattr(run.SyntheticSource, "window", window)
    monkeypatch.setattr(run.SyntheticSource, "batches", lambda self, times: np.split(np.arange(len(times)), len(times)))
    short = {**CONFIG["cells"][0], "name": "short_hold", "max_days_at_sea": 2}
    long = {**CONFIG["cells"][3], "name": "long_drift", "max_days_at_sea": 6}
    config = {**CONFIG, "catalog": shared_cache(path), "launches": {"start": "2001-06-01", "end": "2001-06-02"},
              "analysis": {"supplies": [2], "by": []}, "cells": [short, long]}
    out = run.run(write_world(tmp_path, config), tmp_path / "out")
    assert len(windows) == 2
    for t0, t1, env in windows:
        t = np.concatenate(env.wind.times + env.current.times)
        assert t0 <= t.min() and t.max() <= t1
    res = VoyageResults.load(out / "long_drift.npz")
    assert (res.status == END_HORIZON).all() and (res.days_at_sea == 6.0).all()


def test_cells_glob_reruns_some_cells_and_keeps_the_summary(tiny, tmp_path):
    path, out = tiny
    copy_outputs(out, tmp_path)
    (tmp_path / "puerto_raft_drift.npz").unlink()
    assert run.main(["run", str(path), "--out", str(tmp_path), "--cells", "puerto_*"]) == 0
    assert files(tmp_path) == files(out)


def test_results_from_other_settings_are_left_out_of_the_summary(tiny, tmp_path, caplog):
    path, out = tiny
    copy_outputs(out, tmp_path)
    changed = write_world(tmp_path / "changed", {**CONFIG, "seed": 8, "catalog": shared_cache(path)})
    run.run(changed, tmp_path, cells="island_raft_drift")
    assert set(pd.read_csv(tmp_path / "summary.csv")["cell"]) == {"island_raft_drift"}
    assert "puerto_canoe_hold.npz was made with other settings" in caplog.text


def test_a_rule_changed_under_the_same_name_gets_its_own_raster(tiny, tmp_path):
    # A sensitivity rerun (the main rule at half range) must not reuse the cached raster of the old rule.
    path, out = tiny
    cache = tmp_path / "cache"
    cache.mkdir()
    for cached in (path.parent / "cache").iterdir():
        (cache / cached.name).write_bytes(cached.read_bytes())
    rules = [{**CONFIG["rules"][0], "scale": 0.5}, CONFIG["rules"][1]]
    config = write_world(tmp_path, {**CONFIG, "catalog": {**CONFIG["catalog"], "cache_dir": str(cache)},
                                    "rules": rules})
    run.run(config, tmp_path / "out", cells="puerto_raft_drift")
    assert len(list(cache.glob("raster-main-*.npz"))) == 2 and len(list(cache.glob("raster-uniform40-*.npz"))) == 1
    old, new = (VoyageResults.load(d / "puerto_raft_drift.npz") for d in (out, tmp_path / "out"))
    both = np.isfinite(old.hit_days[0]) & np.isfinite(new.hit_days[0])
    assert both.any() and np.all(new.hit_days[0, both] > old.hit_days[0, both])
    np.testing.assert_array_equal(new.hit_days[1], old.hit_days[1])


def test_results_on_other_land_are_left_out_of_the_summary(tiny, tmp_path, caplog):
    # The land is part of a result's settings: after a height in the catalog changes, cells made on the
    # old land are not summarized with those made on the new.
    path, out = tiny
    copy_outputs(out, tmp_path / "out")
    changed = write_world(tmp_path / "changed", {**CONFIG, "catalog": shared_cache(path)})
    world = tmp_path / "changed" / "world.yaml"
    world.write_text(WORLD.replace("height_m: 500", "height_m: 300"))
    run.run(changed, tmp_path / "out", cells="island_raft_drift")
    assert set(pd.read_csv(tmp_path / "out" / "summary.csv")["cell"]) == {"island_raft_drift"}
    assert "puerto_canoe_hold.npz was made with other settings" in caplog.text
    # Back on the old land, the old results stand again (the config records the moved cache).
    world.write_text(WORLD)
    run.run(changed, tmp_path / "out", cells="island_raft_drift")
    again, before = files(tmp_path / "out"), files(out)
    assert again.pop("resolved_config.yaml") != before.pop("resolved_config.yaml") and again == before


def test_a_configured_gshhg_dir_without_shorelines_is_an_error(tiny, tmp_path, capsys):
    path, _ = tiny
    (tmp_path / "gshhg").mkdir()
    config = write_world(tmp_path, {**CONFIG, "catalog": {**shared_cache(path), "gshhg_dir": "gshhg"}})
    assert run.main(["run", str(config), "--out", str(tmp_path / "out")]) == 1
    assert "catalog.gshhg_dir: no GSHHG shapefile" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_unreadable_results_are_left_out_of_the_summary(tiny, tmp_path, caplog):
    # A cell's file cut short (a run killed while saving it) is reported, not a crash of later runs.
    path, out = tiny
    copy_outputs(out, tmp_path)
    whole = (tmp_path / "puerto_raft_drift.npz").read_bytes()
    (tmp_path / "puerto_raft_drift.npz").write_bytes(whole[:2000])
    assert run.main(["run", str(path), "--out", str(tmp_path), "--cells", "island_*"]) == 0
    assert "puerto_raft_drift.npz is unreadable" in caplog.text
    assert "puerto_raft_drift" not in set(pd.read_csv(tmp_path / "summary.csv")["cell"])
    assert not list(tmp_path.glob("*.tmp"))


def test_an_unwritable_output_or_a_bad_index_file_stops_the_run_before_it_sails(tiny, tmp_path, capsys, monkeypatch):
    path, _ = tiny

    def refuse(*args, **kwargs):
        raise AssertionError("sailed")

    monkeypatch.setattr(run, "sail", refuse)
    (tmp_path / "a_file").write_text("")
    for out in (tmp_path / "a_file", tmp_path / "a_file" / "sub"):
        assert run.main(["run", str(path), "--out", str(out)]) == 1
        assert f"cannot write outputs to {out}" in capsys.readouterr().err

    analysis = {"supplies": [20], "by": ["enso"], "n_boot": 50, "enso_index": "index.txt"}
    config = write_world(tmp_path / "enso", {**CONFIG, "catalog": shared_cache(path), "analysis": analysis})
    for text, message in ((None, "No such file"), ("garbage\n<html>404</html>\n", "no index values in"),
                          ("SEAS YR ANOM\nDJF 1970 1.2\nJFM 1970 1.1\n", "the index covers 1970-01 to 1970-02")):
        if text is not None:
            (tmp_path / "enso" / "index.txt").write_text(text)
        assert run.main(["run", str(config), "--out", str(tmp_path / "enso" / "out")]) == 1
        assert message in capsys.readouterr().err
        assert not (tmp_path / "enso" / "out").exists()


SEASONS = ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")


def test_enso_labels_come_from_the_index_file(tiny, tmp_path):
    path, out = tiny
    copy_outputs(out, tmp_path / "out")
    # A RONI-format file with an El Nino from AMJ (centred on May) 2001 to the end of the year.
    roni = ["SEAS  YR   ANOM"] + [f"{season} {year} {1.2 if year == 2001 and k >= 4 else 0.0:.2f}"
                                  for year in (2000, 2001, 2002) for k, season in enumerate(SEASONS)]
    (tmp_path / "enso").mkdir()
    (tmp_path / "enso" / "roni.txt").write_text("\n".join(roni) + "\n")
    analysis = {"supplies": [20], "by": ["enso"], "n_boot": 50, "enso_index": "roni.txt"}
    config = write_world(tmp_path / "enso", {**CONFIG, "catalog": shared_cache(path), "analysis": analysis})
    run.run(config, tmp_path / "out", cells="puerto_canoe_hold")
    df = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert set(df["cell"]) == {c["name"] for c in CONFIG["cells"]}   # the cache moved, the results stand
    assert set(df["enso"]) == {"el_nino"} and (df["n_events"] == 1).all()
    assert set(df["enso_source"]) == {"cpc"}
    # One event, but June and July launches fall in different July-June years: two clusters.
    assert (df["n_years"] == 2).all()


def test_the_1940s_take_their_enso_labels_from_hadisst(tiny, tmp_path):
    # SPEC: before RONI starts (1950), HadISST Nino 3.4 re-centred on 1991-2020 labels the launches,
    # flagged as a different index. Here an El Nino from 1946 to 1948 in HadISST, neutral after.
    path, _ = tiny
    rows = [f"{year} " + " ".join("1.50" if 1946 <= year <= 1948 else "0.00" for _ in range(12))
            for year in range(1940, 2021)]
    (tmp_path / "hadisst.data").write_text("\n".join(["1940 2020", *rows, "-99.99", "HadISST1 Nino 3.4"]) + "\n")
    roni = ["SEAS  YR   ANOM"] + [f"{season} {year} 0.00" for year in range(1950, 2021) for season in SEASONS]
    (tmp_path / "roni.txt").write_text("\n".join(roni) + "\n")
    analysis = {"supplies": [20], "by": ["enso"], "n_boot": 50, "enso_index": "roni.txt",
                "enso_index_early": "hadisst.data"}
    launches = {"start": "1947-04-28", "end": "1950-12-31", "every_days": 90}
    config = write_world(tmp_path, {**CONFIG, "catalog": shared_cache(path), "analysis": analysis,
                                    "launches": launches, "cells": CONFIG["cells"][3:]})
    df = pd.read_csv(run.run(config, tmp_path / "out") / "summary.csv")
    n = {(e, s): n for e, s, n in zip(df["enso"], df["enso_source"], df["n"])}
    # The three-month mean carries the event into January 1949.
    month = geo.from_epoch_hours(run.launch_times(run.load_config(config)["launches"])).astype("datetime64[M]")
    jan_1949, jan_1950 = np.datetime64("1949-01"), np.datetime64("1950-01")
    assert n == {("el_nino", "hadisst"): np.count_nonzero(month <= jan_1949),
                 ("neutral", "hadisst"): np.count_nonzero((month > jan_1949) & (month < jan_1950)),
                 ("neutral", "cpc"): np.count_nonzero(month >= jan_1950)}


# --- files: ERA5- and GLORYS-like NetCDF ------------------------------------------

LON = np.arange(-100.0, -75.5, 1.0)
LAT_DOWN = np.arange(8.0, -10.5, -1.0)   # ERA5 runs north to south


def era5_month(directory, month, u=-6.0, gust=0.0):
    """An ERA5-like file of 6-hourly 10 m winds for one month (``"1992-12"``); ``gust`` m/s varies u
    over a 5-day period."""
    start = np.datetime64(month, "M")
    end = start + np.timedelta64(1, "M")
    t = np.arange(start.astype("datetime64[h]"), end.astype("datetime64[h]"), np.timedelta64(6, "h"))
    shape = (t.size, LAT_DOWN.size, LON.size)
    dims = ("valid_time", "latitude", "longitude")
    u_t = u + gust * np.sin(2 * np.pi * geo.to_epoch_hours(t) / 120.0)
    ds = xr.Dataset({"u10": (dims, np.broadcast_to(u_t[:, None, None], shape).astype(np.float32)),
                     "v10": (dims, np.zeros(shape, np.float32))},
                    coords={"valid_time": t.astype("datetime64[ns]"), "latitude": LAT_DOWN, "longitude": LON,
                            "number": 0, "expver": ("valid_time", np.full(t.size, "0001"))})
    directory.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(directory / f"era5_uv10_{month}.nc")


def glorys_days(path, first, last, u=-0.2):
    """A GLORYS-like file of daily surface currents with a length-1 depth axis and a land column (NaN)."""
    t = np.arange(np.datetime64(first), np.datetime64(last) + np.timedelta64(1, "D")).astype("datetime64[ns]")
    lat = LAT_DOWN[::-1]
    field = np.full((t.size, 1, lat.size, LON.size), u, np.float32)
    field[..., -1] = np.nan
    dims = ("time", "depth", "latitude", "longitude")
    ds = xr.Dataset({"uo": (dims, field), "vo": (dims, np.where(np.isnan(field), np.nan, 0.0).astype(np.float32))},
                    coords={"time": t, "depth": [0.494], "latitude": lat, "longitude": LON})
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def climatology(path, u=-0.1):
    shape = (12, LAT_DOWN.size, LON.size)
    dims = ("month", "latitude", "longitude")
    xr.Dataset({"uo": (dims, np.full(shape, u, np.float32)), "vo": (dims, np.zeros(shape, np.float32))},
               coords={"month": np.arange(1, 13), "latitude": LAT_DOWN, "longitude": LON}).to_netcdf(path)


FILES_CONFIG = {
    **CONFIG,
    "name": "files",
    "environment": {"type": "files", "wind": {"paths": "era5/era5_uv10_*.nc"},
                    "current": {"paths": "glorys/glorys_uv_*.nc"},
                    "climatology": {"path": "glorys/climatology.nc"}, "climatology_before": "1993-01-01"},
    "launches": {"start": "1992-12-25", "end": "1993-01-08", "every_days": 7},
    "rules": CONFIG["rules"][:1],
    "analysis": {"supplies": [20], "by": ["month", "currents"], "n_boot": 50},
    "cells": [{"name": "puerto_raft_drift", "departure": "Puerto", "vessel": "drift_raft", "strategy": "drift",
               "destinations": {"regions": ["polynesia"]}, "max_days_at_sea": 20}],
}


@pytest.fixture
def data(tmp_path):
    for month in ("1992-12", "1993-01"):
        era5_month(tmp_path / "era5", month)
    glorys_days(tmp_path / "glorys" / "glorys_uv_1993-01-01_1993-01-31.nc", "1993-01-01", "1993-01-31")
    climatology(tmp_path / "glorys" / "climatology.nc")
    return write_world(tmp_path, FILES_CONFIG)


def test_file_windows_pair_early_launches_with_the_climatology(data):
    cfg = run.load_config(data)
    source = run.make_source(cfg["environment"], data.parent)
    times = run.launch_times(cfg["launches"])
    assert [b.tolist() for b in source.batches(times)] == [[0], [1, 2]]
    at = np.array([-90.0]), np.array([-1.0])

    early = source.window(times[0], times[0] + 20 * 24)
    assert early.current_is_climatology
    assert early.current.sample(*at, times[:1] + 24)[0] == pytest.approx(-0.1)
    new_year = geo.to_epoch_hours(["1992-12-31T21"])   # between the last December and first January winds
    assert early.wind.sample(*at, new_year)[0] == pytest.approx(-6.0)
    late = source.window(times[1], times[2] + 20 * 24)
    assert not late.current_is_climatology
    assert late.current.sample(*at, times[1:2] + 24)[0] == pytest.approx(-0.2)
    assert late.current.sample(np.array([-76.2]), at[1], times[1:2] + 24)[0] == pytest.approx(-0.2)  # land dropped

    with pytest.raises(ValueError, match="wind files cover"):
        source.check(times[1], times[1] + 40 * 24)
    with pytest.raises(ValueError, match="current files cover"):   # daily currents end before the winds
        source.check(times[1], geo.to_epoch_hours("1993-01-31T12"))


def test_launches_after_a_mid_month_climatology_cutoff_sail_on_the_reanalysis(tiny, tmp_path):
    # climatology_before may fall mid-month: the batch splits there, so later launches never sail on the
    # climatology (whose current is 0.1 m/s weaker) while labelled reanalysis.
    path, _ = tiny
    for month in ("1993-01", "1993-02"):
        era5_month(tmp_path / "era5", month)
    glorys_days(tmp_path / "glorys" / "glorys_uv_1993.nc", "1993-01-01", "1993-02-28")
    climatology(tmp_path / "glorys" / "climatology.nc")
    environment = {**FILES_CONFIG["environment"], "climatology_before": "1993-01-15"}
    launches = {"start": "1993-01-01", "end": "1993-01-29", "every_days": 7}
    config = write_world(tmp_path, {**FILES_CONFIG, "environment": environment, "launches": launches,
                                    "catalog": shared_cache(path)})
    cfg = run.load_config(config)
    times = run.launch_times(cfg["launches"])
    assert [b.tolist() for b in run.make_source(cfg["environment"], tmp_path).batches(times)] == [[0, 1], [2, 3, 4]]
    mixed = VoyageResults.load(run.run(config, tmp_path / "mixed") / "puerto_raft_drift.npz")
    no_clim = {key: value for key, value in environment.items() if key != "climatology"}
    config = write_world(tmp_path, {**FILES_CONFIG, "environment": no_clim, "launches": launches,
                                    "catalog": shared_cache(path)})
    reanalysis = VoyageResults.load(run.run(config, tmp_path / "reanalysis") / "puerto_raft_drift.npz")
    np.testing.assert_array_equal(mixed.end_lon[2:], reanalysis.end_lon[2:])
    assert np.all(mixed.end_lon[:2] - reanalysis.end_lon[:2] > 1.0)
    df = pd.read_csv(tmp_path / "mixed" / "summary.csv")
    assert set(zip(df["currents"], df["n"])) == {("climatology", 2), ("reanalysis", 3)}


def test_a_missing_file_is_found_before_sailing(data):
    era5_month(data.parent / "era5", "1993-02")
    (data.parent / "era5" / "era5_uv10_1993-01.nc").unlink()
    source = run.make_source(run.load_config(data)["environment"], data.parent)
    t = geo.to_epoch_hours("1992-12-25")
    with pytest.raises(ValueError, match="is a file missing"):
        source.check(t, t + 60 * 24)


def test_files_grid_runs_and_flags_climatology_launches(data, tmp_path):
    out = run.run(data, tmp_path / "out")
    res = VoyageResults.load(out / "puerto_raft_drift.npz")
    assert res.current_is_climatology
    # The climatology's current is 0.1 m/s weaker: 20 days of it leave the raft ~170 km further east.
    assert res.end_lon[0] - res.end_lon[1:].max() > 1.0
    df = pd.read_csv(out / "summary.csv")
    assert set(zip(df["month"], df["currents"], df["n"])) == {(1, "reanalysis", 2), (12, "climatology", 1)}


def test_monthly_file_windows_give_the_results_of_one_batch(tiny, tmp_path, monkeypatch):
    """Time-varying winds read a launch month at a time sail every voyage as one window over all launches
    does; without a climatology every launch is labelled reanalysis."""
    path, _ = tiny
    for month in ("1993-01", "1993-02"):
        era5_month(tmp_path / "era5", month, gust=2.0)
    glorys_days(tmp_path / "glorys" / "glorys_uv_1993-01-01_1993-02-28.nc", "1993-01-01", "1993-02-28")
    environment = {key: value for key, value in FILES_CONFIG["environment"].items() if key != "climatology"}
    config = write_world(tmp_path, {**FILES_CONFIG, "environment": environment, "catalog": shared_cache(path),
                                    "launches": {"start": "1993-01-01", "end": "1993-02-05", "every_days": 7}})
    cfg = run.load_config(config)
    times = run.launch_times(cfg["launches"])
    assert [b.size for b in run.make_source(cfg["environment"], tmp_path).batches(times)] == [5, 1]

    by_month = run.run(config, tmp_path / "by_month")
    monkeypatch.setattr(run.FileSource, "batches", lambda self, times: [np.arange(len(times))])
    assert files(run.run(config, tmp_path / "one")) == files(by_month)
    res = VoyageResults.load(by_month / "puerto_raft_drift.npz")
    assert (res.status == END_HORIZON).all() and not res.current_is_climatology   # no window ran short
    df = pd.read_csv(by_month / "summary.csv")
    assert set(zip(df["month"], df["currents"], df["n"])) == {(1, "reanalysis", 5), (2, "reanalysis", 1)}


def test_reading_a_window_takes_little_more_memory_than_the_window(tmp_path):
    # Hourly winds make windows of GBs (SPEC: the time-step test), so a window is read file by file into
    # one array, not concatenated and then flipped to ascending latitude in further copies.
    lon, lat = np.arange(-100.0, -69.9, 0.25), np.arange(10.0, -15.1, -0.25)   # ERA5: descending latitude
    for month in ("2010-01", "2010-02", "2010-03", "2010-04"):
        start, end = np.datetime64(month, "M"), np.datetime64(month, "M") + np.timedelta64(1, "M")
        t = np.arange(start.astype("datetime64[h]"), end.astype("datetime64[h]"), np.timedelta64(6, "h"))
        data = np.random.default_rng(0).normal(size=(t.size, lat.size, lon.size)).astype(np.float32)
        dims = ("valid_time", "latitude", "longitude")
        xr.Dataset({"u10": (dims, data), "v10": (dims, -data)},
                   coords={"valid_time": t.astype("datetime64[ns]"), "latitude": lat, "longitude": lon}
                   ).to_netcdf(tmp_path / f"era5_{month}.nc")
    series = run._FileSeries(tmp_path, {"paths": "era5_*.nc", "u": "u10", "v": "v10"}, "wind")
    tracemalloc.start()
    try:
        field = series.window(series.times[0], series.times[-1])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 1.75 * (field.u.nbytes + field.v.nbytes)
    np.testing.assert_array_equal(field.lat, lat[::-1])
    with xr.open_dataset(tmp_path / "era5_2010-03.nc") as ds:
        k = np.searchsorted(field.time, geo.to_epoch_hours(ds["valid_time"].values[0]))
        np.testing.assert_array_equal(field.u[k], ds["u10"].values[0, ::-1])
        np.testing.assert_array_equal(field.v[k], ds["v10"].values[0, ::-1])


def test_each_batch_lets_the_last_batch_fields_go_before_reading_its_own(tiny, tmp_path, monkeypatch):
    path, out = tiny
    earlier = []

    def window(self, t0, t1):
        gc.collect()
        assert all(ref() is None for ref in earlier), "the last batch's fields are still held"
        env = Environment(self.env.wind, self.env.current, wind_cap_ms=self.env.wind_cap_ms)
        earlier.append(weakref.ref(env))
        return env

    monkeypatch.setattr(run.SyntheticSource, "window", window)
    monkeypatch.setattr(run.SyntheticSource, "batches", lambda self, times: np.array_split(np.arange(len(times)), 3))
    run.run(path, tmp_path)
    assert len(earlier) == 3 and files(tmp_path) == files(out)


# --- configs and the command line -----------------------------------------------------

# SPEC "Model design": the far side's destinations.
EASTERN_POLYNESIA = {"South Marquesas", "North Marquesas", "Tuamotus", "Society Islands", "Austral Islands", "Gambier",
                     "Pitcairn Islands", "Rapa Nui"}
COLOMBIA_TO_CENTRAL_CHILE = {"Coast: Colombia", "Coast: Ecuador", "Coast: Northern Peru", "Coast: Southern Peru",
                             "Coast: Northern Chile", "Coast: Central Chile"}


@pytest.mark.parametrize("name", ["synthetic_demo", "coarse_decade"])
def test_shipped_configs_resolve_with_the_far_side_as_destination(name):
    path = ROOT / "configs" / f"{name}.yaml"
    cfg = run.load_config(path)
    cat = catalog.build_catalog(path.parent / cfg["catalog"]["config"])   # the placeholder has every group
    groups = list(dict.fromkeys(lm.group for lm in cat.landmasses))
    cells = run.resolve_cells(cfg, path.parent, cat, groups)
    assert [c.name for c in cells] == [c["name"] for c in cfg["cells"]]
    for cell in cells:
        dest = {g for g, d in zip(groups, cell.dest_groups) if d}
        from_polynesia = cat.group_region[cell.departure["group"]] == "polynesia"
        assert dest == (COLOMBIA_TO_CENTRAL_CHILE if from_polynesia else EASTERN_POLYNESIA), cell.name
        assert groups[cell.origin_group] == cell.departure["group"]
        assert cell.target is None or cell.target["group"] in dest
        # The voyages carry those destinations: only they end a voyage (SPEC).
        np.testing.assert_array_equal(cell.launches(np.arange(2), np.zeros(2)).dest_groups,
                                      np.broadcast_to(cell.dest_groups, (2, len(groups))))


def test_a_single_destination_region_needs_no_list(tmp_path):
    cell = {**CONFIG["cells"][0], "destinations": {"regions": "americas"}}
    cfg = run.load_config(write_world(tmp_path, {**CONFIG, "cells": [cell]}))
    assert cfg["cells"][0]["destinations"] == {"regions": ["americas"], "groups": []}


@pytest.mark.parametrize("change, message", [
    ({"colour": "red"}, "unknown keys"),
    ({"analysis": {"by": ["enso"]}}, "no enso_index"),
    ({"launches": {"start": "2001-02-01", "end": "2001-01-01"}}, "start <= end"),
    ({"cells": [{**CONFIG["cells"][0], "strategy": "drift"}]}, "drift craft only drift"),
    ({"cells": [{**CONFIG["cells"][0], "target": None}]}, "needs a target"),
    ({"cells": [{**CONFIG["cells"][0], "departure": "Atlantis"}]}, "not among the departures"),
    ({"cells": [{**CONFIG["cells"][0], "max_days_at_sea": 10}]}, "no supply limit"),
    ({"cells": [{**CONFIG["cells"][0], "destinations": {"regions": ["america"]}}]},
     "cells[0].destinations: unknown regions ['america']"),
    # Values of the wrong type name their key.
    ({"seed": None}, "seed must be a number"),
    ({"sim": {"track_every": 2.5}}, "sim.track_every must be a whole number"),
    ({"rules": [{"name": "main", "radius_km": "far"}]}, "rules[0].radius_km must be a number"),
    ({"cells": [{**CONFIG["cells"][0], "strategy": {"name": "hold_course", "tack_hysteresis_deg": "wide"}}]},
     "cells[0].strategy:"),
    ({"environment": {"type": "files", "wind": {"paths": ["a.nc"]}, "current": {"paths": "b.nc"}}},
     "environment.wind.paths must be a path"),
    ({"analysis": {"n_boot": 0}}, "analysis.n_boot must be at least 1"),
    ({"analysis": {"enso_index_early": "hadisst.data"}}, "configure both"),
])
def test_bad_configs_fail_with_a_message(tmp_path, capsys, change, message):
    path = write_world(tmp_path, {**CONFIG, **change})
    assert run.main(["run", str(path), "--out", str(tmp_path / "out")]) == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_broken_yaml_fails_with_a_message(tmp_path, capsys):
    path = tmp_path / "broken.yaml"
    path.write_text("name: [tiny\n")
    assert run.main(["run", str(path), "--out", str(tmp_path / "out")]) == 1
    assert f"{path} is not valid YAML" in capsys.readouterr().err


def test_bad_inputs_beyond_the_config_fail_with_a_message(tmp_path, capsys, data):
    # A directory for a config, a place without a longitude, a wrong variable name in the data files.
    assert run.main(["run", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    assert "Is a directory" in capsys.readouterr().err
    config = write_world(tmp_path / "places", CONFIG)
    (config.parent / "places.yaml").write_text(PLACES.replace("lon: -80.3, ", "", 1))
    assert run.main(["run", str(config), "--out", str(tmp_path / "out")]) == 1
    assert "departure 'Puerto': no lon" in capsys.readouterr().err
    environment = {**FILES_CONFIG["environment"], "wind": {"paths": "era5/era5_uv10_*.nc", "u": "u100"}}
    config = write_world(data.parent, {**FILES_CONFIG, "environment": environment})
    assert run.main(["run", str(config), "--out", str(tmp_path / "out")]) == 1
    assert "has no variable 'u100'; it has ['u10', 'v10']" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_fetch_dry_runs_list_their_downloads(tmp_path, capsys, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access in a test")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", refuse)
    assert run.main(["fetch", "era5", "--years", "2010", "--months", "1-2", "--out", str(tmp_path),
                     "--dry-run"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"would fetch {tmp_path / 'era5_uv10_2010-01.nc'}", f"would fetch {tmp_path / 'era5_uv10_2010-02.nc'}"]
    assert run.main(["fetch", "enso", "--out", str(tmp_path), "--dry-run"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(fetch.ENSO_FILES)
    assert run._ints("2010-2012,2015") == [2010, 2011, 2012, 2015]
    # A reversed range is an error, not "nothing to fetch"; so is an hour that does not exist.
    for args, message in ((["--years", "2019-2010"], "ends before it starts"),
                          (["--years", "2010", "--months", "3-1"], "ends before it starts"),
                          (["--years", "2010", "--months", "1", "--hours", "25"], "bad hour 25")):
        assert run.main(["fetch", "era5", *args, "--out", str(tmp_path), "--dry-run"]) == 1
        assert message in capsys.readouterr().err


def test_reruns_in_new_processes_are_byte_identical(tmp_path):
    # A rerun is a new Python process with its own string-hash seed: nothing written may depend on set or
    # hash order. The synthetic demo has many destination groups per cell (eight islands, six coasts).
    doc = yaml.safe_load((ROOT / "configs" / "synthetic_demo.yaml").read_text(encoding="utf-8"))
    doc["catalog"] = {**doc["catalog"], "config": str(ROOT / "configs" / doc["catalog"]["config"]),
                      "cache_dir": str(tmp_path / "cache")}
    doc.update(vessels=str(ROOT / "configs" / doc["vessels"]), places=str(ROOT / "configs" / doc["places"]),
               launches={**doc["launches"], "every_days": 61}, rules=doc["rules"][:1],
               raster={**doc["raster"], "res_deg": 0.2}, analysis={**doc["analysis"], "supplies": [30]},
               cells=[{**cell, "max_days_at_sea": 30} for cell in doc["cells"]])
    config = tmp_path / "demo.yaml"
    config.write_text(yaml.safe_dump(doc, sort_keys=False))
    outs = []
    for seed in ("1", "2"):
        outs.append(tmp_path / f"out{seed}")
        subprocess.run([sys.executable, "-m", "voyage_sim", "run", str(config), "--out", str(outs[-1])],
                       env={**os.environ, "PYTHONHASHSEED": seed}, check=True, capture_output=True)
    assert files(outs[0]) == files(outs[1])


def test_python_dash_m_runs_the_command_line(tmp_path):
    done = subprocess.run([sys.executable, "-m", "voyage_sim", "fetch", "gshhg", "--out", str(tmp_path),
                           "--dry-run"], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == f"would fetch {tmp_path}"
