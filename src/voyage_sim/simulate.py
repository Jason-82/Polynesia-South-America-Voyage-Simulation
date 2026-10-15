"""The integrator: many voyages stepped together through wind and current.

Every voyage advances in fixed steps of ``dt_hours`` from its own launch time.
A step is one of:

1. **In port**: read the wind at the port and leave if the strategy says so
   (all but wait-for-wind leave at launch). A voyage may leave at any step up
   to ``max_wait_days`` after launch (only at launch, when that is 0); one still
   in port after its last chance gives up (``DID_NOT_SAIL``). Waiting takes
   time but no supplies.
2. **At sea**: read wind and current; the strategy picks a heading from the
   wind relative to the water; steering error is added; the vessel turns the
   heading into a velocity through the water, and the current (plus an optional
   random walk) carries it over the ground. The midpoint integrator re-reads
   both fields half a step ahead and takes the whole step with the velocity
   found there, on the same heading.

Then two checks, in this order. **Land in sight?** The step's segment is swept
for every sighting rule not yet satisfied, recording the days at sea
(interpolated within the step), landmass and position of the first sighting.
**Supplies left?** A voyage ends once every rule has sighted land
(``END_ALL_SIGHTED``) or at the horizon (``END_HORIZON``). Because landfall is
checked first, a sighting on the last day of supplies counts; one in the part of
a last step that runs past the horizon does not.

A track does not depend on the sighting rule or the supply limit, so one run to
the longest horizon answers every rule and every shorter limit: success under
rule r with S days of supplies is ``hit_days[r] <= S``. A field value of NaN
(outside the data's coverage) ends the voyage where it is (``LEFT_DOMAIN``).

Steering error never carries a heading inside the vessel's no-go cone, and
neither does a wind shift at the midpoint: a crew sailing close-hauled steers
by the sail, so such a heading is set to close-hauled on the same side.
Without this, every close-hauled step would stall into leeway drift about half
the time, and upwind sailing would be badly understated.

Random draws (per voyage: speed factor and side of the crosswind leeway; per
step: steering error and random-walk displacement) come from ``noise`` keyed by
voyage id and the step count since launch, so a voyage's track does not depend
on which other voyages share its run.
"""

from __future__ import annotations

import dataclasses
import json
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from voyage_sim import __version__, geo, noise
from voyage_sim.strategies import StepContext, steer_toward

# Random channels. normal(c) draws on uniform channels 2c and 2c + 1, so the
# uniform channel sits well clear of them.
CH_SPEED = 1      # normal, per voyage (step 0): speed factor
CH_HEADING = 2    # normal, per step: steering error
CH_DX = 3         # normal, per step: random walk, east
CH_DY = 4         # normal, per step: random walk, north
CH_CW = 100       # uniform, per voyage (step 0): side of the crosswind leeway

# Status codes
WAITING, AT_SEA, END_ALL_SIGHTED, END_HORIZON, DID_NOT_SAIL, LEFT_DOMAIN = range(6)
STATUS_NAMES = ("waiting", "at_sea", "all_sighted", "horizon", "did_not_sail", "left_domain")

INTEGRATORS = ("midpoint", "euler")
_MIN_SPEED_FACTOR = 0.3
_STEP_TOL = 1e-6  # a duration within this many steps of a whole number counts as that number


@dataclass
class Launches:
    """Where and when each voyage starts, its home group and its target; one entry per voyage.

    Scalars broadcast to every voyage. Voyage ids must be unique: they key each
    voyage's random streams and identify it in results.
    """

    voyage_id: np.ndarray      # int64, stable; keys the random streams
    lon: np.ndarray            # departure point (just offshore)
    lat: np.ndarray
    t0: np.ndarray             # epoch hours of launch (waiting starts here)
    origin_group: np.ndarray   # index into detector.groups, -1 for none
    target_lon: np.ndarray     # NaN if no target
    target_lat: np.ndarray

    def __post_init__(self):
        self.voyage_id = np.atleast_1d(np.asarray(self.voyage_id))
        if not np.issubdtype(self.voyage_id.dtype, np.integer) or self.voyage_id.ndim != 1:
            raise ValueError("voyage_id must be a 1-D integer array")
        self.voyage_id = self.voyage_id.astype(np.int64)
        n = self.voyage_id.size
        if np.unique(self.voyage_id).size != n:
            raise ValueError("voyage_id must be unique: it keys each voyage's random streams")
        if np.issubdtype(np.asarray(self.t0).dtype, np.datetime64):
            raise TypeError("t0 must be epoch hours; convert with geo.to_epoch_hours")
        for name, dtype in (("lon", np.float64), ("lat", np.float64), ("t0", np.float64),
                            ("origin_group", np.int64), ("target_lon", np.float64), ("target_lat", np.float64)):
            value = np.asarray(getattr(self, name), dtype=dtype)
            if value.ndim > 1 or value.size not in (1, n):
                raise ValueError(f"{name} must be a scalar or have one entry per voyage ({n})")
            setattr(self, name, np.broadcast_to(value, (n,)).copy())
        self.lon = geo.wrap_lon(self.lon)
        self.target_lon = geo.wrap_lon(self.target_lon)

    def __len__(self):
        return self.voyage_id.size


@dataclass
class SimConfig:
    """Integration and noise settings for one run."""

    dt_hours: float = 6.0
    max_days_at_sea: float = 90.0   # horizon; shorter supply limits are applied afterwards
    integrator: str = "midpoint"    # "midpoint" (RK2) | "euler"
    speed_sd: float = 0.10          # per-voyage speed factor ~ N(1, sd), floored at 0.3
    heading_sd_deg: float = 10.0    # per-step steering error
    diffusivity_m2s: float = 0.0    # random-walk dispersion K: step sd = sqrt(2 K dt) per axis
    track_every: int = 4            # record a position every N steps (0 = none)
    seed: int = 0

    def __post_init__(self):
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"integrator must be one of {INTEGRATORS}, got {self.integrator!r}")
        if not (0 < self.dt_hours < np.inf and 0 < self.max_days_at_sea < np.inf):
            raise ValueError("dt_hours and max_days_at_sea must be positive and finite")
        if not (0 <= self.speed_sd < np.inf and 0 <= self.heading_sd_deg < np.inf
                and 0 <= self.diffusivity_m2s < np.inf):
            raise ValueError("speed_sd, heading_sd_deg and diffusivity_m2s must be finite and non-negative")
        if int(self.track_every) != self.track_every or self.track_every < 0:
            raise ValueError("track_every must be a non-negative integer")


_ARRAYS = ("voyage_id", "launch_t", "depart_t", "end_t", "end_lon", "end_lat", "days_at_sea", "status",
           "hit_days", "hit_landmass", "hit_lon", "hit_lat")
_INFO = ("rule_names", "landmass_names", "landmass_groups", "group_names", "current_is_climatology", "meta")


@dataclass
class VoyageResults:
    """How every voyage of a run ended, and where and when it first sighted land under each rule.

    Times are epoch hours. A voyage ends at the end of its last step (``end_t``,
    ``end_lon``, ``end_lat``), and ``days_at_sea`` counts whole steps sailed, so
    the last step may run past the horizon; ``hit_days`` is interpolated within
    its step and never exceeds the horizon.
    Sighting arrays are (n_rules, n): NaN (or landmass -1) where never sighted.
    """

    voyage_id: np.ndarray
    launch_t: np.ndarray
    depart_t: np.ndarray       # NaN if the voyage never sailed
    end_t: np.ndarray
    end_lon: np.ndarray
    end_lat: np.ndarray
    days_at_sea: np.ndarray
    status: np.ndarray         # status codes, see STATUS_NAMES
    hit_days: np.ndarray       # days at sea at first sighting
    hit_landmass: np.ndarray   # index into landmass_names
    hit_lon: np.ndarray        # where the voyage was at first sighting
    hit_lat: np.ndarray
    rule_names: list[str]
    landmass_names: list[str]
    landmass_groups: list[str]   # group name of each landmass
    group_names: list[str]
    tracks: np.ndarray | None    # (n, n_rec, 2) float32 lon/lat
    track_days: np.ndarray | None  # (n_rec,) days since launch
    current_is_climatology: bool
    meta: dict                   # config echo, vessel, strategy, seed, version

    def to_frame(self) -> pd.DataFrame:
        """One row per voyage; per-rule columns ``hit_days_<rule>``, ``hit_landmass_<rule>``,
        ``hit_group_<rule>``, ``hit_lon_<rule>``, ``hit_lat_<rule>`` (None/NaN where never sighted)."""
        cols = {name: getattr(self, name) for name in _ARRAYS[:7]}
        cols["status"] = np.array(STATUS_NAMES, dtype=object)[self.status]
        # Index -1 (no sighting) reads the trailing None.
        names = np.array([*self.landmass_names, None], dtype=object)
        groups = np.array([*self.landmass_groups, None], dtype=object)
        for r, rule in enumerate(self.rule_names):
            cols[f"hit_days_{rule}"] = self.hit_days[r]
            cols[f"hit_landmass_{rule}"] = names[self.hit_landmass[r]]
            cols[f"hit_group_{rule}"] = groups[self.hit_landmass[r]]
            cols[f"hit_lon_{rule}"] = self.hit_lon[r]
            cols[f"hit_lat_{rule}"] = self.hit_lat[r]
        return pd.DataFrame(cols)

    def save(self, path):
        """Write an ``.npz`` at exactly ``path``: the arrays, plus names and meta as JSON.

        Equal results give identical bytes, so reruns can be compared file by file.
        """
        arrays = {name: getattr(self, name) for name in _ARRAYS}
        if self.tracks is not None:
            arrays["tracks"], arrays["track_days"] = self.tracks, self.track_days
        info = _jsonable({name: getattr(self, name) for name in _INFO})
        arrays["info"] = np.array(json.dumps(info, sort_keys=True))
        _write_npz(path, arrays)

    @classmethod
    def load(cls, path) -> VoyageResults:
        """Read results written by ``save``."""
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: z[name] for name in _ARRAYS}
            has_tracks = "tracks" in z.files
            tracks = z["tracks"] if has_tracks else None
            track_days = z["track_days"] if has_tracks else None
            info = json.loads(z["info"].item())
        return cls(**arrays, tracks=tracks, track_days=track_days, **info)


def simulate(env, vessel, strategy, launches, detector, cfg) -> VoyageResults:
    """Sail every launch until it sights land under every rule of ``detector``, reaches
    the horizon, leaves the coverage of ``env``, or gives up waiting in port."""
    run = _Run(env, vessel, strategy, launches, detector, cfg)
    k = 0
    run.record(k)
    while np.any(run.status <= AT_SEA):
        run.wait(k)
        run.sail(k)
        k += 1
        if cfg.track_every and k % cfg.track_every == 0:
            run.record(k)
    if cfg.track_every and run.track_steps[-1] != k:
        run.record(k)  # every track ends where its voyage ended
    return run.results()


class _Run:
    """The state of every voyage in one ``simulate`` call, advanced a step at a time."""

    def __init__(self, env, vessel, strategy, launches, detector, cfg):
        self.env, self.vessel, self.strategy, self.detector, self.cfg = env, vessel, strategy, detector, cfg
        self.launches = launches
        if not 0 <= strategy.max_wait_days < np.inf:
            raise ValueError(f"strategy {strategy.name!r}: max_wait_days must be finite and non-negative")
        if np.any((launches.origin_group < -1) | (launches.origin_group >= len(detector.groups))):
            raise ValueError(f"origin_group must be -1 or an index into detector.groups (< {len(detector.groups)})")
        n, n_rules = len(launches), len(detector.rasters)
        self.ids = launches.voyage_id
        self.dt_s = cfg.dt_hours * geo.SECONDS_PER_HOUR
        self.dt_days = cfg.dt_hours / geo.HOURS_PER_DAY
        self.max_steps = _whole_steps(cfg.max_days_at_sea * geo.HOURS_PER_DAY, cfg.dt_hours)
        # The last step at which a voyage still in port may leave: within max_wait_days of launch.
        self.last_wait_step = int(np.floor(strategy.max_wait_days * geo.HOURS_PER_DAY / cfg.dt_hours + _STEP_TOL))

        speed_draw = noise.normal(cfg.seed, self.ids, 0, CH_SPEED)
        self.speed_factor = np.maximum(_MIN_SPEED_FACTOR, 1.0 + cfg.speed_sd * speed_draw)
        self.cw_sign = np.where(noise.uniform(cfg.seed, self.ids, 0, CH_CW) < 0.5, -1.0, 1.0)

        self.lon, self.lat = launches.lon.copy(), launches.lat.copy()
        # Every voyage starts in port; strategies that do not wait leave at the first step.
        self.status = np.full(n, WAITING, dtype=np.int8)
        self.depart_t = np.full(n, np.nan)
        self.end_t = np.full(n, np.nan)
        self.steps_at_sea = np.zeros(n, dtype=np.int64)
        self.hit_days = np.full((n_rules, n), np.nan)
        self.hit_landmass = np.full((n_rules, n), -1, dtype=np.int64)
        self.hit_lon = np.full((n_rules, n), np.nan)
        self.hit_lat = np.full((n_rules, n), np.nan)
        self.armed = np.zeros((n_rules, n), dtype=bool)
        self.state = strategy.init_state(n)
        self.tracks, self.track_steps = [], []

    # --- one step -------------------------------------------------------------

    def wait(self, k):
        """Voyages in port: read the wind and leave if the strategy says so; give up at the last chance."""
        idx = np.flatnonzero(self.status == WAITING)
        if idx.size == 0:
            return
        t = self.launches.t0[idx] + k * self.cfg.dt_hours
        # In port the crew judges the wind itself; the current matters once under way.
        wind_u, wind_v = self.env.wind.sample(self.lon[idx], self.lat[idx], t)
        ok = np.isfinite(wind_u) & np.isfinite(wind_v)
        self._end(idx[~ok], LEFT_DOMAIN, t[~ok])
        idx, t, wind_u, wind_v = idx[ok], t[ok], wind_u[ok], wind_v[ok]
        ready = self._strategy_call(self.strategy.ready_to_depart, idx, self._context(idx, t, wind_u, wind_v))
        self.status[idx[ready]] = AT_SEA
        self.depart_t[idx[ready]] = t[ready]
        if k >= self.last_wait_step:
            self._end(idx[~ready], DID_NOT_SAIL, t[~ready])

    def sail(self, k):
        """Voyages at sea: move one step, look for land, check supplies."""
        idx = np.flatnonzero(self.status == AT_SEA)
        if idx.size == 0:
            return
        cfg = self.cfg
        lon0, lat0 = self.lon[idx], self.lat[idx]
        t = self.launches.t0[idx] + k * cfg.dt_hours
        forcing, ok = self._forcing(lon0, lat0, t)
        idx, lon0, lat0, t, forcing = self._drop_lost(ok, idx, lon0, lat0, t, forcing)

        wind_u, wind_v, cur_u, cur_v = forcing
        heading = self._strategy_call(self.strategy.heading, idx,
                                      self._context(idx, t, wind_u - cur_u, wind_v - cur_v))
        if cfg.heading_sd_deg > 0 and self.vessel.can_sail:
            heading = heading + cfg.heading_sd_deg * noise.normal(cfg.seed, self.ids[idx], k, CH_HEADING)
        walk_u, walk_v = self._random_walk(idx, k)
        u, v = self._ground_velocity(idx, heading, forcing)
        u, v = u + walk_u, v + walk_v
        if cfg.integrator == "midpoint":
            lon_h, lat_h = geo.advance(lon0, lat0, u, v, 0.5 * self.dt_s)
            forcing, ok = self._forcing(lon_h, lat_h, t + 0.5 * cfg.dt_hours)
            idx, lon0, lat0, t, forcing, heading, walk_u, walk_v = self._drop_lost(
                ok, idx, lon0, lat0, t, forcing, heading, walk_u, walk_v)
            u, v = self._ground_velocity(idx, heading, forcing)
            u, v = u + walk_u, v + walk_v
        lon1, lat1 = geo.advance(lon0, lat0, u, v, self.dt_s)

        self._look_for_land(idx, lon0, lat0, lon1, lat1)
        self.lon[idx], self.lat[idx] = lon1, lat1
        self.steps_at_sea[idx] += 1
        t_end = t + cfg.dt_hours
        all_sighted = ~np.isnan(self.hit_days[:, idx]).any(axis=0)
        out_of_supplies = ~all_sighted & (self.steps_at_sea[idx] >= self.max_steps)
        self._end(idx[all_sighted], END_ALL_SIGHTED, t_end[all_sighted])
        self._end(idx[out_of_supplies], END_HORIZON, t_end[out_of_supplies])

    def _look_for_land(self, idx, lon0, lat0, lon1, lat1):
        """Record first sightings along each voyage's segment for the rules it has not yet satisfied."""
        unsighted = np.isnan(self.hit_days[:, idx])
        hit, frac, landmass, armed = self.detector.check_segment(
            lon0, lat0, lon1, lat1, self.launches.origin_group[idx], self.armed[:, idx], unsighted)
        self.armed[:, idx] = armed
        days = (self.steps_at_sea[idx] + frac) * self.dt_days
        # When the horizon is not a whole number of steps, the last step overruns it; land first
        # seen in the overrun came into sight after the supplies ran out.
        r, j = np.nonzero(hit & (days <= self.cfg.max_days_at_sea))
        f, cols = frac[r, j], idx[j]
        self.hit_days[r, cols] = days[r, j]
        self.hit_landmass[r, cols] = landmass[r, j]
        # Same linear lon/lat interpolation along the segment as the detector uses.
        self.hit_lon[r, cols] = geo.wrap_lon(lon0[j] + f * geo.angle_diff(lon1[j], lon0[j]))
        self.hit_lat[r, cols] = lat0[j] + f * (lat1[j] - lat0[j])

    # --- helpers --------------------------------------------------------------

    def _forcing(self, lon, lat, t):
        """(wind_u, wind_v, current_u, current_v) and where all four are finite."""
        wind = self.env.wind.sample(lon, lat, t)
        current = self.env.current.sample(lon, lat, t)
        forcing = (*wind, *current)
        return forcing, np.logical_and.reduce([np.isfinite(x) for x in forcing])

    def _drop_lost(self, ok, idx, lon0, lat0, t, forcing, *rest):
        """End voyages whose forcing is missing, where they stand; return the inputs without them."""
        self._end(idx[~ok], LEFT_DOMAIN, t[~ok])
        if ok.all():
            return (idx, lon0, lat0, t, forcing, *rest)
        return (idx[ok], lon0[ok], lat0[ok], t[ok], tuple(x[ok] for x in forcing),
                *(np.broadcast_to(x, ok.shape)[ok] for x in rest))

    def _ground_velocity(self, idx, heading, forcing):
        """Water velocity on ``heading`` (kept out of the no-go cone) plus the current."""
        wind_u, wind_v, cur_u, cur_v = forcing
        rel_u, rel_v = wind_u - cur_u, wind_v - cur_v
        heading, _ = steer_toward(heading, geo.wind_from_deg(rel_u, rel_v), self.vessel.min_twa_deg, 0, 0.0)
        u, v, _ = self.vessel.water_velocity(heading, rel_u, rel_v, self.cw_sign[idx], self.speed_factor[idx])
        return u + cur_u, v + cur_v

    def _random_walk(self, idx, k):
        """Random-walk velocity: a displacement of sqrt(2 K dt) sd per axis, spread over the step."""
        cfg = self.cfg
        if cfg.diffusivity_m2s == 0:
            return 0.0, 0.0
        scale = np.sqrt(2.0 * cfg.diffusivity_m2s / self.dt_s)
        return (scale * noise.normal(cfg.seed, self.ids[idx], k, CH_DX),
                scale * noise.normal(cfg.seed, self.ids[idx], k, CH_DY))

    def _context(self, idx, t, wind_u, wind_v):
        return StepContext(self.lon[idx], self.lat[idx], t, wind_u, wind_v, self.launches.target_lon[idx],
                           self.launches.target_lat[idx], self.vessel, self.cfg.dt_hours,
                           np.ones(idx.size, dtype=bool))

    def _strategy_call(self, method, idx, ctx):
        """Call a strategy method on voyages ``idx`` with their slice of its state, then store the slice back."""
        sub = {key: arr[idx] for key, arr in self.state.items()}
        out = method(sub, ctx)
        for key, arr in sub.items():
            self.state[key][idx] = arr
        return out

    def _end(self, idx, status, t):
        self.status[idx] = status
        self.end_t[idx] = t

    # --- output ---------------------------------------------------------------

    def record(self, k):
        """Append every voyage's position after ``k`` steps to the tracks (if tracks are kept)."""
        if self.cfg.track_every:
            self.tracks.append(np.column_stack([self.lon, self.lat]).astype(np.float32))
            self.track_steps.append(k)

    def results(self) -> VoyageResults:
        det, cfg = self.detector, self.cfg
        has_tracks = bool(cfg.track_every)
        strategy_params = _jsonable(self.strategy) if dataclasses.is_dataclass(self.strategy) else {}
        meta = {
            "version": __version__,
            "seed": cfg.seed,
            "config": dataclasses.asdict(cfg),
            "vessel": _jsonable(self.vessel),
            "strategy": {"name": self.strategy.name, **strategy_params},
            "substep_km": det.substep_km,
        }
        return VoyageResults(
            voyage_id=self.ids.copy(), launch_t=self.launches.t0.copy(), depart_t=self.depart_t,
            end_t=self.end_t, end_lon=self.lon, end_lat=self.lat, days_at_sea=self.steps_at_sea * self.dt_days,
            status=self.status, hit_days=self.hit_days, hit_landmass=self.hit_landmass, hit_lon=self.hit_lon,
            hit_lat=self.hit_lat, rule_names=list(det.rule_names), landmass_names=list(det.landmass_names),
            landmass_groups=[det.groups[g] for g in det.landmass_group], group_names=list(det.groups),
            tracks=np.stack(self.tracks, axis=1) if has_tracks else None,
            track_days=np.array(self.track_steps) * self.dt_days if has_tracks else None,
            current_is_climatology=bool(self.env.current_is_climatology), meta=meta)


def _whole_steps(hours, dt_hours) -> int:
    """Steps of ``dt_hours`` needed to cover ``hours``, robust to float rounding."""
    return int(np.ceil(hours / dt_hours - _STEP_TOL))


def _jsonable(x):
    """Plain JSON types for a config echo: dataclasses, mappings, sequences, NumPy values."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: _jsonable(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, Mapping):
        return {str(key): _jsonable(value) for key, value in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_jsonable(value) for value in x]
    if isinstance(x, np.generic):
        return x.item()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def _write_npz(path, arrays):
    """Like ``np.savez_compressed``, but with fixed member timestamps so equal arrays give equal bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, arr in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
