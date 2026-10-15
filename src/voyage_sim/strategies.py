"""Steering strategies: how a crew chooses a heading each step.

A strategy turns what a voyage knows (position, wind relative to the water,
target) into a compass heading to steer, or NaN to let the craft drift. It
never moves anything: the simulator adds steering error and asks the vessel how
fast it goes on that heading in that wind.

Every rule that wants a course the wind may not allow goes through
``steer_toward``: steer the course if it is at least the vessel's minimum angle
off the wind, otherwise sail close-hauled on the tack nearer to it, changing
tack only when the other is better by more than a hysteresis angle. The
hysteresis is what turns a beat to windward into long boards instead of a tack
every step.

Strategies keep per-voyage memory (tack, route phase) as arrays in a dict from
``init_state``; ``heading`` updates them in place, only for active voyages.
Strategies read two things from the vessel: ``min_twa_deg`` and ``can_sail``.
A rule that sails no closer to the wind than the vessel could says so in
``closest_twa_deg``, and the simulator holds steering error to that limit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from voyage_sim import geo

if TYPE_CHECKING:
    from voyage_sim.vessels import Vessel

# Close-hauled headings sit this far outside the no-go cone: the vessel
# re-derives the true wind angle from the heading, and without the margin float
# rounding puts a few percent of them a hair inside the cone, where they stall.
_EDGE_EPS_DEG = 1e-9


@dataclass
class StepContext:
    """What a crew knows when choosing headings, one entry per voyage."""

    lon: np.ndarray
    lat: np.ndarray
    t: np.ndarray                 # epoch hours
    wind_u: np.ndarray            # relative to the water, m/s
    wind_v: np.ndarray
    target_lon: np.ndarray        # NaN when the voyage has no target
    target_lat: np.ndarray
    vessel: Vessel
    dt_hours: float
    active: np.ndarray            # bool; only these voyages are being decided


def steer_toward(desired_deg, wind_from_deg, min_twa_deg, tack, hysteresis_deg):
    """Heading that best makes ``desired_deg`` in this wind, and the tack it is on.

    A desired course at least ``min_twa_deg`` off the wind is steered as is, and
    the tack becomes the side the wind is on, so a later beat starts from there.
    Otherwise the vessel sails close-hauled: tack +1 steers ``wind_from +
    min_twa``, tack -1 steers ``wind_from - min_twa``. It keeps its current tack
    unless the other heading is closer to the desired course by more than
    ``hysteresis_deg``; tack 0 (unset) takes the closer one, +1 on a tie.
    A NaN course or wind gives a NaN heading and leaves the tack as it was.
    """
    desired, wind_from, tack = np.broadcast_arrays(
        np.asarray(desired_deg, dtype=float), np.asarray(wind_from_deg, dtype=float),
        np.asarray(tack).astype(np.int8))
    off = geo.angle_diff(desired, wind_from)  # > 0: the course lies clockwise of the wind
    a = np.abs(off)
    min_twa = np.asarray(min_twa_deg, dtype=float)
    direct = a >= min_twa

    # Inside the cone the nearer tack is the one on the course's side of the
    # wind. Its advantage over the other is computed from |off| alone, so a dead
    # upwind course is an exact tie whatever the wind direction.
    nearer = np.where(off < 0, -1, 1)
    gain = np.minimum(min_twa + a, 360.0 - min_twa - a) - (min_twa - a)
    beat_tack = np.where((tack == 0) | ((tack != nearer) & (gain > hysteresis_deg)), nearer, tack)

    edge = np.minimum(min_twa + _EDGE_EPS_DEG, 180.0)
    plus = geo.wrap360(wind_from + edge)
    minus = geo.wrap360(wind_from - edge)

    valid = np.isfinite(off)
    heading = np.where(direct, geo.wrap360(desired), np.where(beat_tack > 0, plus, minus))
    new_tack = np.where(valid, np.where(direct, np.sign(off), beat_tack), tack).astype(np.int8)
    return np.where(valid, heading, np.nan), new_tack


def _steer(state, ctx, desired, min_twa_deg, hysteresis_deg):
    """``steer_toward`` on each voyage's own tack, recording the new tack where active."""
    wind_from = geo.wind_from_deg(ctx.wind_u, ctx.wind_v)
    heading, tack = steer_toward(desired, wind_from, min_twa_deg, state["tack"], hysteresis_deg)
    state["tack"][ctx.active] = tack[ctx.active]
    return heading


def _bearing_to_target(ctx):
    return geo.initial_bearing_deg(ctx.lon, ctx.lat, ctx.target_lon, ctx.target_lat)


class Strategy:
    """A rule for choosing headings, with per-voyage memory."""

    name: ClassVar[str] = ""
    max_wait_days: float = 0.0    # >0 only for wait-for-wind; the simulator enforces it

    def init_state(self, n: int) -> dict[str, np.ndarray]:
        """Fresh per-voyage memory for ``n`` voyages."""
        return {}

    def ready_to_depart(self, state, ctx: StepContext) -> np.ndarray:
        """Where a voyage waiting in port leaves now (default: at once)."""
        return np.ones(np.shape(ctx.lon), dtype=bool)

    def closest_twa_deg(self, vessel: Vessel) -> float:
        """The closest angle to the wind this rule ever sails ``vessel`` at; the simulator keeps
        steering error and wind shifts outside it too. Default: the vessel's own limit."""
        return vessel.min_twa_deg

    def heading(self, state, ctx: StepContext) -> np.ndarray:
        """Compass heading to steer; NaN = do not steer, always for inactive voyages
        and for vessels that cannot sail. Updates ``state`` only where ``ctx.active``."""
        if not ctx.vessel.can_sail:
            return np.full(np.shape(ctx.lon), np.nan)
        return np.where(ctx.active, self._choose(state, ctx), np.nan)

    def _choose(self, state, ctx: StepContext) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class Drift(Strategy):
    """No steering: the craft goes where the current and leeway take it."""

    name: ClassVar[str] = "drift"

    def _choose(self, state, ctx):
        return np.full(np.shape(ctx.lon), np.nan)


@dataclass(frozen=True)
class Downwind(Strategy):
    """Sail only off the wind: never closer than ``min_twa_deg`` or the vessel's own
    limit, whichever is wider.

    Steers the allowed heading nearest the target, which clamps an upwind target
    to the edge of the cone on its side (dead upwind, the side the tack state is
    already on). Without a target it runs dead downwind.
    """

    min_twa_deg: float = 90.0
    name: ClassVar[str] = "downwind"

    def __post_init__(self):
        if not 0 <= self.min_twa_deg <= 180:
            raise ValueError("downwind: min_twa_deg must be within [0, 180]")

    def init_state(self, n):
        return {"tack": np.zeros(n, dtype=np.int8)}

    def closest_twa_deg(self, vessel):
        return max(vessel.min_twa_deg, self.min_twa_deg)

    def _choose(self, state, ctx):
        limit = self.closest_twa_deg(ctx.vessel)
        desired = _bearing_to_target(ctx)
        # Zero hysteresis makes steer_toward an exact clamp to the nearer cone edge.
        heading = _steer(state, ctx, desired, limit, 0.0)
        run = geo.wrap360(geo.wind_from_deg(ctx.wind_u, ctx.wind_v) + 180.0)
        return np.where(np.isnan(desired), run, heading)


@dataclass(frozen=True)
class HoldCourse(Strategy):
    """Steer for the target; tack when it lies closer to the wind than the vessel can sail."""

    tack_hysteresis_deg: float = 15.0
    name: ClassVar[str] = "hold_course"

    def __post_init__(self):
        if not self.tack_hysteresis_deg >= 0:
            raise ValueError(f"{self.name}: tack_hysteresis_deg must be >= 0")

    def init_state(self, n):
        return {"tack": np.zeros(n, dtype=np.int8)}

    def _choose(self, state, ctx):
        return _steer(state, ctx, _bearing_to_target(ctx), ctx.vessel.min_twa_deg, self.tack_hysteresis_deg)


@dataclass(frozen=True)
class WaitForWind(Strategy):
    """Stay in port until the wind lets the vessel lay the target, then hold course.

    The wind is fair when the direct course is at least ``margin_deg`` wider of
    it than the vessel's limit and its speed is within [min_tws_kn, max_tws_kn].
    Voyages with no target, and vessels that cannot sail, never leave.
    """

    max_wait_days: float = 60.0
    margin_deg: float = 10.0
    min_tws_kn: float = 5.0
    max_tws_kn: float = 25.0
    tack_hysteresis_deg: float = 15.0
    name: ClassVar[str] = "wait_for_wind"

    # At sea it is hold_course. Borrowed rather than inherited so the fields keep
    # the contract's positional order.
    init_state = HoldCourse.init_state
    _choose = HoldCourse._choose

    def __post_init__(self):
        HoldCourse.__post_init__(self)
        if not (self.max_wait_days >= 0 and 0 <= self.min_tws_kn <= self.max_tws_kn):
            raise ValueError(f"{self.name}: need max_wait_days >= 0 and 0 <= min_tws_kn <= max_tws_kn")

    def ready_to_depart(self, state, ctx):
        off_wind = geo.true_wind_angle(_bearing_to_target(ctx), geo.wind_from_deg(ctx.wind_u, ctx.wind_v))
        tws_kn = np.hypot(ctx.wind_u, ctx.wind_v) / geo.KNOT_MS
        fair = (off_wind >= ctx.vessel.min_twa_deg + self.margin_deg) & (tws_kn >= self.min_tws_kn) & (
            tws_kn <= self.max_tws_kn)
        return fair & bool(ctx.vessel.can_sail)


@dataclass(frozen=True)
class LatitudeSailing(Strategy):
    """Make the goal latitude, run east or west along it, then close the target.

    The goal latitude is ``via_lat`` if set (-38 gives the westerlies route),
    else the target's. Phase 0 steers due north or south until within
    ``lat_tol_deg`` of it, or across it (a fast step can jump the whole
    tolerance band). Phase 1 aims at the point ``lookahead_deg`` of
    longitude ahead on that parallel, which pulls the vessel back onto it after
    any set. Phase 2 starts within ``final_approach_deg`` of the target's
    longitude and holds course to the target. Phases only advance; every leg is
    steered through ``steer_toward``, so headwinds are beaten against.
    """

    via_lat: float | None = None
    lat_tol_deg: float = 0.5
    lookahead_deg: float = 5.0
    final_approach_deg: float = 2.0
    tack_hysteresis_deg: float = 15.0
    name: ClassVar[str] = "latitude"

    def __post_init__(self):
        if self.via_lat is not None and not -90 < self.via_lat < 90:
            raise ValueError(f"{self.name}: via_lat must be within (-90, 90)")
        if not (self.lat_tol_deg > 0 and self.lookahead_deg > 0 and self.final_approach_deg >= 0
                and self.tack_hysteresis_deg >= 0):
            raise ValueError(f"{self.name}: need lat_tol_deg, lookahead_deg > 0 and "
                             "final_approach_deg, tack_hysteresis_deg >= 0")

    def init_state(self, n):
        # side: which way the goal latitude lay when phase 0 began (+1 north, 0 unset).
        return {"phase": np.zeros(n, dtype=np.int8), "tack": np.zeros(n, dtype=np.int8),
                "side": np.zeros(n, dtype=np.int8)}

    def _choose(self, state, ctx):
        goal_lat = ctx.target_lat if self.via_lat is None else self.via_lat
        dlon = geo.angle_diff(ctx.target_lon, ctx.lon)  # > 0: the target lies east
        phase, side = state["phase"], state["side"]
        toward = np.sign(goal_lat - ctx.lat)
        start = ctx.active & (phase == 0) & (side == 0) & np.isfinite(toward)
        side[start] = toward[start]
        reached = (np.abs(ctx.lat - goal_lat) <= self.lat_tol_deg) | (toward * side < 0)
        # In order, so a voyage already on the parallel near the target's
        # longitude goes straight to the final approach.
        phase[ctx.active & (phase == 0) & reached] = 1
        phase[ctx.active & (phase == 1) & (np.abs(dlon) <= self.final_approach_deg)] = 2

        to_parallel = np.where(goal_lat > ctx.lat, 0.0, 180.0)
        along = geo.initial_bearing_deg(ctx.lon, ctx.lat, ctx.lon + np.sign(dlon) * self.lookahead_deg, goal_lat)
        desired = np.select([phase == 0, phase == 1], [to_parallel, along], _bearing_to_target(ctx))
        has_target = np.isfinite(ctx.target_lon) & np.isfinite(ctx.target_lat)
        return _steer(state, ctx, np.where(has_target, desired, np.nan), ctx.vessel.min_twa_deg,
                      self.tack_hysteresis_deg)


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls for cls in (Drift, Downwind, HoldCourse, WaitForWind, LatitudeSailing)}


def make_strategy(name: str, **params) -> Strategy:
    """Build a strategy from its config key and parameters."""
    if name not in STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    cls = STRATEGIES[name]
    unknown = set(params) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"strategy {name!r}: unknown parameters {sorted(unknown)}")
    return cls(**params)
