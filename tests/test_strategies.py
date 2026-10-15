from types import SimpleNamespace

import numpy as np
import pytest

from voyage_sim import geo
from voyage_sim.strategies import (
    Downwind, Drift, HoldCourse, LatitudeSailing, StepContext, WaitForWind, make_strategy, steer_toward,
)

CANOE = SimpleNamespace(min_twa_deg=75.0, can_sail=True)
DRIFTER = SimpleNamespace(min_twa_deg=180.0, can_sail=False)
KN = geo.KNOT_MS


def make_ctx(lon, lat, target_lon, target_lat, wind_from_deg, wind_kn=15.0, vessel=CANOE, active=True):
    """Context for voyages in a wind blowing from a compass direction; every array has one entry per voyage."""
    *arrays, active = (np.array(a) for a in np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(x, dtype=float)) for x in (lon, lat, target_lon, target_lat, wind_from_deg,
                                                               wind_kn, active))))
    lon, lat, target_lon, target_lat, wind_from_deg, wind_kn = arrays
    active = active.astype(bool)
    wind_u, wind_v = geo.heading_to_uv(wind_kn * KN, wind_from_deg + 180.0)
    return StepContext(lon=lon, lat=lat, t=np.zeros(lon.shape), wind_u=wind_u, wind_v=wind_v,
                       target_lon=target_lon, target_lat=target_lat, vessel=vessel, dt_hours=6.0, active=active)


# --- steer_toward -----------------------------------------------------------------

def test_steer_toward_direct_course_sets_tack_side():
    # Northerly wind, 75 deg limit: east, south-west and the cone edge itself are sailable.
    heading, tack = steer_toward([90.0, 200.0, 75.0, 180.0], 0.0, 75.0, np.zeros(4, np.int8), 15.0)
    assert heading == pytest.approx([90.0, 200.0, 75.0, 180.0])
    # Wind over the side it is on: course clockwise of the wind is tack +1.
    assert tack.tolist() == [1, -1, 1, -1]
    assert tack.dtype == np.int8


def test_steer_toward_close_hauled_picks_nearer_tack():
    heading, tack = steer_toward([10.0, 350.0, 0.0], 0.0, 75.0, np.zeros(3, np.int8), 15.0)
    assert heading == pytest.approx([75.0, 285.0, 75.0])
    assert tack.tolist() == [1, -1, 1]  # dead upwind ties go to +1


def test_close_hauled_headings_are_sailable_by_vessel_test():
    # The vessel re-derives the TWA from the heading; rounding must never put a
    # close-hauled heading inside its no-go cone.
    rng = np.random.default_rng(1)
    u, v = rng.normal(size=(2, 20_000))
    wf = geo.wind_from_deg(u, v)
    for tack in (-1, 1):
        heading, _ = steer_toward(wf, wf, 75.0, np.full(wf.size, tack, np.int8), 15.0)
        twa = geo.true_wind_angle(heading, geo.wind_from_deg(u, v))
        assert np.all(twa >= 75.0)
        assert twa == pytest.approx(np.full(wf.size, 75.0), abs=1e-6)


def test_hysteresis_keeps_tack_until_other_is_clearly_better():
    # On tack -1 (heading 285): the other tack is closer by 10, 15 and 20 deg.
    heading, tack = steer_toward([5.0, 7.5, 10.0], 0.0, 75.0, np.full(3, -1, np.int8), 15.0)
    assert heading == pytest.approx([285.0, 285.0, 75.0])
    assert tack.tolist() == [-1, -1, 1]
    # Symmetric from tack +1.
    heading, tack = steer_toward([-5.0, -10.0], 0.0, 75.0, np.ones(2, np.int8), 15.0)
    assert heading == pytest.approx([75.0, 285.0])
    assert tack.tolist() == [1, -1]


def test_dead_upwind_is_an_exact_tie_in_any_wind():
    # Course straight into the wind from any direction: a fresh tack goes +1, and
    # with zero hysteresis (downwind's clamp) an existing tack is never flipped by rounding.
    wf = geo.wind_from_deg(*np.random.default_rng(4).normal(size=(2, 20_000)))
    _, tack = steer_toward(wf, wf, 75.0, np.zeros(wf.size, np.int8), 15.0)
    assert np.all(tack == 1)
    for side in (-1, 1):
        heading, tack = steer_toward(wf, wf, 90.0, np.full(wf.size, side, np.int8), 0.0)
        assert np.all(tack == side)
        assert heading == pytest.approx(geo.wrap360(wf + side * 90.0), abs=1e-6)


def test_steer_toward_nan_course_keeps_tack():
    heading, tack = steer_toward([np.nan, 10.0], [0.0, np.nan], 75.0, np.array([-1, 1], np.int8), 15.0)
    assert np.all(np.isnan(heading))
    assert tack.tolist() == [-1, 1]


# --- simple strategies ----------------------------------------------------------------

def test_drift_never_steers():
    ctx = make_ctx(-140.0, -10.0, -80.0, -10.0, [0.0, 90.0, 270.0])
    assert np.all(np.isnan(Drift().heading({}, ctx)))


@pytest.mark.parametrize("name", ["drift", "downwind", "hold_course", "wait_for_wind", "latitude"])
def test_vessel_that_cannot_sail_never_steers(name):
    strategy = make_strategy(name)
    state = strategy.init_state(3)
    before = {k: a.copy() for k, a in state.items()}
    ctx = make_ctx(-140.0, -10.0, [-80.0, np.nan, -150.0], -12.0, 90.0, vessel=DRIFTER)
    assert np.all(np.isnan(strategy.heading(state, ctx)))
    assert all(np.array_equal(state[k], before[k]) for k in state)


def test_downwind_clamps_to_cone_edge_and_runs_without_target():
    # Northerly wind at the equator; targets due N, NE-ish, NW-ish, due E, due S, none.
    ctx = make_ctx(-120.0, 0.0, [-120.0, -119.0, -121.0, -110.0, -120.0, np.nan],
                   [5.0, 5.0, 5.0, 0.0, -5.0, np.nan], 0.0)
    strategy = Downwind()  # 90 deg limit, wider than the canoe's 75
    state = strategy.init_state(6)
    heading = strategy.heading(state, ctx)
    # Upwind targets clamp to the beam on their side (dead upwind: tack +1);
    # the beam itself and anything abaft it are steered directly.
    assert heading == pytest.approx([90.0, 90.0, 270.0, 90.0, 180.0, 180.0])
    # Dead upwind keeps whichever side the voyage is already on.
    state["tack"][0] = -1
    assert strategy.heading(state, ctx)[0] == pytest.approx(270.0)


def test_downwind_uses_the_wider_limit_and_never_enters_the_cone():
    narrow = SimpleNamespace(min_twa_deg=110.0, can_sail=True)
    ctx = make_ctx(-120.0, 0.0, -110.0, 0.0, 0.0, vessel=narrow)  # target due east, 90 off the wind
    assert Downwind().heading(Downwind().init_state(1), ctx) == pytest.approx([110.0])

    rng = np.random.default_rng(2)
    n = 2000
    ctx = make_ctx(-120.0, 0.0, rng.uniform(-130, -110, n), rng.uniform(-10, 10, n), rng.uniform(0, 360, n))
    strategy = Downwind(min_twa_deg=100.0)
    heading = strategy.heading(strategy.init_state(n), ctx)
    wf = geo.wind_from_deg(ctx.wind_u, ctx.wind_v)
    bearing = geo.initial_bearing_deg(ctx.lon, ctx.lat, ctx.target_lon, ctx.target_lat)
    assert np.all(geo.true_wind_angle(heading, wf) >= 100.0)
    free = geo.true_wind_angle(bearing, wf) >= 100.0
    assert heading[free] == pytest.approx(bearing[free])
    # Clamped headings sit on the cone edge on the target's side of the wind.
    clamped = ~free
    assert geo.true_wind_angle(heading[clamped], wf[clamped]) == pytest.approx(np.full(clamped.sum(), 100.0))
    assert np.all(np.sign(geo.angle_diff(heading, wf))[clamped] == np.sign(geo.angle_diff(bearing, wf))[clamped])


# --- beating to windward ---------------------------------------------------------------

def test_hold_course_beats_to_windward():
    # Targets 500 km dead upwind in a steady wind from three directions; the
    # canoe makes 5 kn along whatever heading it is given, 6-hour steps.
    wind_dir = np.array([0.0, 120.0, 250.0])
    n, dt_h, speed = 3, 6.0, 5.0 * KN
    lon, lat = np.full(n, -140.0), np.full(n, -20.0)
    target_lon, target_lat = geo.advance(lon, lat, *geo.heading_to_uv(1.0, wind_dir), 500e3)
    start_km = geo.haversine_km(lon, lat, target_lon, target_lat)
    strategy = HoldCourse()
    state = strategy.init_state(n)
    arrived = np.zeros(n, bool)
    steps = np.zeros(n, int)
    tacks = np.zeros(n, int)
    for _ in range(200):
        ctx = make_ctx(lon, lat, target_lon, target_lat, wind_dir, active=~arrived)
        before = state["tack"].copy()
        heading = strategy.heading(state, ctx)
        tacks += ctx.active & (before != 0) & (state["tack"] != before)
        new_lon, new_lat = geo.advance(lon, lat, *geo.heading_to_uv(speed, np.nan_to_num(heading)), dt_h * 3600)
        lon, lat = np.where(ctx.active, new_lon, lon), np.where(ctx.active, new_lat, lat)
        steps += ctx.active
        arrived |= geo.haversine_km(lon, lat, target_lon, target_lat) < 20.0
        if arrived.all():
            break
    assert arrived.all()
    assert np.all((tacks >= 3) & (tacks <= steps // 2))
    made_good = (start_km - geo.haversine_km(lon, lat, target_lon, target_lat)) * 1e3 / (steps * dt_h * 3600)
    assert made_good == pytest.approx(np.full(n, speed * np.cos(np.radians(75.0))), rel=0.05)


# --- wait for wind -------------------------------------------------------------------

def test_wait_for_wind_departs_only_in_fair_wind():
    # Target due north; the canoe needs the course 75 + 10 deg off the wind and 5-25 kn.
    wind_from = [90.0, 80.0, 180.0, 180.0, 180.0, 0.0, 180.0]
    wind_kn = [10.0, 10.0, 10.0, 3.0, 30.0, 10.0, 10.0]
    target_lat = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.nan]
    ctx = make_ctx(-140.0, -10.0, -140.0, target_lat, wind_from, wind_kn)
    strategy = WaitForWind()
    assert strategy.max_wait_days == 60.0
    assert strategy.ready_to_depart(strategy.init_state(7), ctx).tolist() == [
        True, False, True, False, False, False, False]
    assert not WaitForWind().ready_to_depart({}, make_ctx(-140.0, -10.0, -140.0, 0.0, 90.0, vessel=DRIFTER)).any()


def test_wait_for_wind_at_sea_is_hold_course():
    rng = np.random.default_rng(3)
    n = 500
    ctx = make_ctx(-140.0, -10.0, rng.uniform(-150, -130, n), rng.uniform(-20, 0, n), rng.uniform(0, 360, n))
    wait, hold = WaitForWind(tack_hysteresis_deg=20.0), HoldCourse(tack_hysteresis_deg=20.0)
    s_wait, s_hold = wait.init_state(n), hold.init_state(n)
    s_wait["tack"][:] = s_hold["tack"][:] = rng.choice([-1, 0, 1], n)
    assert np.array_equal(wait.heading(s_wait, ctx), hold.heading(s_hold, ctx))
    assert np.array_equal(s_wait["tack"], s_hold["tack"])
    assert HoldCourse.max_wait_days == 0.0


# --- latitude sailing -----------------------------------------------------------------------

def test_latitude_sailing_phases():
    # Westerly wind, so every leg here is sailable directly.
    strategy = LatitudeSailing()
    state = strategy.init_state(1)

    def steer(lon, lat, target_lon=-80.0, target_lat=-12.0):
        heading = strategy.heading(state, make_ctx(lon, lat, target_lon, target_lat, 270.0))[0]
        return heading, int(state["phase"][0])

    assert steer(-140.0, -10.0) == (pytest.approx(180.0), 0)       # south to the target's latitude
    heading, phase = steer(-140.0, -11.6)                           # within 0.5 deg: run east
    assert phase == 1
    assert heading == pytest.approx(geo.initial_bearing_deg(-140.0, -11.6, -135.0, -12.0))
    assert 90.0 < heading < 100.0                                    # converging back onto the parallel
    heading, phase = steer(-81.5, -12.3)                            # within 2 deg of longitude: close it
    assert phase == 2
    assert heading == pytest.approx(geo.initial_bearing_deg(-81.5, -12.3, -80.0, -12.0))
    assert steer(-140.0, -5.0)[1] == 2                               # phases never go back

    # Westbound, already on the parallel; and straight to phase 2 when both hold at once (northerly wind).
    west = LatitudeSailing()
    state_w = west.init_state(2)
    heading = west.heading(state_w, make_ctx([-80.0, -141.0], -12.0, -140.0, -12.2, 0.0))
    assert state_w["phase"].tolist() == [1, 2]
    assert 265.0 < heading[0] < 275.0
    assert heading[1] == pytest.approx(geo.initial_bearing_deg(-141.0, -12.0, -140.0, -12.2))


def test_latitude_sailing_via_westerlies():
    strategy = LatitudeSailing(via_lat=-38.0)
    # Three voyages bound for (-75, -12): on the target's parallel, near -38, and near the coast at -38.
    ctx = make_ctx([-140.0, -140.0, -76.0], [-12.0, -37.8, -38.0], -75.0, -12.0, 270.0)
    state = strategy.init_state(3)
    heading = strategy.heading(state, ctx)
    assert state["phase"].tolist() == [0, 1, 2]
    assert heading[0] == pytest.approx(180.0)    # head south to the westerlies first
    assert heading[1] == pytest.approx(geo.initial_bearing_deg(-140.0, -37.8, -135.0, -38.0))
    assert heading[2] == pytest.approx(geo.initial_bearing_deg(-76.0, -38.0, -75.0, -12.0))


def test_latitude_sailing_ends_phase_0_on_crossing_the_parallel():
    # A step longer than the 1 deg tolerance band (7 kn for 12 h is 1.4 deg) can
    # jump the whole band. Crossing the parallel still ends phase 0; without that
    # the vessel would turn back and bounce across it for ever.
    strategy = LatitudeSailing()
    state = strategy.init_state(2)

    def steer(lat):
        return strategy.heading(state, make_ctx(-140.0, lat, -80.0, -12.0, 270.0))

    assert steer([-11.4, -12.6]) == pytest.approx([180.0, 0.0])   # approaching from north and south
    assert state["phase"].tolist() == [0, 0]
    heading = steer([-12.8, -11.2])                                # each one step past the band
    assert state["phase"].tolist() == [1, 1]
    assert heading == pytest.approx(geo.initial_bearing_deg(-140.0, [-12.8, -11.2], -135.0, -12.0))


def test_latitude_sailing_beats_against_a_headwind():
    # Phase 0 due south into a southerly: close-hauled 75 deg off it, not straight into it.
    strategy = LatitudeSailing(via_lat=-38.0)
    state = strategy.init_state(1)
    ctx = make_ctx(-140.0, -10.0, -75.0, -12.0, [180.0])
    assert strategy.heading(state, ctx)[0] == pytest.approx(255.0)  # dead upwind: fresh tack goes +1
    assert state["tack"][0] == 1
    # A slight veer towards the other tack is not enough to make it go about.
    ctx = make_ctx(-140.0, -10.0, -75.0, -12.0, [185.0])
    assert strategy.heading(state, ctx)[0] == pytest.approx(260.0)


# --- state discipline ----------------------------------------------------------------------

def test_inactive_voyages_are_not_decided_and_keep_state():
    active = np.array([False, True, False])
    ctx = make_ctx(-140.0, -10.0, -140.0, 0.0, 0.0, active=active)  # target dead upwind of all three
    hold = HoldCourse()
    state = {"tack": np.array([0, 0, -1], np.int8)}
    heading = hold.heading(state, ctx)
    assert state["tack"].tolist() == [0, 1, -1]
    assert np.isnan(heading[[0, 2]]).all() and heading[1] == pytest.approx(75.0)

    lat_strategy = LatitudeSailing()
    state = lat_strategy.init_state(3)
    lat_strategy.heading(state, make_ctx(-81.0, -12.0, -80.0, -12.0, 270.0, active=active))
    assert state["phase"].tolist() == [0, 2, 0]


@pytest.mark.parametrize("name", ["drift", "downwind", "hold_course", "wait_for_wind", "latitude"])
def test_empty_batch(name):
    strategy = make_strategy(name)
    ctx = make_ctx(np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    state = strategy.init_state(0)
    assert strategy.heading(state, ctx).shape == (0,)
    assert strategy.ready_to_depart(state, ctx).shape == (0,)


def test_make_strategy():
    assert isinstance(make_strategy("drift"), Drift)
    assert make_strategy("latitude", via_lat=-38.0).via_lat == -38.0
    assert make_strategy("wait_for_wind", max_wait_days=30).max_wait_days == 30
    # Positional order is the contract's: max_wait_days first.
    assert WaitForWind(30.0).max_wait_days == 30.0
    with pytest.raises(ValueError, match="unknown strategy"):
        make_strategy("row")
    with pytest.raises(ValueError, match="unknown parameters"):
        make_strategy("downwind", via_lat=-38.0)
    with pytest.raises(ValueError):
        make_strategy("wait_for_wind", min_tws_kn=30.0, max_tws_kn=20.0)
