import dataclasses

import numpy as np
import pandas as pd
import pytest

from voyage_sim import analyze, geo
from voyage_sim.fields import Environment, FunctionField, UniformField
from voyage_sim.landfall import LandfallDetector, Landmass, SightingRaster, SightingRule, circle_ring
from voyage_sim.simulate import (
    DID_NOT_SAIL,
    END_ALL_SIGHTED,
    END_HORIZON,
    LEFT_DOMAIN,
    Launches,
    SimConfig,
    VoyageResults,
    simulate,
)
from voyage_sim.strategies import Drift, HoldCourse, Strategy, WaitForWind
from voyage_sim.vessels import Leeway, Vessel, parametric_polar

KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)
# Raster cell centres fall on the equator (-4.025 + 80.5 * 0.05 = 0), so along it the
# sighting margin is interpolated in longitude only and stays exactly linear in distance.
BOUNDS = (-152.0, -136.0, -4.025, 3.975)
T0 = float(geo.to_epoch_hours("2000-03-01"))
START = -150.0
DRIFT_MS = 0.5
KM_PER_DAY = DRIFT_MS * 86.4
QUIET = dict(speed_sd=0.0, heading_sd_deg=0.0)

RAFT = Vessel("raft", "drift", Leeway(0.0))
CANOE = Vessel("canoe", "sailing", Leeway(0.04, cw_slope=0.015),
               parametric_polar(75, 7.0, 0.45, [(75, 0.7), (90, 0.9), (110, 1.0), (150, 1.0), (180, 0.85)]),
               min_twa_deg=75, min_sail_tws_kn=2.0, heave_to_tws_kn=34.0)
CURRENT_EAST = Environment(UniformField(0.0, 0.0), UniformField(DRIFT_MS, 0.0))
WIND_FROM_WEST = Environment(UniformField(8.0, 0.0), UniformField(0.0, 0.0))


def east(lon, km):
    """Longitude ``km`` east along the equator."""
    return lon + km / KM_PER_DEG


def point(name, lon, lat=0.0, group=None):
    return Landmass(name, group or name, 0.0, [np.array([[lon, lat]])])


def detector(landmasses, radii=(40.0,)):
    rasters = [SightingRaster.build(landmasses, SightingRule(f"u{r:g}", mode="uniform", radius_km=r), BOUNDS,
                                    res_deg=0.05, cap_km=100.0) for r in radii]
    return LandfallDetector(landmasses, rasters)


def launches(lon=START, lat=0.0, t0=T0, origin=-1, target=(np.nan, np.nan), dest=None, n=None):
    n = n or max(np.size(lon), np.size(lat), np.size(t0), np.size(origin))
    return Launches(np.arange(n), lon, lat, t0, origin, *target, dest_groups=dest)


def take(lau, idx):
    values = {f.name: getattr(lau, f.name) for f in dataclasses.fields(lau)}
    return Launches(**{name: x if x is None else x[idx] for name, x in values.items()})


def assert_same(a, b, idx=slice(None)):
    """Every per-voyage output of ``a`` equals ``b`` at voyages ``idx`` of ``b``."""
    for name in ("voyage_id", "depart_t", "end_t", "end_lon", "end_lat", "days_at_sea", "status",
                 "other_days", "other_landmass", "other_lon", "other_lat"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name)[idx], err_msg=name)
    for name in ("hit_days", "hit_landmass", "hit_lon", "hit_lat"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name)[:, idx], err_msg=name)
    # ``b`` may run longer (other voyages still at sea): past its end a track holds its final position.
    n_rec = a.tracks.shape[1]
    np.testing.assert_array_equal(a.tracks, b.tracks[idx, :n_rec])
    tail = b.tracks[idx, n_rec:]
    np.testing.assert_array_equal(tail, np.broadcast_to(a.tracks[:, -1:], tail.shape))


def swirl(lon, lat, t):
    """A wind from the west-southwest veering and backing over days and across the domain, never ahead
    of the beam for an eastward course, so no voyage beats and every track is smooth."""
    d = np.radians(260 + 45 * np.sin(2 * np.pi * t / 96.0) + 15 * np.sin(np.radians(40 * lon))
                   + 15 * np.cos(np.radians(60 * lat)))
    speed = 8 + 3 * np.sin(2 * np.pi * t / 60.0)
    return -speed * np.sin(d), -speed * np.cos(d)


def eddies(lon, lat, t):
    """An eastward current with eddies a few hundred km across (divergence-free)."""
    a, b = np.radians(60 * lat), np.radians(60 * lon)
    return 0.4 + 0.3 * np.cos(a) * np.sin(b), -0.3 * np.sin(a) * np.cos(b)


@pytest.fixture(scope="module")
def target_ahead():
    """A point island 300 km east of the start, on the equator."""
    return detector([point("Target", east(START, 300.0))])


# --- motion and sighting ------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["euler", "midpoint"])
def test_drift_sights_island_at_the_analytic_time(target_ahead, integrator):
    cfg = SimConfig(integrator=integrator, max_days_at_sea=10.0, track_every=2, **QUIET)
    res = simulate(CURRENT_EAST, RAFT, Drift(), launches(), target_ahead, cfg)
    expected = (300.0 - 40.0) / KM_PER_DAY
    # The margin is exactly linear along the equator, so only its float32 storage limits
    # the sighting: far inside one detector sub-step (2 km).
    assert res.hit_days[0, 0] == pytest.approx(expected, abs=0.01 / KM_PER_DAY)
    assert res.hit_lon[0, 0] == pytest.approx(east(START, 260.0), abs=0.01 / KM_PER_DEG)
    assert res.hit_lat[0, 0] == 0.0 and res.hit_landmass[0, 0] == 0
    # The voyage ends at the end of the step in which it sighted land.
    steps = np.ceil(expected / 0.25)
    assert res.status[0] == END_ALL_SIGHTED and res.days_at_sea[0] == steps * 0.25
    assert res.end_t[0] == T0 + steps * 6.0 and res.depart_t[0] == T0
    assert res.end_lon[0] == pytest.approx(east(START, steps * 0.25 * KM_PER_DAY), abs=1e-9)
    # Tracks every two steps, plus the final position, at days since launch.
    np.testing.assert_allclose(res.track_days, [*np.arange(0, steps * 0.25, 0.5), steps * 0.25])
    np.testing.assert_allclose(res.tracks[0, :, 0], east(START, res.track_days * KM_PER_DAY), atol=1e-4)


def test_canoe_running_before_the_wind_sights_target_at_the_sighting_radius():
    centre = (east(START, 500.0), 1.0)
    det = detector([Landmass("Target", "Target", 100.0, [circle_ring(*centre, 8.0)])])
    lau = launches(target=centre, n=20)
    res = simulate(WIND_FROM_WEST, CANOE, HoldCourse(), lau, det, SimConfig(max_days_at_sea=10.0))
    assert np.all(res.status == END_ALL_SIGHTED) and np.all(res.hit_landmass == 0)
    # Each voyage sights land ~40 km off the coast, whatever its noisy course: raster
    # interpolation (0.05 deg cells) is the only error.
    off_coast = geo.haversine_km(res.hit_lon[0], res.hit_lat[0], *centre) - 8.0
    np.testing.assert_allclose(off_coast, 40.0, atol=0.5)

    # Without noise the canoe makes its polar speed straight down the bearing to the island.
    quiet = simulate(WIND_FROM_WEST, CANOE, HoldCourse(), take(lau, [0]), det,
                     SimConfig(max_days_at_sea=10.0, **QUIET))
    bearing = geo.initial_bearing_deg(START, 0.0, *centre)
    u, v, sailing = CANOE.water_velocity(bearing, 8.0, 0.0, 1.0)
    assert sailing
    sail_km = geo.haversine_km(START, 0.0, *centre) - 48.0
    assert quiet.hit_days[0, 0] == pytest.approx(sail_km / (np.hypot(u, v) * 86.4), rel=0.005)


def test_steering_error_never_stalls_a_close_hauled_canoe():
    # A dead beat against a steady wind: with steering error pushing half the close-hauled
    # headings into the no-go cone, the canoe would stall into leeway drift and lose ground.
    det = detector([point("Far", east(START, 1500.0))])
    lau = launches(target=(east(START, 1500.0), 0.0), n=50)
    headwind = Environment(UniformField(-8.0, 0.0), UniformField(0.0, 0.0))
    res = simulate(headwind, CANOE, HoldCourse(), lau, det, SimConfig(max_days_at_sea=5.0, track_every=0))
    made_good = (res.end_lon - START) * KM_PER_DEG / 5.0
    # Close-hauled at 75 deg makes ~57 km/day to windward; steering error costs some of it.
    assert made_good.min() > 30.0


# --- supplies ----------------------------------------------------------------------

@pytest.mark.parametrize("supply", [3.0, 2.85])
def test_one_run_answers_every_shorter_supply_limit(supply):
    centre = (-137.5, 0.0)
    det = detector([point("Target", *centre)])
    lau = launches(lon=np.linspace(-151.5, -139.5, 25), lat=np.tile([-0.3, 0.0, 0.3], 9)[:25], target=centre)
    cfg = SimConfig(max_days_at_sea=6.0, diffusivity_m2s=100.0, track_every=2)
    long = simulate(WIND_FROM_WEST, CANOE, HoldCourse(), lau, det, cfg)
    short = simulate(WIND_FROM_WEST, CANOE, HoldCourse(), lau, det, dataclasses.replace(cfg, max_days_at_sea=supply))
    # A limit that is not a whole number of steps: the last step runs on to 3 days.
    last_step_end = np.ceil(supply / 0.25) * 0.25
    if supply != last_step_end:
        assert np.any((long.hit_days[0] > supply) & (long.hit_days[0] <= last_step_end))

    sighted = ~np.isnan(short.hit_days[0])
    assert 0 < sighted.sum() < (long.hit_days[0] <= 6.0).sum()   # some voyages need the longer limit
    np.testing.assert_array_equal(sighted, long.hit_days[0] <= supply)
    for name in ("hit_days", "hit_landmass", "hit_lon", "hit_lat"):
        np.testing.assert_array_equal(getattr(short, name)[:, sighted], getattr(long, name)[:, sighted])
    np.testing.assert_array_equal(analyze.success(short, "u40", supply), analyze.success(long, "u40", supply))
    np.testing.assert_array_equal(short.status == END_ALL_SIGHTED, sighted)
    assert np.all(short.status[~sighted] == END_HORIZON) and np.all(short.days_at_sea[~sighted] == last_step_end)
    # Identical tracks up to the shorter limit.
    n_rec = short.tracks.shape[1]
    np.testing.assert_array_equal(short.track_days, long.track_days[:n_rec])
    np.testing.assert_array_equal(short.tracks, long.tracks[:, :n_rec])


def test_sighting_on_the_last_day_of_supplies_counts():
    # Land comes into sight 2.6 days out: within the step from 2.5 to 2.75 days.
    sight_days = 2.6
    det = detector([point("Target", east(START, 40.0 + sight_days * KM_PER_DAY))])

    def run(supply):
        return simulate(CURRENT_EAST, RAFT, Drift(), launches(), det, SimConfig(max_days_at_sea=supply, **QUIET))

    last_step = run(2.75)
    assert last_step.status[0] == END_ALL_SIGHTED
    assert last_step.hit_days[0, 0] == pytest.approx(sight_days, abs=1e-6)
    out_of_supplies = run(2.5)
    assert out_of_supplies.status[0] == END_HORIZON and np.isnan(out_of_supplies.hit_days[0, 0])
    # Supplies for 2.55 days: the last step still runs to 2.75 days, but the land seen
    # in its overrun came into sight after the supplies ran out.
    overrun = run(2.55)
    assert overrun.status[0] == END_HORIZON and np.isnan(overrun.hit_days[0, 0])
    assert overrun.hit_landmass[0, 0] == -1 and overrun.days_at_sea[0] == 2.75
    # Supplies for exactly the time to the sighting (not a whole number of steps) also count it.
    exact_limit = float(last_step.hit_days[0, 0])
    exact = run(exact_limit)
    assert exact.status[0] == END_ALL_SIGHTED and exact.hit_days[0, 0] == exact_limit
    assert analyze.success(exact, "u40", exact_limit)[0]


# --- randomness --------------------------------------------------------------------

def test_reproducible_by_seed_and_independent_of_batch():
    centre = (-140.0, 0.5)
    det = detector([point("Target", *centre)])
    env = Environment(FunctionField(swirl), FunctionField(eddies))
    lau = launches(lat=np.linspace(-1.0, 1.0, 12), t0=T0 + 7.0 * np.arange(12), target=centre)
    cfg = SimConfig(max_days_at_sea=6.0, diffusivity_m2s=50.0, track_every=3, seed=7)
    res = simulate(env, CANOE, HoldCourse(), lau, det, cfg)
    assert np.all(res.status == END_ALL_SIGHTED)

    assert_same(simulate(env, CANOE, HoldCourse(), lau, det, cfg), res)
    other = simulate(env, CANOE, HoldCourse(), lau, det, dataclasses.replace(cfg, seed=8))
    assert np.all(other.hit_days[0] != res.hit_days[0])
    n_rec = min(other.tracks.shape[1], res.tracks.shape[1])
    assert np.all(np.any(other.tracks[:, 1:n_rec] != res.tracks[:, 1:n_rec], axis=(1, 2)))

    # A voyage's outcome does not depend on what else shares its run, or the order.
    for j in (0, 5, 11):
        alone = simulate(env, CANOE, HoldCourse(), take(lau, [j]), det, cfg)
        assert_same(alone, res, [j])
    order = np.arange(12)[::-1]
    reordered = simulate(env, CANOE, HoldCourse(), take(lau, order), det, cfg)
    assert_same(reordered, res, order)


@pytest.mark.parametrize("integrator", ["euler", "midpoint"])
def test_random_walk_spreads_voyages_at_the_set_diffusivity(target_ahead, integrator):
    diffusivity, days = 1000.0, 10.0
    calm = Environment(UniformField(0.0, 0.0), UniformField(0.0, 0.0))
    res = simulate(calm, RAFT, Drift(), launches(n=2000), target_ahead,
                   SimConfig(integrator=integrator, max_days_at_sea=days, diffusivity_m2s=diffusivity, track_every=0))
    dx = (res.end_lon - START) * KM_PER_DEG * 1000.0
    dy = res.end_lat * KM_PER_DEG * 1000.0
    variance = 2 * diffusivity * days * 86400.0   # m^2 per axis
    # 2000 samples estimate a variance to ~3% (1 sd).
    np.testing.assert_allclose([dx.var(), dy.var()], variance, rtol=0.12)
    assert abs(dx.mean()) < 4 * np.sqrt(variance / 2000) and abs(dy.mean()) < 4 * np.sqrt(variance / 2000)
    assert abs(np.corrcoef(dx, dy)[0, 1]) < 0.1


# --- waiting in port ---------------------------------------------------------------

def test_wait_for_wind_waits_without_spending_supplies():
    switch = T0 + 96.0   # headwind until then, fair wind after
    env = Environment(FunctionField(lambda lon, lat, t: (np.where(t < switch, -8.0, 8.0), 0.0)),
                      UniformField(0.0, 0.0))
    centre = (east(START, 300.0), 0.0)
    det = detector([point("Target", *centre)])
    t0 = T0 + np.array([0.0, 1.0, 50.0, 101.0, 96.0 - 10 * 24.0, 96.0 - 11 * 24.0])
    strategy = WaitForWind(max_wait_days=10.0)
    # Two days of supplies, though the first voyages wait four days in port.
    res = simulate(env, CANOE, strategy, launches(t0=t0, target=centre), det,
                   SimConfig(max_days_at_sea=2.0, track_every=1, **QUIET))

    sailed = slice(0, 5)
    # Each leaves at its first step on or after the switch; the fourth launch is already fair,
    # and the fifth catches the switch at its last chance, ten days after launch.
    np.testing.assert_array_equal(res.depart_t[sailed], T0 + np.array([96.0, 97.0, 98.0, 101.0, 96.0]))
    assert np.all(res.status[sailed] == END_ALL_SIGHTED)
    # Same fair wind, same passage: days at sea count from departure, not launch.
    assert np.all(res.hit_days[0, sailed] == res.hit_days[0, 3])
    assert res.hit_days[0, 3] < 2.0
    np.testing.assert_allclose(res.end_t[sailed] - res.depart_t[sailed], res.days_at_sea[sailed] * 24.0)
    in_port = res.track_days < 4.0
    assert np.all(res.tracks[0, in_port, 0] == np.float32(START))

    # Launched eleven days before the switch: gives up after ten, never having sailed.
    assert res.status[5] == DID_NOT_SAIL and np.isnan(res.depart_t[5])
    assert res.end_t[5] == t0[5] + 240.0 and res.days_at_sea[5] == 0.0
    assert np.isnan(res.hit_days[0, 5]) and res.end_lon[5] == START


def test_wait_for_wind_with_no_time_to_wait_still_needs_a_fair_wind():
    centre = (east(START, 300.0), 0.0)
    det = detector([point("Target", *centre)])
    cfg = SimConfig(max_days_at_sea=0.5, track_every=0, **QUIET)
    headwind = Environment(UniformField(-8.0, 0.0), UniformField(0.0, 0.0))
    for max_wait_days in (0.0, 0.2):   # less than one 6-hour step: the launch is the only chance
        strategy = WaitForWind(max_wait_days=max_wait_days)
        stay = simulate(headwind, CANOE, strategy, launches(target=centre), det, cfg)
        assert stay.status[0] == DID_NOT_SAIL and np.isnan(stay.depart_t[0]) and stay.end_t[0] == T0
        go = simulate(WIND_FROM_WEST, CANOE, strategy, launches(target=centre), det, cfg)
        assert go.depart_t[0] == T0 and go.status[0] == END_HORIZON and go.days_at_sea[0] == 0.5


def test_strategies_without_waiting_sail_at_launch(target_ahead):
    res = simulate(WIND_FROM_WEST, CANOE, HoldCourse(), launches(target=(-140.0, 0.0)), target_ahead,
                   SimConfig(max_days_at_sea=0.25, track_every=0, **QUIET))
    assert res.depart_t[0] == T0 and res.days_at_sea[0] == 0.25 and res.end_lon[0] > START
    assert res.tracks is None and res.track_days is None


# --- coverage, home islands --------------------------------------------------------

@pytest.mark.parametrize("integrator", ["euler", "midpoint"])
def test_leaving_the_forcing_ends_the_voyage_where_it_is(target_ahead, integrator):
    cfg = SimConfig(integrator=integrator, max_days_at_sea=10.0, **QUIET)
    edge = -149.0
    # Current data end at 149 W.
    env = Environment(UniformField(0.0, 0.0),
                      FunctionField(lambda lon, lat, t: (np.where(lon < edge, DRIFT_MS, np.nan), 0.0)))
    res = simulate(env, RAFT, Drift(), launches(lon=[START, -148.0]), target_ahead, cfg)
    assert np.all(res.status == LEFT_DOMAIN)
    # It stops at the first position, or (midpoint) half-step position, outside the data.
    step_deg = 0.25 * KM_PER_DAY / KM_PER_DEG
    assert abs(res.end_lon[0] - edge) < step_deg
    assert res.days_at_sea[0] < 10.0 and res.end_t[0] == T0 + res.days_at_sea[0] * 24.0
    # Launched outside the data: ends at once, where it was launched.
    assert res.days_at_sea[1] == 0.0 and res.end_lon[1] == -148.0 and res.end_t[1] == T0

    # Wind data end a day after launch.
    env = Environment(FunctionField(lambda lon, lat, t: (np.where(t < T0 + 24.0, 0.0, np.nan), 0.0)),
                      UniformField(DRIFT_MS, 0.0))
    res = simulate(env, RAFT, Drift(), launches(), target_ahead, cfg)
    assert res.status[0] == LEFT_DOMAIN and res.end_t[0] == T0 + 24.0 and res.days_at_sea[0] == 1.0


def test_home_island_counts_only_after_it_has_dropped_out_of_sight():
    home = Landmass("Home", "Home", 100.0, [circle_ring(START, 0.0, 5.0)])
    det = detector([home, point("Away", east(START, 600.0))])
    # Out to sea for three days, then the current turns and brings the raft home.
    env = Environment(UniformField(0.0, 0.0),
                      FunctionField(lambda lon, lat, t: (np.where(t < T0 + 72.0, DRIFT_MS, -DRIFT_MS), 0.0)))
    launch_lon = east(START, 15.0)   # 10 km off the coast, well within the 40 km radius
    res = simulate(env, RAFT, Drift(), launches(lon=launch_lon, origin=[0, -1]), det,
                   SimConfig(max_days_at_sea=10.0, **QUIET))
    # From home: not landfall at launch; sighted again on the way back, 45 km from its centre.
    furthest = 15.0 + 3 * KM_PER_DAY
    assert res.hit_landmass[0, 0] == 0
    assert res.hit_days[0, 0] == pytest.approx(3 + (furthest - 45.0) / KM_PER_DAY, abs=1e-4)
    # With no home group the same island is "sighted" at once.
    assert res.hit_days[0, 1] == 0.0 and res.hit_landmass[0, 1] == 0 and res.hit_lon[0, 1] == launch_lon

    # A return leg, home its destination, sights it at the same moment. A voyage bound for Away
    # sails on, and records its home as other land once that has dropped out of sight and come back.
    legs = simulate(env, RAFT, Drift(), launches(lon=launch_lon, origin=0, dest=[[True, False], [False, True]], n=2),
                    det, SimConfig(max_days_at_sea=10.0, **QUIET))
    assert legs.hit_days[0, 0] == res.hit_days[0, 0] and legs.hit_landmass[0, 0] == 0
    assert np.isnan(legs.other_days[0]) and legs.other_landmass[0] == -1
    assert np.isnan(legs.hit_days[0, 1]) and legs.status[1] == END_HORIZON
    assert legs.other_days[1] == res.hit_days[0, 0] and legs.other_landmass[1] == 0
    assert legs.other_lon[1] == res.hit_lon[0, 0] and legs.other_lat[1] == 0.0


def test_only_destinations_end_a_voyage_and_other_land_is_recorded_on_the_way():
    # Along the equator: Reef 150 km out and 30 km north of the track, so in sight under u40 from
    # 123.5 km (never under u20); Isle 400 km out on the track, in sight from 360 km (u40), 380 km (u20).
    reef = point("Reef", east(START, 150.0), 30.0 / KM_PER_DEG)
    det = detector([reef, point("Isle", east(START, 400.0))], radii=(40.0, 20.0))
    assert det.groups == ["Reef", "Isle"]
    reef_km = 150.0 - np.sqrt(40.0**2 - 30.0**2)
    cfg = SimConfig(max_days_at_sea=12.0, **QUIET)

    # Without destinations any land counts: Reef under u40, Isle under u20.
    anywhere = simulate(CURRENT_EAST, RAFT, Drift(), launches(), det, cfg)
    np.testing.assert_array_equal(anywhere.hit_landmass[:, 0], [0, 1])
    assert anywhere.hit_days[0, 0] == pytest.approx(reef_km / KM_PER_DAY, abs=0.01)
    assert np.isnan(anywhere.other_days[0]) and anywhere.other_landmass[0] == -1

    # Bound for Isle: Reef is sailed past and recorded; bound for Reef: Isle comes after the
    # destination under u40, so it is not land on the way; bound nowhere: only the first other land.
    dest = np.array([[False, True], [True, False], [False, False]])
    res = simulate(CURRENT_EAST, RAFT, Drift(), launches(dest=dest, n=3), det, cfg)
    np.testing.assert_array_equal(res.hit_landmass, [[1, 0, -1], [1, -1, -1]])
    np.testing.assert_allclose(res.hit_days[:, 0], np.array([360.0, 380.0]) / KM_PER_DAY, atol=1e-4)
    assert res.hit_days[0, 1] == anywhere.hit_days[0, 0] and np.isnan(res.hit_days[1, 1])
    np.testing.assert_array_equal(res.status, [END_ALL_SIGHTED, END_HORIZON, END_HORIZON])
    np.testing.assert_array_equal(res.other_landmass, [0, -1, 0])
    np.testing.assert_allclose(res.other_days[[0, 2]], reef_km / KM_PER_DAY, atol=0.01)
    assert np.isnan(res.other_days[1]) and res.other_days[0] == res.other_days[2]
    np.testing.assert_allclose(res.other_lon[[0, 2]], east(START, reef_km), atol=0.01)
    assert np.all(res.other_lat[[0, 2]] == 0.0) and np.isnan(res.other_lon[1])
    # Destinations do not change tracks, and each voyage's outcome is its own.
    assert np.all(res.end_lon[1:] == res.end_lon[1])
    assert_same(simulate(CURRENT_EAST, RAFT, Drift(), take(launches(dest=dest, n=3), [2]), det, cfg), res, [2])

    # Within one day-long step (86-130 km): North comes into sight at 93.5 km, South takes over as
    # the nearer from about 122 km. Other land seen after the destination in the same step is not
    # on the way; seen before it, it is.
    pair = detector([point("North", east(START, 120.0), 30.0 / KM_PER_DEG),
                     point("South", east(START, 124.0), -30.0 / KM_PER_DEG)])
    res = simulate(CURRENT_EAST, RAFT, Drift(), launches(dest=[[True, False], [False, True]], n=2), pair,
                   SimConfig(dt_hours=24.0, max_days_at_sea=5.0, **QUIET))
    north_km = 120.0 - np.sqrt(40.0**2 - 30.0**2)
    np.testing.assert_array_equal(res.hit_landmass[0], [0, 1])
    assert res.hit_days[0, 0] == pytest.approx(north_km / KM_PER_DAY, abs=0.01)
    assert res.hit_days[0, 1] == pytest.approx(122.0 / KM_PER_DAY, abs=3.5 / KM_PER_DAY)
    np.testing.assert_array_equal(res.other_landmass, [-1, 0])
    assert np.isnan(res.other_days[0]) and res.other_days[1] == res.hit_days[0, 0]


def test_other_land_is_what_the_complementary_destinations_would_sight_first():
    # Tracks do not depend on destinations, so a voyage's first other land is the first rule-0
    # sighting of a run bound for every other group, whenever that comes before its destination.
    rng = np.random.default_rng(0)
    isles = [point(f"I{i}", lon, lat, group=f"G{i % 4}")
             for i, (lon, lat) in enumerate(zip(rng.uniform(-148.5, -143.0, 10), rng.uniform(-2.5, 2.5, 10)))]
    det = detector(isles, radii=(40.0, 20.0))
    dest = rng.random((60, 4)) < 0.4
    lau = launches(lat=np.linspace(-3.0, 3.0, 60), dest=dest)
    env = Environment(UniformField(0.0, 0.0), FunctionField(eddies))
    cfg = SimConfig(max_days_at_sea=15.0, diffusivity_m2s=300.0, track_every=0)
    res = simulate(env, RAFT, Drift(), lau, det, cfg)
    rest = simulate(env, RAFT, Drift(), dataclasses.replace(lau, dest_groups=~dest), det, cfg)
    first = rest.hit_days[0]
    before = first < np.where(np.isnan(res.hit_days[0]), np.inf, res.hit_days[0])
    assert before.sum() >= 10 and (np.isfinite(first) & ~before).sum() >= 3   # both cases occur
    np.testing.assert_array_equal(res.other_days, np.where(before, first, np.nan))
    np.testing.assert_array_equal(res.other_landmass, np.where(before, rest.hit_landmass[0], -1))
    np.testing.assert_array_equal(res.other_lon, np.where(before, rest.hit_lon[0], np.nan))
    np.testing.assert_array_equal(res.other_lat, np.where(before, rest.hit_lat[0], np.nan))


# --- step size, several rules ------------------------------------------------------

def test_six_hour_steps_agree_with_one_hour_steps():
    centre = (-141.0, 1.0)
    coast = Landmass("Coast", "Coast", 100.0, [np.array([[-138.0, -6.0], [-138.0, 6.0]])], kind="coast")
    det = detector([Landmass("Target", "Target", 100.0, [circle_ring(*centre, 8.0)]), coast])
    env = Environment(FunctionField(swirl), FunctionField(eddies))
    lau = launches(lat=np.linspace(-2.0, 2.0, 16), t0=T0 + 9.0 * np.arange(16), target=centre)
    raft = Vessel("raft", "drift", Leeway(0.03, cw_slope=0.01))
    for vessel, strategy in ((CANOE, HoldCourse()), (raft, Drift())):
        days = {}
        for dt, integrator in ((1.0, "midpoint"), (6.0, "midpoint"), (6.0, "euler")):
            cfg = SimConfig(dt_hours=dt, integrator=integrator, max_days_at_sea=40.0, track_every=0, **QUIET)
            days[dt, integrator] = simulate(env, vessel, strategy, lau, det, cfg).hit_days[0]
        ref = days[1.0, "midpoint"]
        assert np.isfinite(ref).sum() >= 12
        err = {key: np.abs(d / ref - 1) for key, d in days.items()}
        np.testing.assert_array_equal(np.isnan(days[6.0, "midpoint"]), np.isnan(ref))
        assert np.nanmax(err[6.0, "midpoint"]) < 0.02
        # Second order: six-hour midpoint steps beat six-hour Euler steps.
        assert np.nanmean(err[6.0, "midpoint"]) < np.nanmean(err[6.0, "euler"])


def test_a_larger_sighting_radius_never_sights_later():
    det = detector([Landmass("Target", "Target", 100.0, [circle_ring(east(START, 300.0), 0.0, 5.0)])],
                   radii=(20.0, 40.0, 60.0))
    lau = launches(lat=np.linspace(-0.8, 0.8, 41))
    res = simulate(CURRENT_EAST, RAFT, Drift(), lau, det,
                   SimConfig(max_days_at_sea=12.0, diffusivity_m2s=200.0, **QUIET))
    seen = ~np.isnan(res.hit_days)
    for small, large in ((0, 1), (1, 2)):
        assert np.all(seen[large] | ~seen[small])
        both = seen[small] & seen[large]
        assert np.all(res.hit_days[large, both] <= res.hit_days[small, both])
    # Voyages that passed within 60 but not 20 km sail on to the horizon.
    patterns = {tuple(col) for col in seen.T}
    assert {(True, True, True), (False, True, True), (False, False, True), (False, False, False)} <= patterns
    np.testing.assert_array_equal(res.status == END_ALL_SIGHTED, seen.all(axis=0))
    assert np.all(res.days_at_sea[~seen.all(axis=0)] == 12.0)


class DueEast(Strategy):
    """Steer due east, whatever happens."""

    name = "due_east"

    def _choose(self, state, ctx):
        return np.full(np.shape(ctx.lon), 90.0)


def test_steering_error_spread_does_not_depend_on_the_time_step():
    # Due east on a beam reach in a steady north wind, on a boat that makes 6 knots at any angle to
    # it: steering error alone moves voyages off the equator, as a random walk across the course.
    boat = Vessel("boat", "sailing", Leeway(0.0), parametric_polar(20, 6.0, 1.0, [(20, 1.0), (180, 1.0)]),
                  min_twa_deg=20)
    env = Environment(UniformField(0.0, -8.0), UniformField(0.0, 0.0))
    det = detector([point("Behind", -151.9, 3.9)])
    spread = {}
    for dt in (1.0, 6.0):
        cfg = SimConfig(dt_hours=dt, max_days_at_sea=5.0, speed_sd=0.0, track_every=0)
        res = simulate(env, boat, DueEast(), launches(n=1000), det, cfg)
        assert np.all(res.status == END_HORIZON)
        spread[dt] = res.end_lat.std() * KM_PER_DEG
    # 20 six-hour steps of 66.7 km, each 10 deg off course (1 sd): about sqrt(20) * 66.7 km * sin(10 deg).
    step_km = 6.0 * geo.KNOT_MS * 6 * 3.6
    assert spread[6.0] == pytest.approx(np.sqrt(20) * step_km * np.radians(10.0), rel=0.1)
    # Without the sqrt(6 / dt) scaling, one-hour steps would spread voyages sqrt(6) times less.
    assert spread[1.0] == pytest.approx(spread[6.0], rel=0.2)


@pytest.mark.parametrize("integrator, samples_per_step", [("euler", 1), ("midpoint", 2)])
def test_wind_above_the_cap_is_slowed_to_it_keeping_its_direction(target_ahead, integrator, samples_per_step):
    # A spurious 100 m/s wind toward the north-east, capped at 50 m/s: the raft drifts exactly as in
    # a true 50 m/s wind from the same direction.
    raft = Vessel("raft", "drift", Leeway(0.03))
    cfg = SimConfig(integrator=integrator, max_days_at_sea=1.0, track_every=0, **QUIET)

    def run(wind, cap):
        env = Environment(UniformField(*wind), UniformField(0.0, 0.0), wind_cap_ms=cap)
        return simulate(env, raft, Drift(), launches(n=3), target_ahead, cfg)

    capped, true_50, uncapped = run((60.0, 80.0), 50.0), run((30.0, 40.0), None), run((60.0, 80.0), None)
    np.testing.assert_array_equal(capped.end_lon, true_50.end_lon)
    np.testing.assert_array_equal(capped.end_lat, true_50.end_lat)
    assert np.all(uncapped.end_lat > 1.9 * capped.end_lat)
    # Three voyages read the wind once in port, then once (Euler) or twice (midpoint) in each of four steps.
    assert capped.meta["n_wind_capped"] == 3 * (1 + 4 * samples_per_step) and capped.meta["wind_cap_ms"] == 50.0
    assert true_50.meta["n_wind_capped"] == uncapped.meta["n_wind_capped"] == 0
    # Below the cap nothing changes, and an infinite cap is no cap.
    np.testing.assert_array_equal(run((30.0, 40.0), 50.0).end_lat, true_50.end_lat)
    infinite = run((60.0, 80.0), np.inf)
    np.testing.assert_array_equal(infinite.end_lat, uncapped.end_lat)
    assert np.all(infinite.status == uncapped.status) and infinite.meta["n_wind_capped"] == 0


# --- results -----------------------------------------------------------------------

def test_results_frame_and_save_load(tmp_path, target_ahead):
    det = detector([point("Target", east(START, 300.0), group="Isles"), point("Other", -140.0, 2.0)],
                   radii=(20.0, 40.0))
    lau = launches(lat=[0.0, 0.3, 3.0])
    res = simulate(CURRENT_EAST, RAFT, Drift(), lau, det, SimConfig(max_days_at_sea=8.0))
    df = res.to_frame()
    assert len(df) == 3 and list(df["voyage_id"]) == [0, 1, 2]
    assert list(df["status"]) == ["all_sighted", "horizon", "horizon"]
    assert list(df["hit_landmass_u40"][:2]) == ["Target", "Target"] and pd.isna(df["hit_landmass_u40"][2])
    assert list(df["hit_group_u40"][:2]) == ["Isles", "Isles"] and pd.isna(df["hit_group_u40"][2])
    assert pd.isna(df["hit_landmass_u20"][1]) and np.isnan(df["hit_days_u20"][1])
    np.testing.assert_array_equal(df["hit_days_u40"], res.hit_days[1])

    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    res.save(a)
    simulate(CURRENT_EAST, RAFT, Drift(), lau, det, SimConfig(max_days_at_sea=8.0)).save(b)
    assert a.read_bytes() == b.read_bytes()
    back = VoyageResults.load(a)
    assert_same(back, res)
    for name in ("launch_t", "track_days"):
        np.testing.assert_array_equal(getattr(back, name), getattr(res, name))
    assert back.rule_names == ["u20", "u40"] and back.landmass_groups == ["Isles", "Other"]
    assert back.group_names == ["Isles", "Other"] and back.current_is_climatology is False
    assert back.meta == res.meta and back.meta["vessel"]["heave_to_tws_kn"] == np.inf
    assert back.meta["strategy"] == {"name": "drift"} and back.meta["config"]["max_days_at_sea"] == 8.0

    untracked = simulate(CURRENT_EAST, RAFT, Drift(), lau, det, SimConfig(max_days_at_sea=1.0, track_every=0))
    untracked.save(a)
    assert VoyageResults.load(a).tracks is None

    # Bound for Other only: Target, in sight of the first voyage under u20, is other land.
    assert np.all(np.isnan(res.other_days)) and df["other_landmass"].isna().all()
    bound = simulate(CURRENT_EAST, RAFT, Drift(), launches(lat=[0.0, 0.3, 3.0], dest=[False, True]), det,
                     SimConfig(max_days_at_sea=8.0))
    df = bound.to_frame()
    assert df["other_landmass"][0] == "Target" and df["other_group"][0] == "Isles"
    assert df["other_landmass"][1:].isna().all() and df["other_group"][1:].isna().all()
    assert df["other_days"][0] == pytest.approx(280.0 / KM_PER_DAY, abs=1e-4)
    np.testing.assert_array_equal(df["other_lon"], bound.other_lon)
    np.testing.assert_array_equal(df["other_lat"], bound.other_lat)
    bound.save(a)
    assert_same(VoyageResults.load(a), bound)


def test_input_validation(target_ahead):
    with pytest.raises(ValueError):
        SimConfig(integrator="rk4")
    with pytest.raises(ValueError):
        SimConfig(track_every=1.5)
    with pytest.raises(ValueError, match="unique"):
        Launches([1, 1], START, 0.0, T0, -1, np.nan, np.nan)
    with pytest.raises(TypeError):
        Launches([1], START, 0.0, np.datetime64("2000-01-01"), -1, np.nan, np.nan)
    with pytest.raises(ValueError, match="origin_group"):
        simulate(CURRENT_EAST, RAFT, Drift(), launches(origin=1), target_ahead, SimConfig())   # one group only
    lau = launches(n=3)   # scalars broadcast to every voyage
    assert len(lau) == 3 and np.all(lau.lon == START) and np.all(np.isnan(lau.target_lat))
    assert lau.dest_groups is None
    np.testing.assert_array_equal(launches(dest=[False, True], n=3).dest_groups, [[False, True]] * 3)
    with pytest.raises(ValueError, match="dest_groups"):
        launches(dest=[[0, 1]] * 3, n=3)   # not bool
    with pytest.raises(ValueError, match="dest_groups"):
        launches(dest=[[False, True]] * 2, n=3)
    with pytest.raises(ValueError, match="dest_groups"):
        simulate(CURRENT_EAST, RAFT, Drift(), launches(dest=[True, False]), target_ahead, SimConfig())
