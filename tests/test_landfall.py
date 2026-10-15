import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from voyage_sim import geo
from voyage_sim.landfall import (
    LandfallDetector,
    Landmass,
    SightingRaster,
    SightingRule,
    attach_gshhg,
    circle_ring,
    geometric_range_km,
    load_catalog,
)

KM_PER_DEG = np.radians(geo.EARTH_RADIUS_KM)
BOUNDS = (-150.0, -144.0, -20.0, -14.0)
ATOLL_LON, ATOLL_LAT, ATOLL_R = -147.0, -17.0, 3.0   # atoll rim: a 3 km circle


def offset(lon, lat, east_km, north_km):
    """Point displaced east/north by km on the local tangent plane."""
    return lon + east_km / (KM_PER_DEG * np.cos(np.radians(lat))), lat + north_km / KM_PER_DEG


def rim_km(lon, lat):
    """Exact great-circle distance to the atoll's rim."""
    return np.abs(geo.haversine_km(ATOLL_LON, ATOLL_LAT, lon, lat) - ATOLL_R)


def atoll():
    return Landmass("Atoll", "Tuamotu", 25.0, [circle_ring(ATOLL_LON, ATOLL_LAT, ATOLL_R)], kind="atoll")


def along(lon0, lat0, lon1, lat1, frac):
    return lon0 + frac * (lon1 - lon0), lat0 + frac * (lat1 - lat0)


@pytest.fixture(scope="module")
def atoll_rasters():
    rules = [SightingRule(f"u{r}", mode="uniform", radius_km=r) for r in (20, 26, 60)]
    return {r.name: SightingRaster.build([atoll()], r, BOUNDS, res_deg=0.02, cap_km=100.0) for r in rules}


# --- sighting radii -------------------------------------------------------------

def test_geometric_range_known_values():
    # 3 m eye, standard refraction: a sea-level target, a 25 m atoll canopy, a 1000 m peak.
    np.testing.assert_allclose(geometric_range_km([0.0, 25.0, 1000.0]), [6.63, 25.76, 127.65], atol=0.01)
    # Without refraction and with the eye at the waterline it is the plain horizon distance sqrt(2 R h).
    assert geometric_range_km(100.0, eye_height_m=0.0, refraction_k=0.0) == pytest.approx(
        np.sqrt(2 * geo.EARTH_RADIUS_KM * 0.1))
    # Refraction stretches every range by 1 / sqrt(1 - k).
    ratio = geometric_range_km([25.0, 1000.0], 3.0, 0.13) / geometric_range_km([25.0, 1000.0], 3.0, 0.0)
    np.testing.assert_allclose(ratio, 1 / np.sqrt(0.87))


def test_sighting_rules():
    heights = np.array([0.0, 25.0, 1000.0])
    kinds = ["coast", "atoll", "island"]
    np.testing.assert_array_equal(SightingRule("u40", mode="uniform").radius_km_for(heights, kinds), 40.0)
    np.testing.assert_allclose(SightingRule("h").radius_km_for(heights, kinds), geometric_range_km(heights))
    # Scaled by half: [3.3, 12.9, 63.8] km, then floored at 10 and capped at 60.
    rule = SightingRule("h", scale=0.5, floor_km=10.0, cap_km=60.0)
    np.testing.assert_allclose(rule.radius_km_for(heights, kinds), [10.0, 0.5 * geometric_range_km(25.0), 60.0])
    with pytest.raises(ValueError):
        SightingRule("bad", mode="visual")
    with pytest.raises(ValueError):
        SightingRule("bad", floor_km=50.0, cap_km=20.0)
    with pytest.raises(ValueError):
        SightingRule("bad", bird_km=-1.0)
    with pytest.raises(ValueError, match="kinds"):
        rule.radius_km_for(heights, ["coast", "atoll", "reef"])


def test_bird_range_extends_islands_and_atolls_but_not_coasts():
    # An 18 m atoll canopy clears the horizon at 22.9 km; its birds are seen out to 37 km.
    assert geometric_range_km(18.0) == pytest.approx(22.87, abs=0.01)
    heights = np.array([18.0, 18.0, 18.0, 1000.0])
    kinds = np.array(["atoll", "island", "coast", "island"])
    birds = SightingRule("h", bird_km=37.0)
    np.testing.assert_allclose(birds.radius_km_for(heights, kinds),
                               [37.0, 37.0, geometric_range_km(18.0), geometric_range_km(1000.0)])
    # Scaling applies to the bird range too.
    np.testing.assert_allclose(SightingRule("h", bird_km=37.0, scale=0.5).radius_km_for(heights, kinds),
                               [18.5, 18.5, 0.5 * geometric_range_km(18.0), 0.5 * geometric_range_km(1000.0)])
    # Then the floor and cap, as for visual ranges.
    np.testing.assert_allclose(SightingRule("h", bird_km=37.0, floor_km=30.0, cap_km=100.0).radius_km_for(
        heights, kinds), [37.0, 37.0, 30.0, 100.0])
    # Uniform radii ignore height, kind and birds; a scalar kind serves every landmass.
    np.testing.assert_array_equal(
        SightingRule("u20", mode="uniform", radius_km=20.0, bird_km=37.0).radius_km_for(heights, kinds), 20.0)
    np.testing.assert_allclose(birds.radius_km_for(heights, "coast"), geometric_range_km(heights))


def test_raster_uses_each_landmass_kind():
    # An atoll and a stretch of coast of the same height: only the atoll gets the bird range.
    reef = Landmass("Reef", "Reefs", 18.0, [[(-148.0, -17.0)]], kind="atoll")
    coast = Landmass("Shore", "Shore", 18.0, [[(-145.0, -19.0), (-145.0, -15.0)]], kind="coast")
    raster = SightingRaster.build([reef, coast], SightingRule("h", bird_km=37.0), BOUNDS, res_deg=0.05,
                                  cap_km=100.0)
    np.testing.assert_allclose(raster.radius_km, [37.0, geometric_range_km(18.0)])
    margin, index = raster.query(*offset(-148.0, -17.0, -50.0, 0.0))
    assert index == 0 and margin == pytest.approx(13.0, abs=0.5)
    margin, index = raster.query(*offset(-145.0, -17.0, -30.0, 0.0))
    assert index == 1 and margin == pytest.approx(30.0 - geometric_range_km(18.0), abs=0.5)


def test_circle_ring_is_closed_at_exact_distance():
    ring = circle_ring(-140.0, -20.0, 10.0, n=32)
    assert ring.shape == (33, 2) and np.array_equal(ring[0], ring[-1])
    np.testing.assert_allclose(geo.haversine_km(-140.0, -20.0, ring[:, 0], ring[:, 1]), 10.0, rtol=1e-9)


def test_landmass_validation():
    lm = Landmass("P", "G", 10.0, [[200.0, -10.0]])   # a single point, lon given in 0..360
    assert lm.rings[0].shape == (1, 2) and lm.rings[0][0, 0] == pytest.approx(-160.0)
    with pytest.raises(ValueError):
        Landmass("P", "G", 10.0, [[(-140.0, -10.0)]], kind="reef")
    with pytest.raises(ValueError):
        Landmass("P", "G", -1.0, [[(-140.0, -10.0)]])
    with pytest.raises(ValueError):
        Landmass("P", "G", 10.0, [np.zeros((3, 3))])


# --- rasters --------------------------------------------------------------------

@pytest.mark.parametrize("res", [0.02, 0.05])
def test_raster_margin_matches_analytic_distance(res):
    raster = SightingRaster.build([atoll()], SightingRule("u26", mode="uniform", radius_km=26.0), BOUNDS,
                                  res_deg=res, cap_km=100.0)
    assert raster.margin_km.dtype == np.float32
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(*BOUNDS[:2], 20_000), rng.uniform(*BOUNDS[2:], 20_000)
    margin, index = raster.query(lon, lat)
    truth = np.minimum(rim_km(lon, lat) - 26.0, 100.0)
    # Away from the rim's sharp curvature and the kink where the cap takes over.
    band = (rim_km(lon, lat) > 10.0) & (truth < 100.0 - 2 * raster.cell_km)
    assert np.abs(margin[band] - truth[band]).max() < 1.0
    beyond = rim_km(lon, lat) > 126.0 + 2 * raster.cell_km
    np.testing.assert_allclose(margin[beyond], 100.0)
    assert np.all(index[beyond] == -1)
    assert np.all(index[truth < 50.0] == 0)


def test_raster_index_is_argmin_of_margin_not_of_distance():
    low = Landmass("Low", "A", 5.0, [[(-147.0, -17.0)]])     # sighted from ~15 km
    high = Landmass("High", "B", 1000.0, [[(-145.0, -17.0)]])  # sighted from ~128 km, 213 km east of Low
    rule = SightingRule("h")
    raster = SightingRaster.build([low, high], rule, BOUNDS, res_deg=0.05, cap_km=150.0)
    radius = rule.radius_km_for([5.0, 1000.0], "island")
    lon, lat = offset(-147.0, -17.0, np.array([30.0, 60.0]), 0.0)
    margins = np.stack([geo.haversine_km(lm.rings[0][0, 0], lm.rings[0][0, 1], lon, lat) - r
                        for lm, r in zip((low, high), radius)])
    margin, index = raster.query(lon, lat)
    # 30 km from Low it governs; at 60 km the peak, 2.5 times as far away, is nearer to coming into sight.
    np.testing.assert_array_equal(index, [0, 1])
    np.testing.assert_allclose(margin, margins.min(axis=0), atol=0.5)


def test_coarse_coastline_is_densified():
    # A two-vertex coast 445 km long: a point abeam its middle is near the coast, not its ends.
    coast = Landmass("Coast", "Coast", 0.0, [[(-147.0, -19.0), (-147.0, -15.0)]], kind="coast")
    raster = SightingRaster.build([coast], SightingRule("u20", mode="uniform", radius_km=20.0), BOUNDS,
                                  res_deg=0.05, cap_km=100.0)
    lon, lat = offset(-147.0, -17.0, 30.0, 0.0)
    margin, index = raster.query(lon, lat)
    assert margin == pytest.approx(10.0, abs=0.5) and index == 0


def test_raster_matches_brute_force_at_high_latitude():
    # Where meridians converge the per-landmass search box must still cover every cell within reach.
    coast = Landmass("Coast", "Chile", 0.0, [[(-75.0, -52.0), (-74.0, -46.0), (-73.5, -40.0), (-72.0, -36.0)]],
                     kind="coast")
    peak = Landmass("Peak", "Islas", 1000.0, [[(-80.0, -45.0)]])
    rule, bounds, cap = SightingRule("h"), (-90.0, -62.0, -58.0, -30.0), 150.0
    raster = SightingRaster.build([coast, peak], rule, bounds, res_deg=0.1, cap_km=cap)
    ny, nx = raster.margin_km.shape
    glon, glat = np.meshgrid(bounds[0] + (np.arange(nx) + 0.5) * 0.1, bounds[2] + (np.arange(ny) + 0.5) * 0.1)
    # Reference: every cell against the coast sampled every ~0.2 km, and the peak's exact distance.
    ring = coast.rings[0]
    s = np.linspace(0.0, 1.0, 4000)[:, None]
    pts = np.vstack([a + s * (b - a) for a, b in zip(ring[:-1], ring[1:])])
    radius = rule.radius_km_for([0.0, 1000.0], ["coast", "island"])
    chord, _ = cKDTree(geo.lonlat_to_xyz(pts[:, 0], pts[:, 1])).query(
        geo.lonlat_to_xyz(glon, glat), distance_upper_bound=float(geo.km_to_chord(2 * cap)))
    to_coast = np.where(np.isfinite(chord), geo.chord_to_km(chord), np.inf) - radius[0]
    to_peak = geo.haversine_km(-80.0, -45.0, glon, glat) - radius[1]
    truth = np.minimum(np.minimum(to_coast, to_peak), cap)
    assert np.abs(raster.margin_km - truth).max() < 0.6   # densify_km / 2, plus float32
    clear = np.abs(to_coast - to_peak) > 1.0
    want = np.where(truth >= cap, -1, np.where(to_coast < to_peak, 0, 1))
    np.testing.assert_array_equal(raster.index[clear], want[clear])


def test_query_outside_bounds_or_nonfinite():
    # A raster tight around the atoll, so its edge cells are in sight and clamping to them would show.
    bounds = (-147.2, -146.8, -17.2, -16.8)
    raster = SightingRaster.build([atoll()], SightingRule("u26", mode="uniform", radius_km=26.0), bounds,
                                  res_deg=0.02, cap_km=100.0)
    # The middle of each edge, and a step of 1e-6 deg outwards.
    lon = np.array([bounds[0], bounds[1], ATOLL_LON, ATOLL_LON])
    lat = np.array([ATOLL_LAT, ATOLL_LAT, bounds[2], bounds[3]])
    d_lon, d_lat = 1e-6 * np.array([-1, 1, 0, 0]), 1e-6 * np.array([0, 0, -1, 1])
    margin, index = raster.query(lon - d_lon, lat - d_lat)
    assert np.all(margin < 0) and np.all(index == 0)
    margin, index = raster.query(np.append(lon + d_lon, [np.nan, ATOLL_LON]),
                                 np.append(lat + d_lat, [ATOLL_LAT, np.nan]))
    np.testing.assert_array_equal(margin, raster.cap_km)
    np.testing.assert_array_equal(index, -1)
    with pytest.raises(ValueError, match="-180"):   # longitudes in 0..360 would miss every landmass
        SightingRaster.build([atoll()], SightingRule("u26", mode="uniform", radius_km=26.0),
                             (210.0, 216.0, -20.0, -14.0))


def test_save_load_roundtrip(atoll_rasters, tmp_path):
    raster = atoll_rasters["u26"]
    path = tmp_path / "u26.npz"
    raster.save(path)
    loaded = SightingRaster.load(path)
    np.testing.assert_array_equal(loaded.margin_km, raster.margin_km)
    np.testing.assert_array_equal(loaded.index, raster.index)
    np.testing.assert_array_equal(loaded.radius_km, raster.radius_km)
    assert loaded.margin_km.dtype == np.float32 and loaded.index.dtype == raster.index.dtype
    assert (loaded.bounds, loaded.res_deg, loaded.cap_km) == (raster.bounds, raster.res_deg, raster.cap_km)
    assert (loaded.rule_name, loaded.landmass_names) == ("u26", ("Atoll",))
    lon, lat = np.linspace(-149, -145, 50), np.linspace(-18, -16, 50)
    for a, b in zip(loaded.query(lon, lat), raster.query(lon, lat)):
        np.testing.assert_array_equal(a, b)


def test_build_many_islands_quickly():
    rng = np.random.default_rng(7)
    centres = np.column_stack([rng.uniform(-158, -122, 200), rng.uniform(-38, -2, 200)])
    lms = [Landmass(f"i{k}", f"g{k % 10}", rng.choice([5.0, 25.0, 500.0]),
                    [circle_ring(lon, lat, rng.uniform(0.5, 3.0), n=16)], kind="atoll")
           for k, (lon, lat) in enumerate(centres)]
    start = time.perf_counter()
    raster = SightingRaster.build(lms, SightingRule("h"), (-160.0, -120.0, -40.0, 0.0), res_deg=0.1)
    assert time.perf_counter() - start < 10.0
    assert raster.margin_km.shape == (400, 400)
    margin, index = raster.query(centres[:, 0], centres[:, 1])
    assert np.all(margin < 0) and np.all(index >= 0)


# --- detection ------------------------------------------------------------------

def test_atoll_passed_between_steps_is_sighted(atoll_rasters):
    detector = LandfallDetector([atoll()], [atoll_rasters["u26"]])
    # Ends 40 km off the rim, passing 5 km off it midway.
    half = np.sqrt((ATOLL_R + 40.0) ** 2 - (ATOLL_R + 5.0) ** 2)
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, -half, ATOLL_R + 5.0)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, half, ATOLL_R + 5.0)
    np.testing.assert_allclose(rim_km(np.array([lon0, lon1]), np.array([lat0, lat1])), 40.0, atol=0.5)
    hit, frac, landmass, _ = detector.check_segment([lon0], [lat0], [lon1], [lat1], [-1], [True], [True])
    assert hit[0, 0] and landmass[0, 0] == 0
    lon, lat = along(lon0, lat0, lon1, lat1, frac[0, 0])
    assert rim_km(lon, lat) == pytest.approx(26.0, abs=1.0)
    assert lon < ATOLL_LON   # on the approach, not the departure


def test_sightings_match_brute_force(atoll_rasters):
    detector = LandfallDetector([atoll()], [atoll_rasters["u26"]])
    rng = np.random.default_rng(3)
    n = 3000
    east, north = rng.uniform(-100.0, 100.0, (2, n))
    length, bearing = rng.uniform(0.0, 150.0, n), rng.uniform(0.0, 2 * np.pi, n)
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, east, north)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, east + length * np.sin(bearing), north + length * np.cos(bearing))
    hit, frac, landmass, _ = detector.check_segment(lon0, lat0, lon1, lat1, np.full(n, -1),
                                                    np.ones(n, bool), np.ones(n, bool))
    t = np.linspace(0.0, 1.0, 801)
    dist = rim_km(*along(lon0[:, None], lat0[:, None], lon1[:, None], lat1[:, None], t))
    closest = dist.min(axis=1)
    clear = np.abs(closest - 26.0) > 1.0
    assert 300 < hit[0].sum() < n - 300
    np.testing.assert_array_equal(hit[0, clear], closest[clear] < 26.0)
    np.testing.assert_array_equal(landmass[0], np.where(hit[0], 0, -1))
    entering = hit[0] & (dist[:, 0] > 27.0)
    lon, lat = along(lon0[entering], lat0[entering], lon1[entering], lat1[entering], frac[0, entering])
    assert np.abs(rim_km(lon, lat) - 26.0).max() < 1.0


def test_long_step_over_small_target_is_not_skipped():
    # A low target seen from 5 km: the step's ends are 70 km out of sight but its middle runs over the rock.
    rock = Landmass("Rock", "R", 0.0, [[(ATOLL_LON, ATOLL_LAT)]])
    raster = SightingRaster.build([rock], SightingRule("u5", mode="uniform", radius_km=5.0), BOUNDS,
                                  res_deg=0.02, cap_km=100.0)
    detector = LandfallDetector([rock], [raster])
    abeam = np.array([0.0, 2.0, 8.0, 12.0])
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, np.full(4, -75.0), abeam)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, np.full(4, 75.0), abeam)
    hit, frac, _, _ = detector.check_segment(lon0, lat0, lon1, lat1, np.full(4, -1), np.ones(4, bool),
                                             np.ones(4, bool))
    np.testing.assert_array_equal(hit[0], [True, True, False, False])
    lon, lat = along(lon0[:2], lat0[:2], lon1[:2], lat1[:2], frac[0, :2])
    np.testing.assert_allclose(geo.haversine_km(ATOLL_LON, ATOLL_LAT, lon, lat), 5.0, atol=1.0)


def test_far_segments_are_skipped(atoll_rasters, monkeypatch):
    raster = atoll_rasters["u26"]
    detector = LandfallDetector([atoll()], [raster])
    queried = []
    query = raster.query

    def counting_query(lon, lat):
        queried.append(np.size(lon))
        return query(lon, lat)

    monkeypatch.setattr(raster, "query", counting_query)
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, np.array([-40.0, 0.0, 150.0]), np.array([150.0, -160.0, 0.0]))
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, np.array([40.0, 80.0, 150.0]), np.array([150.0, -160.0, 80.0]))
    hit, _, _, armed = detector.check_segment(lon0, lat0, lon1, lat1, np.full(3, -1), np.zeros(3, bool),
                                              np.ones(3, bool))
    assert sum(queried) == 6   # the two endpoints of each segment, nothing sampled in between
    assert not hit.any() and armed.all()


def test_home_exclusion_and_arming():
    # Two home islands 100 km apart with 20 km sighting zones, and a foreign island well away.
    h1 = (-147.0, -17.0)
    h2 = offset(*h1, 100.0, 0.0)
    lms = [Landmass("H1", "Home", 10.0, [circle_ring(*h1, 2.0)]),
           Landmass("H2", "Home", 10.0, [circle_ring(*h2, 2.0)]),
           Landmass("F", "Foreign", 10.0, [circle_ring(-145.0, -15.0, 2.0)])]
    raster = SightingRaster.build(lms, SightingRule("u20", mode="uniform", radius_km=20.0), BOUNDS,
                                  res_deg=0.02, cap_km=100.0)
    detector = LandfallDetector(lms, [raster])
    assert detector.groups == ["Home", "Foreign"]
    np.testing.assert_array_equal(detector.landmass_group, [0, 0, 1])

    def east(km):
        lon, lat = offset(*h1, np.asarray(km, dtype=float), 0.0)
        return lon, np.full_like(lon, lat)

    # a: stays inside the home zone; b: leaves it; c: leaves H1's zone and enters H2's in one step;
    # d: launched from the home zone by a voyage from elsewhere.
    lon0, lat0 = east([7.0, 7.0, 7.0, 7.0])
    lon1, lat1 = east([17.0, 62.0, 88.0, 17.0])
    hit, frac, landmass, armed = detector.check_segment(lon0, lat0, lon1, lat1, [0, 0, 0, 1],
                                                        np.zeros(4, bool), np.ones(4, bool))
    np.testing.assert_array_equal(hit[0], [False, False, True, True])
    np.testing.assert_array_equal(armed[0], [False, True, True, False])
    np.testing.assert_array_equal(landmass[0], [-1, -1, 1, 0])
    lon, lat = along(lon0[2], lat0[2], lon1[2], lat1[2], frac[0, 2])
    assert geo.haversine_km(*h2, lon, lat) - 2.0 == pytest.approx(20.0, abs=1.0)
    assert frac[0, 3] == 0.0

    # Next step for b: coming back home after being out of sight is a sighting.
    hit2, frac2, landmass2, _ = detector.check_segment(lon1[1:2], lat1[1:2], lon0[1:2], lat0[1:2], [0],
                                                       armed[:, 1:2], [True])
    assert hit2[0, 0] and landmass2[0, 0] == 0
    lon, lat = along(lon1[1], lat1[1], lon0[1], lat0[1], frac2[0, 0])
    assert geo.haversine_km(*h1, lon, lat) - 2.0 == pytest.approx(20.0, abs=1.0)


def test_foreign_zone_overlapping_home_is_sighted_where_it_takes_over():
    # Home and foreign islands 30 km apart: their 21 km zones overlap, and the raster names the foreign
    # island from the midpoint on. Never out of sight, so no interpolation: frac is the first sample there.
    home = (-147.0, -17.0)
    foreign = offset(*home, 30.0, 0.0)
    lms = [Landmass("H", "Home", 10.0, [circle_ring(*home, 1.0)]),
           Landmass("F", "Foreign", 10.0, [circle_ring(*foreign, 1.0)])]
    raster = SightingRaster.build(lms, SightingRule("u20", mode="uniform", radius_km=20.0), BOUNDS,
                                  res_deg=0.02, cap_km=100.0)
    detector = LandfallDetector(lms, [raster])
    lon0, lat0 = offset(*home, np.array([5.0, 5.0]), np.zeros(2))
    lon1, lat1 = offset(*home, np.array([25.0, 25.0]), np.zeros(2))
    hit, frac, landmass, armed = detector.check_segment(lon0, lat0, lon1, lat1, [0, 0], [False, True],
                                                        [True, True])
    np.testing.assert_array_equal(hit[0], [True, True])
    np.testing.assert_array_equal(landmass[0], [1, 0])   # once armed, home counts from the start
    assert 13.5 < 5.0 + 20.0 * frac[0, 0] < 17.5 and frac[0, 1] == 0.0
    np.testing.assert_array_equal(armed[0], [False, True])


def test_only_eligible_groups_are_sighted():
    # Island A, 50 km east of the line's origin, and island B at 120 km: 2 km rims, 20 km zones, so
    # land is in sight from 28 to 72 km (A) and from 98 to 142 km (B) along the line.
    def east(km):
        lon, lat = offset(ATOLL_LON, ATOLL_LAT, np.asarray(km, dtype=float), 0.0)
        return lon, np.full_like(lon, lat)

    lms = [Landmass("A", "Isles A", 10.0, [circle_ring(*east(50.0), 2.0)]),
           Landmass("B", "Isles B", 10.0, [circle_ring(*east(120.0), 2.0)])]
    raster = SightingRaster.build(lms, SightingRule("u20", mode="uniform", radius_km=20.0), BOUNDS,
                                  res_deg=0.02, cap_km=100.0)
    detector = LandfallDetector(lms, [raster])
    # 0-3 sail 0 -> 150 km from nowhere: no destinations given, both, B only, neither.
    # 4-6 sail 40 -> 150 km from home A (inside its zone): a return leg not yet armed, the same
    # armed, and an onward voyage that may land anywhere but has not left home's sight yet.
    # 7 is a return leg sailing 100 -> 140 km, within sight of B all the way.
    start = np.array([0.0] * 4 + [40.0] * 3 + [100.0])
    lon0, lat0 = east(start)
    lon1, lat1 = east(np.array([150.0] * 7 + [140.0]))
    eligible = np.array([[True, True], [True, True], [False, True], [False, False],
                         [True, False], [True, False], [True, True], [True, False]])
    origin = np.array([-1] * 4 + [0] * 4)
    armed = np.array([True] * 4 + [False, True, False, False])
    hit, frac, landmass, armed_new = detector.check_segment(lon0, lat0, lon1, lat1, origin, armed,
                                                            np.ones(8, bool), eligible=eligible)
    full = detector.check_segment(lon0[:1], lat0[:1], lon1[:1], lat1[:1], [-1], [True], [True])
    for a, b in zip(full, (hit[:, :1], frac[:, :1], landmass[:, :1], armed_new[:, :1])):
        np.testing.assert_array_equal(a, b)   # all groups eligible is the same as no mask

    np.testing.assert_array_equal(hit[0], [True, True, True, False, False, True, True, False])
    np.testing.assert_array_equal(landmass[0], [0, 0, 1, -1, -1, 0, 1, -1])
    sighted_at = start + frac[0] * (150.0 - start)
    np.testing.assert_allclose(sighted_at[[0, 1, 2, 6]], [28.0, 28.0, 98.0, 98.0], atol=1.0)
    assert frac[0, 5] == 0.0   # armed: home counts from the start
    # Land that does not count is still land: in sight of B throughout, voyage 7 is not armed.
    np.testing.assert_array_equal(armed_new[0], [True] * 7 + [False])
    with pytest.raises(ValueError, match="eligible"):
        detector.check_segment(lon0, lat0, lon1, lat1, origin, armed, np.ones(8, bool), eligible=eligible[:, :1])


def test_eligible_land_is_hidden_where_ineligible_land_has_the_smaller_margin():
    # The documented limitation: islands 30 km apart with 21 km zones overlapping between 9 and 21 km
    # east of the first. The raster names the nearer one, so the eligible second island comes into
    # sight only past the midpoint, not 9 km east of the first.
    near = (-147.0, -17.0)
    far = offset(*near, 30.0, 0.0)
    lms = [Landmass("Near", "Other", 10.0, [circle_ring(*near, 1.0)]),
           Landmass("Far", "Dest", 10.0, [circle_ring(*far, 1.0)])]
    raster = SightingRaster.build(lms, SightingRule("u20", mode="uniform", radius_km=20.0), BOUNDS,
                                  res_deg=0.02, cap_km=100.0)
    detector = LandfallDetector(lms, [raster])
    lon0, lat0 = offset(*near, -40.0, 0.0)
    lon1, lat1 = offset(*near, 60.0, 0.0)
    hit, frac, landmass, _ = detector.check_segment([lon0], [lat0], [lon1], [lat1], [-1], [True], [True],
                                                    eligible=[[False, True]])
    assert hit[0, 0] and landmass[0, 0] == 1
    assert 13.5 < -40.0 + 100.0 * frac[0, 0] < 17.0


def test_rules_select_and_order_rasters(atoll_rasters):
    detector = LandfallDetector([atoll()], [atoll_rasters[name] for name in ("u20", "u26", "u60")])
    # Straight at the atoll from 100 km off its rim to 10 km off, and the same 150 km north of it.
    north = np.array([0.0, 150.0])
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, np.full(2, ATOLL_R + 100.0), north)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, np.full(2, ATOLL_R + 10.0), north)
    every = detector.check_segment(lon0, lat0, lon1, lat1, [-1, -1], np.zeros(2, bool), np.ones(2, bool))
    # Per-rule armed and active rows follow the order of ``rules``.
    armed = np.array([[False, True], [True, False]])
    active = np.array([[True, True], [True, False]])
    some = detector.check_segment(lon0, lat0, lon1, lat1, [-1, -1], armed, active, rules=[2, 0])
    assert [a.shape for a in some] == [(2, 2)] * 4
    for a, b in zip(every[:3], some[:3]):
        np.testing.assert_array_equal(a[[2, 0]][active], b[active])
    np.testing.assert_array_equal(some[3], (every[3][[2, 0]] & active) | armed)
    assert some[0][:, 0].all() and some[1][0, 0] < some[1][1, 0] and not some[0][1, 1]


def test_arming_is_per_rule(atoll_rasters):
    # Leaving the atoll from 5 to 40 km off its rim: out of sight under 20 km, still in sight under 60 km.
    detector = LandfallDetector([atoll()], [atoll_rasters["u20"], atoll_rasters["u60"]])
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, ATOLL_R + 5.0, 0.0)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, ATOLL_R + 40.0, 0.0)
    hit, _, _, armed = detector.check_segment([lon0], [lat0], [lon1], [lat1], [0], [False], [True])
    assert not hit.any()
    np.testing.assert_array_equal(armed[:, 0], [True, False])
    # An empty batch keeps the (n_rules, n) shape.
    empty = np.zeros(0)
    out = detector.check_segment(empty, empty, empty, empty, empty.astype(int), empty.astype(bool),
                                 empty.astype(bool))
    assert [a.shape for a in out] == [(2, 0)] * 4


def test_larger_radius_sights_earlier(atoll_rasters):
    detector = LandfallDetector([atoll()], [atoll_rasters["u20"], atoll_rasters["u60"]])
    assert detector.rule_names == ["u20", "u60"]
    # Straight at the atoll from 100 km off its rim to 10 km off.
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, ATOLL_R + 100.0, 0.0)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, ATOLL_R + 10.0, 0.0)
    hit, frac, landmass, _ = detector.check_segment([lon0], [lat0], [lon1], [lat1], [-1], [True], [True])
    assert hit.all() and np.all(landmass == 0)
    assert frac[1, 0] < frac[0, 0]
    for r, radius in enumerate((20.0, 60.0)):
        assert rim_km(*along(lon0, lat0, lon1, lat1, frac[r, 0])) == pytest.approx(radius, abs=1.0)


def test_inactive_voyages_are_untouched(atoll_rasters):
    detector = LandfallDetector([atoll()], [atoll_rasters["u20"], atoll_rasters["u60"]])
    # Voyages 0-3 run at the atoll from 100 km to 10 km off its rim; voyage 4 passes 150 km north, out of sight.
    north = np.array([0.0, 0.0, 0.0, 0.0, 150.0])
    lon0, lat0 = offset(ATOLL_LON, ATOLL_LAT, np.array([ATOLL_R + 100.0] * 4 + [-40.0]), north)
    lon1, lat1 = offset(ATOLL_LON, ATOLL_LAT, np.array([ATOLL_R + 10.0] * 4 + [40.0]), north)
    lon1[3] = np.nan   # left the domain
    active = np.array([[True, False, False, True, False],
                       [False, False, False, True, True]])   # rule u60 already satisfied for voyage 0
    armed = np.array([[False, True, False, True, False],
                      [True, True, False, False, False]])
    hit, frac, landmass, armed_new = detector.check_segment(lon0, lat0, lon1, lat1, np.full(5, -1),
                                                            armed, active)
    expected_hit = np.zeros((2, 5), bool)
    expected_hit[0, 0] = True
    np.testing.assert_array_equal(hit, expected_hit)
    assert np.isnan(frac[~expected_hit]).all() and np.all(landmass[~expected_hit] == -1)
    # Only active voyages that were out of sight get armed; a skipped far segment arms only where active.
    expected_armed = armed.copy()
    expected_armed[0, 0] = expected_armed[1, 4] = True
    np.testing.assert_array_equal(armed_new, expected_armed)


def test_detector_rejects_raster_for_other_landmasses(atoll_rasters):
    other = Landmass("Elsewhere", "X", 25.0, [[(-146.0, -16.0)]])
    with pytest.raises(ValueError, match="different landmasses"):
        LandfallDetector([other], [atoll_rasters["u26"]])


# --- catalog and shorelines -----------------------------------------------------

CATALOG = """
landmasses:
  - name: Nuku Hiva
    group: Marquesas
    height_m: 1224
    source: test
    point: [-140.1, -8.87]
    radius_km: 9
    gshhg: {match_point: [-140.1, -8.87]}
  - name: Ecuador
    group: "Coast: Ecuador"
    kind: coast
    height_m: 500
    polyline: [[-80.0, 1.0], [-80.9, -1.0], [-80.2, -3.0]]
    gshhg: {box: [-82, -78, -3.5, 1.5]}
  - name: Rapa Nui
    group: Rapa Nui
    height_m: 507
    polygon: [[-109.45, -27.05], [-109.2, -27.1], [-109.4, -27.2]]
  - name: Henderson
    group: Pitcairn
    kind: atoll
    height_m: 33
    point: [-128.32, -24.37]
"""


def test_load_catalog(tmp_path):
    path = tmp_path / "landmasses.yaml"
    path.write_text(CATALOG)
    nuku, ecuador, rapa, henderson = load_catalog(path)
    assert (nuku.group, nuku.kind, nuku.height_m, nuku.source) == ("Marquesas", "island", 1224.0, "test")
    ring = nuku.rings[0]
    assert np.array_equal(ring[0], ring[-1])
    np.testing.assert_allclose(geo.haversine_km(-140.1, -8.87, ring[:, 0], ring[:, 1]), 9.0, rtol=1e-9)
    assert nuku.gshhg == {"match_point": (-140.1, -8.87)}
    assert ecuador.kind == "coast" and ecuador.rings[0].shape == (3, 2)   # polylines stay open
    assert ecuador.gshhg == {"box": (-82.0, -78.0, -3.5, 1.5)}
    assert rapa.rings[0].shape == (4, 2) and np.array_equal(rapa.rings[0][0], rapa.rings[0][-1])
    assert henderson.rings[0].shape == (1, 2) and henderson.kind == "atoll"


@pytest.mark.parametrize("bad, match", [
    ("polygon: [[0, 0], [1, 0], [0, 1]]\n    point: [0, 0]", "exactly one geometry"),
    ("polyline: [[0, 0], [1, 0]]\n    radius_km: 5", "only to point"),
    ("point: [0, 0]\n    colour: green", "unknown landmass keys"),
    ("point: [0, 0]\n    gshhg: {box: [1, 0, 0, 1]}", "box"),
    ("point: [0, 0]\n    radius_km: -5", "positive radius"),
])
def test_load_catalog_rejects_bad_entries(tmp_path, bad, match):
    path = tmp_path / "landmasses.yaml"
    path.write_text(f"landmasses:\n  - name: X\n    group: G\n    height_m: 1\n    {bad}\n")
    with pytest.raises(ValueError, match=match):
        load_catalog(path)


def test_load_catalog_rejects_duplicate_names(tmp_path):
    path = tmp_path / "landmasses.yaml"
    entry = "  - name: X\n    group: G\n    height_m: 1\n    point: [0, 0]\n"
    path.write_text("landmasses:\n" + entry + entry)
    with pytest.raises(ValueError, match="duplicate"):
        load_catalog(path)


def test_attach_gshhg(tmp_path):
    shapefile = pytest.importorskip("shapefile")
    island = [(-140.2, -8.8), (-140.0, -8.8), (-140.0, -9.0), (-140.2, -9.0), (-140.2, -8.8)]
    # Should a file nest shorelines (an island in a lake), the innermost one containing the point is meant.
    around = [(-141.0, -8.0), (-139.0, -8.0), (-139.0, -10.0), (-141.0, -10.0), (-141.0, -8.0)]
    islet = [(-80.85, -5.45), (-80.8, -5.45), (-80.8, -5.5), (-80.85, -5.5), (-80.85, -5.45)]
    # A continent whose coast leaves the box at a westward peninsula (-81.5, -3).
    continent = [(-80.0, 1.0), (-60.0, 1.0), (-60.0, -7.0), (-79.9, -7.0), (-80.1, -5.0), (-80.4, -4.0),
                 (-81.5, -3.0), (-80.5, -2.0), (-80.2, -1.0), (-80.0, 1.0)]
    # Stored starting inside the box, so the run there wraps past the ring's end.
    stored = continent[7:9] + continent[:7] + continent[7:8]
    with shapefile.Writer(str(tmp_path / "GSHHS_f_L1"), shapeType=shapefile.POLYGON) as writer:
        writer.field("id", "N")
        for k, ring in enumerate((around, island, islet, stored)):
            writer.poly([ring])
            writer.record(k)
    shp = tmp_path / "GSHHS_f_L1.shp"
    box = [-81.0, -79.0, -6.0, 0.0]
    lms = [Landmass("Island", "I", 500.0, [[(-140.1, -8.9)]], gshhg={"match_point": [-140.1, -8.9]}),
           Landmass("Coast", "C", 500.0, [[(-80.0, -3.0)]], kind="coast", gshhg={"box": box}),
           Landmass("Mainland", "C", 500.0, [[(-80.0, -3.0)]], kind="coast",
                    gshhg={"match_point": [-70.0, -3.0], "box": box}),
           Landmass("Untouched", "U", 10.0, [[(-120.0, -10.0)]])]
    island_lm, coast_lm, mainland_lm, untouched_lm = attach_gshhg(lms, shp)

    assert len(island_lm.rings) == 1
    np.testing.assert_array_equal(island_lm.rings[0], island)
    # The coast becomes one open polyline per run of vertices inside the box: no chord across the peninsula.
    runs = [continent[4:6], continent[7:9]]
    assert len(mainland_lm.rings) == 2
    for got, want in zip(mainland_lm.rings, runs):
        np.testing.assert_array_equal(got, want)
    # Without a match point every shoreline reaching into the box counts, offshore islets included.
    assert len(coast_lm.rings) == 3
    np.testing.assert_array_equal(coast_lm.rings[0], islet)
    assert untouched_lm is lms[3]
    assert lms[0].rings[0].shape == (1, 2)   # inputs are not modified

    lost = [Landmass("Lost", "L", 10.0, [[(-130.0, -20.0)]], gshhg={"match_point": [-130.0, -20.0]})]
    with pytest.raises(ValueError, match="no polygon"):
        attach_gshhg(lost, shp)
