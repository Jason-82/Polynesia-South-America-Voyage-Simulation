from pathlib import Path

import numpy as np
import pytest

from voyage_sim import geo
from voyage_sim.vessels import Leeway, Polar, Vessel, load_vessels, parametric_polar

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "vessels.yaml"
KN = geo.KNOT_MS
FACTORS = [(75, 0.7), (90, 0.9), (110, 1.0), (150, 1.0), (180, 0.85)]


def wind_from(from_deg, speed_kn):
    """Wind vector (u, v) m/s blowing from a compass direction."""
    u, v = geo.heading_to_uv(speed_kn * KN, from_deg)
    return -u, -v


@pytest.fixture
def canoe():
    return Vessel(
        name="test_canoe", kind="sailing",
        leeway=Leeway(dw_slope=0.04, dw_offset_ms=0.01, cw_slope=0.015),
        polar=parametric_polar(75.0, 7.0, 0.45, FACTORS),
        min_twa_deg=75.0, min_sail_tws_kn=2.0, heave_to_tws_kn=34.0,
    )


# --- polars -------------------------------------------------------------------

def test_parametric_polar_matches_formula():
    polar = parametric_polar(75.0, 7.0, 0.45, FACTORS)
    rng = np.random.default_rng(0)
    twa = rng.uniform(75.0, 180.0, 5000)
    tws = rng.uniform(0.0, 60.0, 5000)
    expected = np.minimum(7.0, 0.45 * tws) * np.interp(twa, *np.transpose(FACTORS))
    assert polar.speed(twa, tws) == pytest.approx(expected, abs=1e-12)
    # Beam reach (factor 0.9): 0.45 * 10 * 0.9 kn; capped at 7 kn times the factor in a gale.
    assert polar.speed(90.0, 10.0) == pytest.approx(4.05)
    assert polar.speed(130.0, 200.0) == pytest.approx(7.0)
    assert polar.speed(np.full(100, 130.0), np.linspace(0, 200, 100)).max() == pytest.approx(7.0)


def test_parametric_polar_zero_below_min_twa():
    polar = parametric_polar(75.0, 7.0, 0.45, FACTORS)
    below = np.array([0.0, 30.0, 74.9, np.nextafter(75.0, 0.0)])
    assert np.all(polar.speed(below, 15.0) == 0.0)
    # The step is sharp: the full factor applies at min_twa itself.
    assert polar.speed(75.0, 10.0) == pytest.approx(0.45 * 10 * 0.7)
    # Factors flat beyond the listed points; a higher min_twa just moves the step.
    later = parametric_polar(100.0, 7.0, 0.45, FACTORS)
    assert later.speed(99.9, 10.0) == 0.0
    assert later.speed(100.0, 10.0) == pytest.approx(4.5 * 0.95)
    assert parametric_polar(0.0, 7.0, 0.45, [(60, 0.5), (120, 1.0)]).speed(10.0, 10.0) == pytest.approx(2.25)


def test_table_polar_bilinear_exact():
    twa = np.array([30.0, 60.0, 100.0, 180.0])
    tws = np.array([0.0, 6.0, 12.0, 25.0])

    def bilinear(a, w):
        return 0.5 + 0.01 * a + 0.2 * w + 0.003 * a * w

    polar = Polar(twa, tws, bilinear(twa[:, None], tws[None, :]))
    rng = np.random.default_rng(1)
    a = rng.uniform(30.0, 180.0, 2000)
    w = rng.uniform(0.0, 25.0, 2000)
    assert polar.speed(a, w) == pytest.approx(bilinear(a, w), rel=1e-12)
    # Inputs outside the table are clipped to its edges.
    assert polar.speed(10.0, 40.0) == pytest.approx(bilinear(30.0, 25.0))
    # Broadcasting: one TWA against several wind speeds.
    assert polar.speed(60.0, tws).shape == (4,)


def test_polar_validation():
    with pytest.raises(ValueError):
        Polar([0.0, 90.0], [2.0, 10.0], np.ones((2, 2)))      # wind speeds must start at 0
    with pytest.raises(ValueError):
        Polar([90.0, 60.0], [0.0, 10.0], np.ones((2, 2)))     # angles must ascend
    with pytest.raises(ValueError):
        Polar([0.0, 90.0], [0.0, 10.0], np.ones((3, 2)))      # shape mismatch
    with pytest.raises(ValueError):
        parametric_polar(75.0, 7.0, 0.0, FACTORS)
    polar = parametric_polar(75.0, 7.0, 0.45, FACTORS)
    with pytest.raises(ValueError):
        polar.speed_kn[0, 0] = 1.0                           # tables are read-only


# --- leeway -------------------------------------------------------------------

def test_leeway_direction_and_magnitude():
    lw = Leeway(dw_slope=0.04, dw_offset_ms=0.02, cw_slope=0.01, cw_offset_ms=0.005)
    # A 10 m/s easterly pushes the craft west; +1 crosswind is to the right of
    # west (north), -1 to the left (south).
    u, v = lw.velocity(np.array([-10.0, -10.0]), np.array([0.0, 0.0]), np.array([1.0, -1.0]))
    assert u == pytest.approx([-0.42, -0.42])
    assert v == pytest.approx([0.105, -0.105])
    # Any wind direction: project onto the downwind and right-hand axes.
    rng = np.random.default_rng(2)
    wu, wv = rng.normal(0, 8, 500), rng.normal(0, 8, 500)
    sign = np.where(rng.uniform(size=500) < 0.5, -1.0, 1.0)
    u, v = lw.velocity(wu, wv, sign)
    w = np.hypot(wu, wv)
    eu, ev = wu / w, wv / w
    assert u * eu + v * ev == pytest.approx(0.04 * w + 0.02)
    assert u * ev - v * eu == pytest.approx(sign * (0.01 * w + 0.005))


def test_leeway_floor_and_calm():
    lw = Leeway(dw_slope=0.04, dw_offset_ms=-0.1, cw_slope=0.01, cw_offset_ms=-0.1)
    # Negative offsets floor each part at zero rather than reversing it, on either side.
    u, v = lw.velocity(np.ones(2), np.zeros(2), np.array([1.0, -1.0]))
    assert np.all(u == 0.0) and np.all(v == 0.0)
    # Calm air: no drift even with positive offsets.
    u, v = Leeway(0.04, 0.05, 0.01, 0.05).velocity(np.zeros(3), np.zeros(3), np.ones(3))
    assert np.all(u == 0.0) and np.all(v == 0.0)


# --- vessels ------------------------------------------------------------------

def test_sailable_thresholds(canoe):
    twa = np.array([75.0, np.nextafter(75.0, 0.0), 120.0, 120.0, 120.0, 120.0, np.nan, 120.0])
    tws = np.array([10.0, 10.0, 2.0, np.nextafter(2.0, 0.0), 34.0, np.nextafter(34.0, 0.0), 10.0, np.nan])
    assert canoe.sailable(twa, tws).tolist() == [True, False, True, False, False, True, False, False]
    drifter = Vessel("drifter", "drift", Leeway(0.03))
    assert not drifter.can_sail and not drifter.sailable(twa, tws).any()


def test_water_velocity_sails_on_polar(canoe):
    # North wind of 10 kn, heading east: beam reach at 4.05 kn.
    wu, wv = wind_from(0.0, 10.0)
    u, v, sailing = canoe.water_velocity(np.array([90.0]), np.array([wu]), np.array([wv]), np.array([1.0]))
    assert sailing.tolist() == [True]
    assert u[0] == pytest.approx(4.05 * KN) and v[0] == pytest.approx(0.0, abs=1e-12)
    # Close-hauled at exactly 75 deg off a north wind, either side of it.
    u, v, sailing = canoe.water_velocity(np.array([75.0, 285.0]), wu, wv, 1.0)
    assert sailing.all()
    assert np.hypot(u, v) == pytest.approx(0.45 * 10 * 0.7 * KN)
    assert geo.uv_to_speed_heading(u, v)[1] == pytest.approx([75.0, 285.0])
    # Oblique wind from 200 deg: steering 320 is 120 off it (factor 1.0); steering
    # 200 is head to wind, so the craft drifts downwind toward 20 deg.
    wu, wv = wind_from(200.0, 10.0)
    u, v, sailing = canoe.water_velocity(np.array([320.0, 200.0]), wu, wv, 0.0)
    assert sailing.tolist() == [True, False]
    speed, heading = geo.uv_to_speed_heading(u, v)
    assert speed[0] == pytest.approx(4.5 * KN) and heading[0] == pytest.approx(320.0)
    assert heading[1] == pytest.approx(20.0)


def test_water_velocity_shapes(canoe):
    wu, wv = wind_from(0.0, 10.0)
    for n in (0, 1, 4):
        u, v, sailing = canoe.water_velocity(np.full(n, 90.0), np.full(n, wu), np.full(n, wv), np.ones(n))
        assert u.shape == v.shape == sailing.shape == (n,) and sailing.dtype == bool
    u, v, sailing = canoe.water_velocity(90.0, wu, wv, 1.0)
    assert np.shape(u) == () and bool(sailing)
    # Wind outside the field (NaN) gives NaN velocity, never a sailing voyage.
    u, v, sailing = canoe.water_velocity(np.array([90.0]), np.array([np.nan]), np.array([np.nan]), 1.0)
    assert np.isnan(u[0]) and np.isnan(v[0]) and not sailing[0]


def test_water_velocity_falls_back_to_leeway(canoe):
    n = 5
    heading = np.array([np.nan, 0.0, 90.0, 90.0, 90.0])   # no steering, head to wind, ok, ok, ok
    tws_kn = np.array([10.0, 10.0, 1.5, 34.0, 10.0])      # ..., becalmed, gale, fine
    wu, wv = wind_from(np.zeros(n), tws_kn)
    sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    u, v, sailing = canoe.water_velocity(heading, wu, wv, sign)
    assert sailing.tolist() == [False, False, False, False, True]
    lu, lv = canoe.leeway.velocity(wu, wv, sign)
    assert u[:4] == pytest.approx(lu[:4]) and v[:4] == pytest.approx(lv[:4])
    # Leeway from a north wind runs south.
    assert np.all(lv < 0)
    # A drift vessel never sails, whatever it is told to steer.
    drifter = Vessel("drifter", "drift", canoe.leeway)
    u, v, sailing = drifter.water_velocity(heading, wu, wv, sign)
    assert not sailing.any()
    assert u == pytest.approx(lu) and v == pytest.approx(lv)


def test_speed_scale_and_speed_factor(canoe):
    wu, wv = wind_from(0.0, 10.0)
    factor = np.array([0.5, 1.0, 2.0])
    fast = canoe.with_overrides(speed_scale=1.2)
    u, v, sailing = fast.water_velocity(np.full(3, 90.0), wu, wv, 1.0, speed_factor=factor)
    assert sailing.all()
    assert u == pytest.approx(4.05 * KN * 1.2 * factor)
    # Speed scaling never touches leeway.
    calm_u, _, sailing = fast.water_velocity(np.full(3, np.nan), wu, wv, 1.0, speed_factor=factor)
    assert not sailing.any()
    assert calm_u == pytest.approx(canoe.leeway.velocity(wu, wv, 1.0)[0])


def test_with_overrides_returns_modified_copy(canoe):
    slow = canoe.with_overrides(speed_scale=0.8, min_twa_deg=90.0)
    assert (slow.speed_scale, slow.min_twa_deg) == (0.8, 90.0)
    assert (canoe.speed_scale, canoe.min_twa_deg) == (1.0, 75.0)
    assert slow.polar == canoe.polar
    # A leeway mapping changes only the named coefficients.
    windier = canoe.with_overrides(leeway={"dw_slope": 0.06})
    assert windier.leeway == Leeway(dw_slope=0.06, dw_offset_ms=0.01, cw_slope=0.015)
    assert canoe.leeway.dw_slope == 0.04
    with pytest.raises(ValueError):
        canoe.with_overrides(min_sail_tws_kn=40.0)          # above heave-to
    with pytest.raises(ValueError):
        canoe.with_overrides(kind="drift")                  # drift craft carry no polar
    with pytest.raises(TypeError):
        canoe.with_overrides(top_speed=9.0)
    # The polar is zero below 75 deg: allowing sailing at 60 would stall the craft
    # (sailing at 0 kn, no leeway), so it is refused, as is a zero speed scale.
    with pytest.raises(ValueError, match="positive speed"):
        canoe.with_overrides(min_twa_deg=60.0)
    with pytest.raises(ValueError):
        canoe.with_overrides(speed_scale=0.0)


# --- configuration --------------------------------------------------------------

def test_load_vessels_config():
    vessels = load_vessels(CONFIG)
    assert {"canoe_75", "canoe_90", "raft_sailing", "drift_canoe", "drift_raft"} <= set(vessels)
    for name, vessel in vessels.items():
        assert vessel.name == name
        assert vessel.can_sail == (vessel.polar is not None)
        assert all(isinstance(s, str) and s for s in vessel.sources.values())

    c75, c90 = vessels["canoe_75"], vessels["canoe_90"]
    assert 3.5 <= c75.polar.speed(90.0, 10.0) <= 4.5
    assert 5.5 <= c75.polar.speed(90.0, 15.0) <= 6.5
    assert c75.polar.speed_kn.max() <= 7.5
    # canoe_90 is the same hull with no windward sailing.
    assert c75.sailable(80.0, 10.0) and not c90.sailable(80.0, 10.0)
    assert c90.polar.speed(80.0, 10.0) == 0.0
    a = np.linspace(90.0, 180.0, 50)
    assert c90.polar.speed(a, 12.0) == pytest.approx(c75.polar.speed(a, 12.0))
    assert c90.sources["min_twa_deg"] != c75.sources["min_twa_deg"]
    # The raft's sailing settings stay flagged until the Dewan & Hosler numbers are adopted.
    assert all("PROVISIONAL" in s for k, s in vessels["raft_sailing"].sources.items()
               if not k.startswith("leeway"))
    assert not vessels["drift_canoe"].can_sail and not vessels["drift_raft"].can_sail
    # Montenegro et al. (2008) seed capsules ride the current with no wind push.
    assert vessels["seed_capsule"].leeway.velocity(np.array([-8.0]), np.array([0.0]), np.array([1]))[0][0] == 0.0


def _write(tmp_path, text):
    path = tmp_path / "vessels.yaml"
    path.write_text(text)
    return path


def test_load_vessels_table_polar_and_parent_sources(tmp_path):
    path = _write(tmp_path, """
vessels:
  tabled:
    kind: sailing
    min_twa_deg: 60
    polar: {type: table, twa_deg: [60, 180], tws_kn: [0, 20], speed_kn: [[0, 4], [0, 8]]}
    leeway: {dw_slope: 0.03}
    sources: {min_twa_deg: a, polar: b, leeway: c}
""")
    vessel = load_vessels(path)["tabled"]
    assert vessel.polar.speed(120.0, 10.0) == pytest.approx(3.0)
    assert vessel.heave_to_tws_kn == np.inf and vessel.leeway.cw_slope == 0.0


@pytest.mark.parametrize("change, message", [
    ("    sources: {leeway.dw_slope: a}", "no sources entry"),
    ("    sources: {min_twa_deg: a, polar: b, leeway: c, speed_scale: d}", "does not have"),
    ("    sources: {min_twa_deg: a, polar: b, leeway: c}\n    hull: twin", "unknown vessel keys"),
    ("    sources: {min_twa_deg: , polar: b, leeway: c}", "non-empty citation"),
    ("    sources: [a, b, c]", "non-empty citation"),
])
def test_load_vessels_rejects_bad_config(tmp_path, change, message):
    path = _write(tmp_path, f"""
vessels:
  bad:
    kind: sailing
    min_twa_deg: 75
    polar: {{type: parametric, max_speed_kn: 7, wind_ratio: 0.45, angle_factors: [[75, 1.0]]}}
    leeway: {{dw_slope: 0.03}}
{change}
""")
    with pytest.raises(ValueError, match=message):
        load_vessels(path)


def test_load_vessels_rejects_polar_zero_past_min_twa(tmp_path):
    # The polar's own zero region (below 85) reaches past the vessel's closest angle (75).
    path = _write(tmp_path, """
vessels:
  stalls:
    kind: sailing
    min_twa_deg: 75
    polar: {type: parametric, min_twa_deg: 85, max_speed_kn: 7, wind_ratio: 0.45, angle_factors: [[75, 1.0]]}
    leeway: {dw_slope: 0.03}
    sources: {min_twa_deg: a, polar: b, leeway: c}
""")
    with pytest.raises(ValueError, match="positive speed"):
        load_vessels(path)
