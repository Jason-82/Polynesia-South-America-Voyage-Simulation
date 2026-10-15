import numpy as np
import pytest

from voyage_sim import geo


def test_wrap_and_angle_diff():
    assert geo.wrap_lon(190.0) == pytest.approx(-170.0)
    assert geo.wrap_lon(-180.0) == pytest.approx(-180.0)
    assert geo.angle_diff(10.0, 350.0) == pytest.approx(20.0)
    assert geo.angle_diff(350.0, 10.0) == pytest.approx(-20.0)


def test_wind_conventions():
    # Air moving west (u < 0) is an easterly: it blows from 90 degrees.
    assert geo.wind_from_deg(-5.0, 0.0) == pytest.approx(90.0)
    # Air moving north is a southerly.
    assert geo.wind_from_deg(0.0, 5.0) == pytest.approx(180.0)
    # Heading north into a northerly is head to wind; heading south is dead downwind.
    assert geo.true_wind_angle(0.0, 0.0) == pytest.approx(0.0)
    assert geo.true_wind_angle(180.0, 0.0) == pytest.approx(180.0)
    assert geo.true_wind_angle(285.0, 0.0) == pytest.approx(75.0)


def test_heading_uv_roundtrip():
    u, v = geo.heading_to_uv(2.0, 90.0)
    assert u == pytest.approx(2.0) and v == pytest.approx(0.0, abs=1e-12)
    s, h = geo.uv_to_speed_heading(-1.0, -1.0)
    assert s == pytest.approx(np.sqrt(2)) and h == pytest.approx(225.0)


def test_advance_matches_haversine():
    # 1 m/s east for a day at the equator: 86.4 km.
    lon, lat = geo.advance(-140.0, 0.0, 1.0, 0.0, 86_400.0)
    assert geo.haversine_km(-140.0, 0.0, lon, lat) == pytest.approx(86.4, rel=1e-3)
    # The same eastward speed covers twice the longitude at 60 degrees south.
    lon60, _ = geo.advance(-140.0, -60.0, 1.0, 0.0, 86_400.0)
    assert (lon60 + 140.0) == pytest.approx(2 * (lon + 140.0), rel=1e-3)
    # North at 1 m/s for 6 h, any latitude: 21.6 km.
    lon2, lat2 = geo.advance(-100.0, -30.0, 0.0, 1.0, 21_600.0)
    assert geo.haversine_km(-100.0, -30.0, lon2, lat2) == pytest.approx(21.6, rel=1e-3)


def test_bearing():
    assert geo.initial_bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(90.0)
    assert geo.initial_bearing_deg(0.0, 0.0, 0.0, -1.0) == pytest.approx(180.0)


def test_chord_roundtrip():
    km = np.array([0.0, 1.0, 40.0, 500.0])
    assert geo.chord_to_km(geo.km_to_chord(km)) == pytest.approx(km)
    a = geo.lonlat_to_xyz(-140.0, -10.0)
    b = geo.lonlat_to_xyz(-139.0, -10.5)
    assert geo.chord_to_km(np.linalg.norm(a - b)) == pytest.approx(
        geo.haversine_km(-140.0, -10.0, -139.0, -10.5), rel=1e-9)


def test_time_helpers():
    h = geo.to_epoch_hours("1976-05-01T00:00")
    assert geo.year_of(h) == 1976 and geo.month_of(h) == 5
    assert geo.from_epoch_hours(h) == np.datetime64("1976-05-01T00:00:00")
    assert geo.to_epoch_hours("1970-01-02") == pytest.approx(24.0)
