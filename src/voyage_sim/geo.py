"""Geodesy, units, angles and time: the conventions every module shares.

- Longitude is degrees east, wrapped to [-180, 180); latitude is degrees north.
- Bearings and headings are compass degrees clockwise from true north, [0, 360).
- Velocities are (u, v) in m/s, u eastward and v northward.
- A wind vector points where the air is going; ``wind_from_deg`` gives where it
  comes from, which is what sailing angles are measured against.
- Times are float64 hours since 1970-01-01T00:00Z ("epoch hours").
- Distances used for sighting land are km; motion is integrated in metres.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean radius
EARTH_RADIUS_KM = EARTH_RADIUS_M / 1000.0
KNOT_MS = 1852.0 / 3600.0  # one knot in m/s
SECONDS_PER_HOUR = 3600.0
HOURS_PER_DAY = 24.0
EPOCH = np.datetime64("1970-01-01T00:00:00", "s")


# --- angles -----------------------------------------------------------------

def wrap_lon(lon):
    """Wrap longitudes to [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


def wrap360(deg):
    """Wrap angles to [0, 360)."""
    return np.asarray(deg, dtype=float) % 360.0


def angle_diff(a, b):
    """Signed smallest difference ``a - b`` in degrees, in [-180, 180)."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 180.0) % 360.0 - 180.0


def heading_to_uv(speed, heading_deg):
    """Velocity components for a speed along a compass heading."""
    h = np.radians(heading_deg)
    return speed * np.sin(h), speed * np.cos(h)


def uv_to_speed_heading(u, v):
    """Speed and compass heading of a velocity (heading is 0 for a zero vector)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.hypot(u, v), wrap360(np.degrees(np.arctan2(u, v)))


def wind_from_deg(u, v):
    """Compass direction the wind blows from, given the air's velocity."""
    return wrap360(np.degrees(np.arctan2(-np.asarray(u, dtype=float), -np.asarray(v, dtype=float))))


def true_wind_angle(heading_deg, wind_from):
    """Angle between heading and wind direction: 0 = head to wind, 180 = dead downwind."""
    return np.abs(angle_diff(heading_deg, wind_from))


# --- distances and motion -----------------------------------------------------

def haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(x, dtype=float)) for x in (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lon1, lat1, lon2, lat2):
    """Initial great-circle bearing from point 1 to point 2."""
    lon1, lat1, lon2, lat2 = (np.radians(np.asarray(x, dtype=float)) for x in (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return wrap360(np.degrees(np.arctan2(y, x)))


def advance(lon, lat, u, v, dt_s):
    """Move positions by velocity (u, v) m/s for ``dt_s`` seconds.

    A tangent-plane step with the east-west scale taken at the step's
    mid-latitude: errors are well under 1% for steps of ~100 km outside polar
    latitudes, far below the uncertainty in the forcing.
    """
    lat = np.asarray(lat, dtype=float)
    dlat = np.degrees(np.asarray(v, dtype=float) * dt_s / EARTH_RADIUS_M)
    lat_mid = lat + 0.5 * dlat
    dlon = np.degrees(np.asarray(u, dtype=float) * dt_s / (EARTH_RADIUS_M * np.cos(np.radians(lat_mid))))
    return wrap_lon(np.asarray(lon, dtype=float) + dlon), np.clip(lat + dlat, -89.9, 89.9)


def lonlat_to_xyz(lon, lat):
    """Unit-sphere Cartesian coordinates, shape (..., 3), for KD-tree queries."""
    lon = np.radians(np.asarray(lon, dtype=float))
    lat = np.radians(np.asarray(lat, dtype=float))
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def chord_to_km(chord):
    """Great-circle distance for a unit-sphere chord length."""
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord, dtype=float) / 2, 0.0, 1.0))


def km_to_chord(km):
    """Unit-sphere chord length for a great-circle distance."""
    return 2 * np.sin(np.asarray(km, dtype=float) / (2 * EARTH_RADIUS_KM))


# --- time ---------------------------------------------------------------------

def to_epoch_hours(t):
    """datetime64, ISO string, or arrays of them -> float hours since 1970-01-01T00Z."""
    t = np.asarray(t, dtype="datetime64[s]")
    return (t - EPOCH).astype("timedelta64[s]").astype(np.float64) / SECONDS_PER_HOUR


def from_epoch_hours(hours):
    """Float epoch hours -> datetime64[s] (rounded to the nearest second)."""
    secs = np.round(np.asarray(hours, dtype=float) * SECONDS_PER_HOUR).astype(np.int64)
    return EPOCH + secs.astype("timedelta64[s]")


def year_of(hours):
    """Calendar year (UTC) of epoch hours."""
    return from_epoch_hours(hours).astype("datetime64[Y]").astype(np.int64) + 1970


def month_of(hours):
    """Calendar month 1-12 (UTC) of epoch hours."""
    return from_epoch_hours(hours).astype("datetime64[M]").astype(np.int64) % 12 + 1
