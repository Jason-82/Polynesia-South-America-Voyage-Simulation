import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

from voyage_sim import geo
from voyage_sim.fields import (
    Environment,
    FunctionField,
    GriddedField,
    MonthlyClimatologyField,
    SumField,
    UniformField,
    monthly_climatology,
)

NAN = np.nan


def multilinear(lon, lat, t):
    """Spanned by 1, lon, lat, t and their products: trilinear interpolation is exact for it."""
    u = 0.5 + 0.02 * lon - 0.03 * lat + 0.01 * t + 1e-3 * lon * lat - 2e-4 * lat * t + 1e-5 * lon * lat * t
    v = -1.0 + 0.01 * lon + 0.05 * lat - 0.02 * t + 3e-4 * lon * t
    return u, v


def grid_values(fn, lon, lat, time):
    return fn(lon[None, None, :], lat[None, :, None], time[:, None, None])


# A deliberately non-uniform grid.
LON = np.array([-160.0, -151.5, -140.0, -122.0, -100.25, -90.0])
LAT = np.array([-45.0, -31.0, -20.5, -3.0, 0.0, 12.0])
TIME = np.array([0.0, 6.0, 18.0, 24.0, 96.0])


@pytest.fixture
def field():
    return GriddedField(LON, LAT, TIME, *grid_values(multilinear, LON, LAT, TIME))


def test_multilinear_field_is_reproduced_exactly(field):
    rng = np.random.default_rng(1)
    n = 5000
    lon = rng.uniform(LON[0], LON[-1], n)
    lat = rng.uniform(LAT[0], LAT[-1], n)
    t = rng.uniform(TIME[0], TIME[-1], n)
    u, v = field.sample(lon, lat, t)
    eu, ev = multilinear(lon, lat, t)
    assert u.dtype == np.float64 and u.shape == (n,)
    assert_allclose(u, eu, rtol=1e-12, atol=1e-12)
    assert_allclose(v, ev, rtol=1e-12, atol=1e-12)


def test_each_cell_is_interpolated_from_its_own_corners():
    # Any cell extrapolates a multilinear field exactly, so only arbitrary node
    # values show that the lookup picks the right cell and the right time slices.
    rng = np.random.default_rng(4)
    u, v = rng.normal(size=(2, TIME.size, LAT.size, LON.size))
    field = GriddedField(LON, LAT, TIME, u, v)
    nodes = np.meshgrid(TIME, LAT, LON, indexing="ij")
    fu, fv = field.sample(nodes[2].ravel(), nodes[1].ravel(), nodes[0].ravel())
    assert_allclose(fu, u.ravel(), rtol=1e-12, atol=1e-12)
    assert_allclose(fv, v.ravel(), rtol=1e-12, atol=1e-12)

    def mid(a):
        return 0.5 * (a[:-1] + a[1:])

    def corner_mean(a):
        nt, ny, nx = a.shape
        return sum(a[k:nt - 1 + k, j:ny - 1 + j, i:nx - 1 + i] for k in (0, 1) for j in (0, 1) for i in (0, 1)) / 8

    centres = np.meshgrid(mid(TIME), mid(LAT), mid(LON), indexing="ij")
    fu, fv = field.sample(centres[2].ravel(), centres[1].ravel(), centres[0].ravel())
    assert_allclose(fu, corner_mean(u).ravel(), rtol=1e-12, atol=1e-12)
    assert_allclose(fv, corner_mean(v).ravel(), rtol=1e-12, atol=1e-12)


def test_datetime64_times_are_refused_not_read_as_hours():
    days = np.arange(np.datetime64("2000-01-01"), np.datetime64("2000-01-03"))
    data = np.zeros((2, LAT.size, LON.size))
    with pytest.raises(TypeError, match="epoch hours"):
        GriddedField(LON, LAT, days, data, data)
    with pytest.raises(TypeError, match="epoch hours"):
        monthly_climatology(days, data, data)


def test_single_time_slice_is_time_invariant():
    u, v = grid_values(multilinear, LON, LAT, np.array([0.0]))
    lon = np.array([-155.0, -130.0, -95.0])
    lat = np.array([-40.0, 5.0, -10.0])
    one = GriddedField(LON, LAT, [1234.0], u, v)
    static = GriddedField(LON, LAT, None, u[0], v[0])
    expected = multilinear(lon, lat, 0.0)
    for f in (one, static):
        assert f.bounds[4:] == (-np.inf, np.inf)
        for t in (-1e7, 0.0, 1e7):
            assert_allclose(f.sample(lon, lat, np.full(3, t)), expected, rtol=1e-12)


def test_outside_the_grid_or_time_range_is_nan(field):
    eps = 1e-9
    lon = np.array([LON[0] - eps, LON[-1] + eps, -130.0, -130.0, -130.0, -130.0, NAN, -130.0])
    lat = np.array([-10.0, -10.0, LAT[0] - eps, LAT[-1] + eps, -10.0, -10.0, -10.0, -10.0])
    t = np.array([10.0, 10.0, 10.0, 10.0, TIME[0] - eps, TIME[-1] + eps, 10.0, NAN])
    u, v = field.sample(lon, lat, t)
    assert np.all(np.isnan(u)) and np.all(np.isnan(v))


def test_grid_edges_and_last_time_are_inside(field):
    lon = np.array([LON[0], LON[-1], LON[-1], LON[0], LON[-1]])
    lat = np.array([LAT[0], LAT[-1], LAT[0], LAT[-1], LAT[-1]])
    t = np.array([TIME[0], TIME[-1], TIME[-1], TIME[0], TIME[0]])
    u, v = field.sample(lon, lat, t)
    eu, ev = multilinear(lon, lat, t)
    assert_allclose(u, eu, rtol=1e-12)
    assert_allclose(v, ev, rtol=1e-12)
    assert field.bounds == (LON[0], LON[-1], LAT[0], LAT[-1], TIME[0], TIME[-1])


def era5_like():
    """ERA5 as the CDS delivers it: valid_time, descending latitude, 0..360 longitude, float32."""
    lon = np.arange(195.0, 295.0 + 0.1, 0.5)
    lat = np.arange(30.0, -50.0 - 0.1, -0.5)
    times = np.arange(np.datetime64("1976-05-01T00"), np.datetime64("1976-05-03T00"), np.timedelta64(6, "h"))
    u, v = grid_values(multilinear, lon - 360.0, lat, geo.to_epoch_hours(times) - geo.to_epoch_hours("1976-05-01"))
    dims = ("valid_time", "latitude", "longitude")
    ds = xr.Dataset(
        {"u10": (dims, u.astype(np.float32)), "v10": (dims, v.astype(np.float32))},
        coords={"valid_time": times.astype("datetime64[ns]"), "latitude": lat, "longitude": lon,
                "number": 0, "expver": ("valid_time", ["0001"] * times.size)},
    )
    return ds


def test_era5_like_dataset_matches_normalized_arrays():
    ds = era5_like()
    field = GriddedField.from_dataset(ds, "u10", "v10")
    hours = geo.to_epoch_hours(ds.valid_time.values)
    lon, lat = ds.longitude.values - 360.0, ds.latitude.values[::-1]
    assert_array_equal(field.lon, lon)
    assert_array_equal(field.lat, lat)
    assert_array_equal(field.time, hours)
    assert field.u.dtype == np.float32
    assert field.bounds == (-165.0, -65.0, -50.0, 30.0, hours[0], hours[-1])

    reference = GriddedField(lon, lat, hours, ds.u10.values[:, ::-1, :], ds.v10.values[:, ::-1, :])
    rng = np.random.default_rng(2)
    n = 2000
    q = (rng.uniform(-165, -65, n), rng.uniform(-50, 30, n), rng.uniform(hours[0], hours[-1], n))
    u, v = field.sample(*q)
    assert u.dtype == v.dtype == np.float64  # float32 storage, float64 samples
    ru, rv = reference.sample(*q)
    assert_array_equal(u, ru)
    assert_array_equal(v, rv)
    # And both are right: only float32 storage rounding separates them from the analytic field.
    eu, ev = multilinear(q[0], q[1], q[2] - hours[0])
    assert_allclose(u, eu, atol=1e-5)
    assert_allclose(v, ev, atol=1e-5)


def test_global_0_360_grid_is_resorted_across_the_seam():
    lon = np.arange(0.0, 360.0, 1.0)
    lat = np.array([-10.0, 0.0, 10.0])
    u = np.broadcast_to(geo.wrap_lon(lon), (3, 360))  # linear in wrapped longitude
    v = np.broadcast_to(lat[:, None], (3, 360))
    field = GriddedField(lon, lat, None, u, v)
    assert field.lon[0] == -180.0 and field.lon[-1] == 179.0
    q = np.array([-179.5, -0.25, 0.75, 120.3, 179.0])
    fu, fv = field.sample(q, np.full(5, 5.0), np.zeros(5))
    assert_allclose(fu, q, rtol=1e-12)
    assert_allclose(fv, 5.0)


def test_grid_straddling_the_antimeridian_is_rejected():
    lon = np.arange(170.0, 200.0, 1.0)
    lat = np.array([-10.0, 0.0])
    with pytest.raises(ValueError, match="antimeridian"):
        GriddedField(lon, lat, None, np.zeros((2, 30)), np.zeros((2, 30)))


def glorys_like(depths=(0.494,)):
    """GLORYS surface currents: (time, depth, latitude, longitude), float32, NaN on land."""
    lon = np.array([-80.0, -79.0, -78.0])
    lat = np.array([-10.0, -9.0, -8.0])
    uo0 = np.array([[1.0, 2.0, NAN],
                    [3.0, NAN, NAN],
                    [NAN, NAN, NAN]])
    uo = np.stack([uo0, 2 * uo0])[:, None].repeat(len(depths), axis=1)
    dims = ("time", "depth", "latitude", "longitude")
    return xr.Dataset(
        {"uo": (dims, uo.astype(np.float32)), "vo": (dims, -uo.astype(np.float32))},
        coords={"time": np.array(["1995-01-01", "1995-01-02"], dtype="datetime64[ns]"),
                "depth": np.array(depths, dtype=np.float32), "latitude": lat, "longitude": lon},
    )


def test_glorys_like_land_corners_are_dropped_and_weights_renormalized():
    ds = glorys_like()
    field = GriddedField.from_dataset(ds, "uo", "vo")
    assert field.u.shape == (2, 3, 3) and field.u.dtype == np.float32
    t0, t1 = field.time
    tm = 0.5 * (t0 + t1)
    lon = np.array([-79.75, -80.0, -78.5, -79.0])
    lat = np.array([-9.5, -10.0, -8.5, -9.0])
    u, v = field.sample(lon, lat, np.array([tm, t0, tm, tm]))
    # Point 0: corners 1, 2, 3 with weights 0.375, 0.125, 0.375 (the NaN corner's 0.125
    # is dropped) -> 1.75 / 0.875 = 2 at the first time, 4 at the second, 3 halfway.
    # Point 1 sits on a sea node. Point 2 is in an all-land cell, point 3 on a land node.
    assert_allclose(u, [3.0, 1.0, 0.0, 0.0], rtol=1e-12)
    assert_allclose(v, [-3.0, -1.0, 0.0, 0.0], rtol=1e-12)

    land = GriddedField.from_dataset(ds, "uo", "vo", land_value=-7.0)
    u, v = land.sample(lon, lat, np.array([tm, t0, tm, tm]))
    assert_allclose(u, [3.0, 1.0, -7.0, -7.0], rtol=1e-12)
    assert_allclose(v, [-3.0, -1.0, -7.0, -7.0], rtol=1e-12)


def test_dataset_with_several_depth_levels_is_rejected():
    with pytest.raises(ValueError, match="depth"):
        GriddedField.from_dataset(glorys_like(depths=(0.494, 1.54)), "uo", "vo")


def test_dataset_without_coordinate_values_is_rejected():
    # Index dimensions (curvilinear or projected grids) must not be read as 0, 1, 2 ... degrees.
    ds = xr.Dataset({"u": (("y", "x"), np.ones((3, 4))), "v": (("y", "x"), np.ones((3, 4)))})
    with pytest.raises(ValueError, match="coordinate values"):
        GriddedField.from_dataset(ds, "u", "v")


MONTHLY = 10.0 * np.arange(1, 13)  # January 10 ... December 120


@pytest.fixture
def climatology():
    lon = np.array([-150.0, -100.0])
    lat = np.array([-30.0, 0.0])
    u12 = np.broadcast_to(MONTHLY[:, None, None], (12, 2, 2))
    return MonthlyClimatologyField(lon, lat, u12, -u12)


def sample_at(field, iso):
    t = geo.to_epoch_hours(np.array(iso, dtype="datetime64[s]"))
    return field.sample(np.full(t.shape, -120.0), np.full(t.shape, -10.0), t)


@pytest.mark.parametrize("year", [1200, 1955, 2000, 2023])
def test_climatology_at_month_centres_equals_that_month(climatology, year):
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)  # 1200 and 2000 are leap years
    centres = [f"{year}-01-16T12", f"{year}-02-15T{'12' if leap else '00'}", f"{year}-04-16T00",
               f"{year}-07-16T12", f"{year}-12-16T12"]
    u, v = sample_at(climatology, centres)
    assert_allclose(u, MONTHLY[[0, 1, 3, 6, 11]], rtol=1e-12)
    assert_allclose(v, -u)


@pytest.mark.parametrize("year", [1200, 1970, 2000, 2023])
def test_climatology_is_linear_between_centres_and_cyclic(climatology, year):
    # New Year is midway between the December and January centres.
    # 23 June 15:00 is a quarter of the 30.5 days from the June centre (16th 00:00) to July's (16th 12:00).
    u, _ = sample_at(climatology, [f"{year}-01-01T00", f"{year}-06-23T15"])
    assert_allclose(u, [(MONTHLY[11] + MONTHLY[0]) / 2, MONTHLY[5] + 0.25 * (MONTHLY[6] - MONTHLY[5])],
                    rtol=1e-12)


def test_climatology_outside_space_or_nan_time_is_nan(climatology):
    u, v = climatology.sample(np.array([-151.0, -120.0]), np.array([-10.0, -10.0]), np.array([0.0, NAN]))
    assert np.all(np.isnan(u)) and np.all(np.isnan(v))


def test_sum_field_adds_the_two_and_is_nan_where_either_is(field, climatology):
    # Winds plus a monthly anomaly on a smaller grid. The third point is outside the anomaly's grid,
    # the fourth after the winds end.
    total = SumField(field, climatology)
    lon = np.array([-120.0, -101.0, -155.0, -120.0])
    lat = np.array([-10.0, -29.0, -10.0, -10.0])
    t = np.array([20.0, 90.0, 20.0, 200.0])
    (au, av), (bu, bv) = field.sample(lon, lat, t), climatology.sample(lon, lat, t)
    assert np.isfinite(au[2]) and np.isfinite(bu[3])
    u, v = total.sample(lon, lat, t)
    assert u.dtype == v.dtype == np.float64
    assert_allclose(u[:2], au[:2] + bu[:2], rtol=1e-12)
    assert_allclose(v[:2], av[:2] + bv[:2], rtol=1e-12)
    assert np.all(np.isnan(u[2:])) and np.all(np.isnan(v[2:]))


def test_environment_wind_cap():
    calm = UniformField(0.0, 0.0)
    assert Environment(calm, calm).wind_cap_ms == 50.0
    assert Environment(calm, calm, wind_cap_ms=None).wind_cap_ms is None
    for bad in (0.0, -5.0, NAN):
        with pytest.raises(ValueError, match="wind_cap_ms"):
            Environment(calm, calm, wind_cap_ms=bad)


def test_monthly_climatology_recovers_monthly_means_with_nans():
    days = np.arange(np.datetime64("2003-01-01"), np.datetime64("2006-01-01"))  # 2004 is a leap year
    t = geo.to_epoch_hours(days)
    month = geo.month_of(t) - 1
    year = geo.year_of(t)
    u = np.empty((t.size, 2, 2))
    u[:, 0, 0] = MONTHLY[month] + (year - 2004)  # years cancel: 2003 and 2005 have equal months
    u[:, 0, 1] = MONTHLY[month]
    u[np.random.default_rng(3).random(t.size) < 0.3, 0, 1] = NAN  # gaps don't move a constant mean
    u[:, 1, 0] = np.where(month == 2, NAN, MONTHLY[month])  # no valid March sample
    u[:, 1, 1] = NAN  # land
    u12, v12 = monthly_climatology(t, u, 2 * u)
    assert u12.shape == (12, 2, 2)
    assert_allclose(u12[:, 0, 0], MONTHLY, rtol=1e-12)
    assert_allclose(u12[:, 0, 1], MONTHLY, rtol=1e-12)
    assert np.isnan(u12[2, 1, 0]) and np.all(np.delete(u12[:, 1, 0], 2) == np.delete(MONTHLY, 2))
    assert np.all(np.isnan(u12[:, 1, 1]))
    assert_allclose(v12, 2 * u12)


def test_uniform_and_function_fields_give_writable_float64_arrays():
    lon = np.array([-150.0, -120.0, -90.0])
    lat = np.array([-20.0, -10.0, 0.0])
    t = np.zeros(3)
    u, v = UniformField(1.5, -2).sample(lon, lat, t)
    assert_array_equal(u, [1.5] * 3)
    assert_array_equal(v, [-2.0] * 3)
    assert u.dtype == v.dtype == np.float64
    u[0] = v[0] = 0.0  # callers may update samples in place
    u, v = FunctionField(lambda lon, lat, t: (0.01 * lon, 3)).sample(lon, lat, t)
    assert_allclose(u, 0.01 * lon)
    assert_array_equal(v, [3.0] * 3)
    assert u.dtype == v.dtype == np.float64
    u[0] = v[0] = 0.0
