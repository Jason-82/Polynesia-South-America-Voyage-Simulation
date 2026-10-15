"""Wind and current fields: (u, v) in m/s at any position and time.

Every field answers ``sample(lon, lat, t) -> (u, v)`` for 1-D arrays of
positions and epoch-hour times, one entry per voyage, and returns NaN where a
point is outside the data; the simulator treats that as leaving the domain.

Gridded data (ERA5 winds, GLORYS currents) are interpolated bilinearly in space
and linearly in time. Ocean products mark land with NaN: NaN corners are dropped
and the remaining weights renormalized, so a voyage close to the coast still
feels the neighbouring sea instead of a NaN that would end it. Where only land
surrounds a point the value is ``land_value`` (0 m/s current on land).

GLORYS starts in 1993, so earlier wind years pair with a monthly current
climatology (``monthly_climatology`` builds it, ``MonthlyClimatologyField``
interpolates it between mid-month centres, cyclically through the year).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from . import geo

_LON_NAMES = ("longitude", "lon", "x")
_LAT_NAMES = ("latitude", "lat", "y")
_TIME_NAMES = ("time", "valid_time")


class VectorField(Protocol):
    """Anything that gives a velocity at positions and times."""

    def sample(self, lon, lat, t) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) float64 m/s at 1-D lon, lat, t (epoch hours); NaN outside the field's coverage."""
        ...


@dataclass
class UniformField:
    """The same (u, v) everywhere and always."""

    u: float
    v: float

    def sample(self, lon, lat, t):
        shape = np.broadcast(lon, lat, t).shape
        return _filled(self.u, shape), _filled(self.v, shape)


@dataclass
class FunctionField:
    """An analytic field, mainly for tests: ``fn(lon, lat, t) -> (u, v)``."""

    fn: Callable

    def sample(self, lon, lat, t):
        lon, lat, t = _as_arrays(lon, lat, t)
        u, v = self.fn(lon, lat, t)
        return _filled(u, lon.shape), _filled(v, lon.shape)


class _LonLatGrid:
    """Normalized lon/lat axes over a stack of (u, v) slices: the spatial half of gridded fields."""

    def __init__(self, lon, lat, u, v, land_value, name):
        self.name = name
        raw_lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        u, v = np.asarray(u), np.asarray(v)
        if raw_lon.ndim != 1 or lat.ndim != 1 or raw_lon.size < 2 or lat.size < 2:
            raise ValueError(f"{self._label}: lon and lat must be 1-D with at least 2 points each")
        if u.ndim != 3 or u.shape != v.shape or u.shape[1:] != (lat.size, raw_lon.size):
            raise ValueError(f"{self._label}: u and v must both have shape (k, {lat.size}, {raw_lon.size}); "
                             f"got {u.shape} and {v.shape}")

        lon = geo.wrap_lon(raw_lon)
        order = np.argsort(lon, kind="stable")
        lon = lon[order]
        step = np.diff(lon)
        # Sorting a grid that straddles the antimeridian leaves an interior gap;
        # a global 0..360 grid closes the circle and leaves none.
        if not np.all(step > 0) or step.max() > 1.5 * np.abs(np.diff(raw_lon)).max():
            raise ValueError(f"{self._label}: longitudes must be distinct in [-180, 180) "
                             "and the grid must not straddle the antimeridian")
        dlat = np.diff(lat)
        if np.all(dlat < 0):
            lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
        elif not np.all(dlat > 0):
            raise ValueError(f"{self._label}: latitudes must be strictly monotonic")
        if np.any(order != np.arange(order.size)):
            u, v = u[..., order], v[..., order]

        # float32 stays float32 in memory (a season of GLORYS is large); sampling is float64.
        dtype = np.float32 if u.dtype == np.float32 and v.dtype == np.float32 else np.float64
        self.lon, self.lat = lon, lat
        self.u = np.ascontiguousarray(u, dtype=dtype)
        self.v = np.ascontiguousarray(v, dtype=dtype)
        self.land_value = float(land_value)

    @property
    def _label(self):
        return self.name or type(self).__name__

    @property
    def _plane(self):
        return self.lat.size * self.lon.size

    def _time_range(self):
        return -np.inf, np.inf

    @property
    def bounds(self):
        """(lon_min, lon_max, lat_min, lat_max, t_min, t_max); edges included."""
        t0, t1 = self._time_range()
        return (float(self.lon[0]), float(self.lon[-1]), float(self.lat[0]), float(self.lat[-1]),
                float(t0), float(t1))

    def _interp(self, lon, lat, slices, valid):
        """Sum over weighted slices of bilinear interpolation, NaN corners dropped.

        ``slices`` holds (flat offset of a (ny, nx) slice, its weight) pairs; a
        weight sum of zero over the non-NaN corners gives ``land_value``. Points
        outside the grid or not ``valid`` (outside the time range) are NaN.
        """
        i, fx, in_lon = _locate(self.lon, geo.wrap_lon(lon))
        j, fy, in_lat = _locate(self.lat, lat)
        nx = self.lon.size
        base = j * nx + i
        corners = ((0, (1 - fx) * (1 - fy)), (1, fx * (1 - fy)), (nx, (1 - fx) * fy), (nx + 1, fx * fy))
        u_flat, v_flat = self.u.reshape(-1), self.v.reshape(-1)
        num_u, den_u, num_v, den_v = (np.zeros(base.shape) for _ in range(4))
        for offset, w_t in slices:
            base_t = base + offset
            for corner, w_xy in corners:
                idx = base_t + corner
                w = w_xy * w_t
                for flat, num, den in ((u_flat, num_u, den_u), (v_flat, num_v, den_v)):
                    val = flat[idx]
                    sea = ~np.isnan(val)
                    w_sea = np.where(sea, w, 0.0)
                    num += w_sea * np.where(sea, val, 0.0)
                    den += w_sea
        outside = ~(in_lon & in_lat & valid)
        u = _renormalized(num_u, den_u, self.land_value)
        v = _renormalized(num_v, den_v, self.land_value)
        u[outside] = np.nan
        v[outside] = np.nan
        return u, v


class GriddedField(_LonLatGrid):
    """Wind or current on a rectilinear lon/lat grid at a sequence of times.

    lon (nx,), lat (ny,), time (nt,) epoch hours, u and v (nt, ny, nx) m/s.
    Longitudes in 0..360 and descending latitudes are normalized to [-180, 180)
    and ascending order. Spacing need not be uniform. One time slice (nt == 1),
    or ``time=None`` with (ny, nx) arrays, is a time-invariant field.
    """

    def __init__(self, lon, lat, time, u, v, *, land_value=0.0, name=""):
        u, v = np.asarray(u), np.asarray(v)
        if time is None and u.ndim == 2:
            u, v = u[None], v[None]
        super().__init__(lon, lat, u, v, land_value, name)
        nt = self.u.shape[0]
        if time is None:
            if nt != 1:
                raise ValueError(f"{self._label}: time=None needs a single (ny, nx) slice, got {nt}")
        else:
            time = _epoch_hours(time, self._label)
            if time.shape != (nt,) or not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0):
                raise ValueError(f"{self._label}: time must be finite, strictly increasing "
                                 f"and match u's first axis ({nt})")
        self.time = time

    @classmethod
    def from_dataset(cls, ds, u, v, **kw) -> "GriddedField":
        """Build from the variables ``u`` and ``v`` of an xarray Dataset.

        Dimension names are detected (longitude|lon|x, latitude|lat|y,
        time|valid_time); without a time dimension the field is time-invariant.
        Any other dimension must have length 1 (GLORYS's surface depth) and is
        squeezed. Times must be datetime64 (decoded), and become epoch hours.
        """
        da = ds[u]
        x, y = _find_dim(da, _LON_NAMES), _find_dim(da, _LAT_NAMES)
        if x not in da.coords or y not in da.coords:
            # Without coordinate values xarray hands back 0, 1, 2, ..., which would pass for degrees.
            raise ValueError(f"{u}: dimensions {x!r} and {y!r} need longitude and latitude coordinate values")
        t = _find_dim(da, _TIME_NAMES, required=False)
        time = None
        if t is not None:
            tv = da[t].values
            if not np.issubdtype(tv.dtype, np.datetime64):
                raise TypeError(f"{u}: time coordinate {t!r} is {tv.dtype}, not datetime64; "
                                "open the data with decoded times")
            time = geo.to_epoch_hours(tv)
        dims = tuple(d for d in (t, y, x) if d is not None)
        kw.setdefault("name", f"{u},{v}")
        return cls(da[x].values, da[y].values, time, _values(da, dims), _values(ds[v], dims), **kw)

    def _time_range(self):
        if self.u.shape[0] == 1:
            return -np.inf, np.inf
        return self.time[0], self.time[-1]

    def sample(self, lon, lat, t):
        lon, lat, t = _as_arrays(lon, lat, t)
        if self.u.shape[0] == 1:
            return self._interp(lon, lat, ((0, 1.0),), True)
        k, ft, in_time = _locate(self.time, t)
        return self._interp(lon, lat, ((k * self._plane, 1 - ft), ((k + 1) * self._plane, ft)), in_time)


class MonthlyClimatologyField(_LonLatGrid):
    """A field that repeats every year, from twelve monthly means.

    lon, lat as for GriddedField; u12, v12 (12, ny, nx), index 0 = January.
    Each monthly mean sits at its month's centre; between centres the field is
    linear in time, cyclic from December to January. Centres come from the
    calendar date of t with actual month lengths (leap years included), so the
    field is valid for any year.
    """

    def __init__(self, lon, lat, u12, v12, *, land_value=0.0, name=""):
        super().__init__(lon, lat, u12, v12, land_value, name)
        if self.u.shape[0] != 12:
            raise ValueError(f"{self._label}: u12 and v12 need 12 monthly slices, got {self.u.shape[0]}")

    def sample(self, lon, lat, t):
        lon, lat, t = _as_arrays(lon, lat, t)
        finite = np.isfinite(t)
        k, f = _month_position(np.where(finite, t, 0.0))
        slices = ((k * self._plane, 1 - f), (((k + 1) % 12) * self._plane, f))
        return self._interp(lon, lat, slices, finite)


def monthly_climatology(time, u, v) -> tuple[np.ndarray, np.ndarray]:
    """Mean by calendar month of (nt, ny, nx) data at epoch-hour ``time`` -> (u12, v12).

    Index 0 is January. NaN-aware: each cell averages its valid samples, every
    sample weighted equally (daily data give a mean over days); a cell with no
    valid sample in a month is NaN there.
    """
    month = geo.month_of(_epoch_hours(time, "monthly_climatology")) - 1
    return _monthly_mean(np.asarray(u), month), _monthly_mean(np.asarray(v), month)


@dataclass
class Environment:
    """The forcing a simulation runs in."""

    wind: VectorField  # 10 m wind
    current: VectorField  # surface current
    current_is_climatology: bool = False  # flagged in outputs (pre-1993 runs)


# --- helpers --------------------------------------------------------------------

def _as_arrays(*xs):
    return np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in xs))


def _epoch_hours(time, label):
    """``time`` as float64 epoch hours; datetime64 is refused, since casting it would read its ticks as hours."""
    time = np.asarray(time)
    if np.issubdtype(time.dtype, np.datetime64):
        raise TypeError(f"{label}: time must be epoch hours, not {time.dtype}; convert with geo.to_epoch_hours")
    return time.astype(np.float64)


def _filled(x, shape):
    """A writable float64 array of ``shape`` from a scalar or broadcastable array."""
    return np.broadcast_to(np.asarray(x, dtype=np.float64), shape).copy()


def _locate(axis, x):
    """Cell index along an ascending axis (clipped to real cells), position within the cell, inside mask.

    A point on the last grid line falls in the last cell at position 1, so it is inside.
    """
    i = np.clip(np.searchsorted(axis, x, side="right") - 1, 0, axis.size - 2)
    lo = axis[i]
    return i, (x - lo) / (axis[i + 1] - lo), (x >= axis[0]) & (x <= axis[-1])


def _renormalized(num, den, land_value):
    out = np.full(num.shape, land_value)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _month_position(t):
    """Month (0 = January) whose centre precedes t, and t's fraction of the way to the next centre."""
    month = geo.from_epoch_hours(t).astype("datetime64[M]")
    starts = geo.to_epoch_hours(month + np.arange(-1, 3).reshape((4,) + (1,) * month.ndim))
    centres = 0.5 * (starts[:-1] + starts[1:])  # previous, this and next month
    later = t >= centres[1]
    lo = np.where(later, centres[1], centres[0])
    hi = np.where(later, centres[2], centres[1])
    m = month.astype(np.int64) % 12
    return np.where(later, m, (m - 1) % 12), (t - lo) / (hi - lo)


def _monthly_mean(x, month):
    if x.ndim != 3 or x.shape[0] != month.size:
        raise ValueError(f"data must be (nt, ny, nx) with nt = {month.size} times; got {x.shape}")
    out = np.full((12,) + x.shape[1:], np.nan, dtype=np.float32 if x.dtype == np.float32 else np.float64)
    for m in range(12):
        sel = x[month == m]
        count = np.count_nonzero(~np.isnan(sel), axis=0)
        total = np.nansum(sel, axis=0, dtype=np.float64)
        out[m] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def _find_dim(da, names, required=True):
    by_lower = {str(d).lower(): d for d in da.dims}
    for name in names:
        if name in by_lower:
            return by_lower[name]
    if required:
        raise ValueError(f"{da.name}: no dimension named any of {names}; dimensions are {da.dims}")
    return None


def _values(da, dims):
    """The variable's values ordered as ``dims``, other length-1 dimensions squeezed out."""
    extra = [d for d in da.dims if d not in dims]
    long = [d for d in extra if da.sizes[d] != 1]
    if long:
        raise ValueError(f"{da.name}: dimensions {long} must have length 1 (select one level first)")
    return da.isel({d: 0 for d in extra}, drop=True).transpose(*dims).values
