"""ENSO climate-state labels for launch dates.

The grid reports success odds separately for El Nino, La Nina and neutral
launches. The main labels come from NOAA's Relative Oceanic Nino Index (RONI),
repeated with the older Oceanic Nino Index (ONI): both are 3-month running means
of Nino 3.4 sea-surface temperature anomalies, RONI with the tropical-wide mean
removed. NOAA's rule makes a month part of an El Nino (La Nina) episode when it
belongs to a run of at least five consecutive overlapping 3-month seasons with
the index >= +0.5 (<= -0.5) degC.

Both indices start in 1950. For the 1940s, PSL's HadISST Nino 3.4 anomaly is
re-centred on 1991-2020 (``recentre``), averaged over 3 months
(``running_mean_3``) and put through the same rule; those labels come from a
different index and are flagged as such by the caller.

Two NOAA text formats are read: CPC's seasonal files (``oni.ascii.txt``,
``RONI.ascii.txt``) and PSL's year-plus-twelve-values format. Every series is
indexed by calendar month as a monthly ``pd.PeriodIndex``; a 3-month season is
filed under its centre month (DJF 1950 -> 1950-01, NDJ 1950 -> 1950-12).
A launch takes the label and index value of its calendar month (UTC), so an
April launch takes the MAM season.

Bootstrap intervals resample July-June years (``enso_year``): an event usually
grows through the northern summer, peaks around December and decays by the
next spring, so a July-June year keeps an event's launches in one cluster
where a calendar year would split them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import geo

EL_NINO, LA_NINA, NEUTRAL, UNKNOWN = "el_nino", "la_nina", "neutral", "unknown"

# Season code -> centre month (1-12). CPC files a season under its centre
# month's year: DJF 1950 is Dec 1949 - Feb 1950, NDJ 1950 is Nov 1950 - Jan 1951.
_SEASON_CENTRE = {
    code: i + 1
    for i, code in enumerate(
        ("DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ")
    )
}
_CPC_MISSING = -99.0  # CPC writes -99.9 for seasons not yet computed

# Means of two-decimal data can land a rounding error short of the threshold.
_THRESHOLD_TOL = 1e-9


def _monthly(series: pd.Series) -> pd.Series:
    """Index ``series`` by monthly periods over its full span, NaN-filling gaps."""
    s = series.copy()
    if not isinstance(s.index, pd.PeriodIndex):
        s.index = pd.DatetimeIndex(s.index).to_period("M")
    s.index = s.index.asfreq("M")
    s = s.sort_index()
    if s.empty:
        return s.astype(float)
    full = pd.period_range(s.index[0], s.index[-1], freq="M", name="month")
    return s.reindex(full).astype(float)


def parse_cpc_seasonal(text: str) -> pd.Series:
    """CPC seasonal index text -> anomaly (degC) indexed by each season's centre month.

    Reads ``oni.ascii.txt`` (``SEAS YR TOTAL ANOM``) and ``RONI.ascii.txt``
    (``SEAS YR ANOM``) alike: the anomaly is the last column.
    """
    months, values = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0].upper() not in _SEASON_CENTRE:
            continue  # header or blank line
        anom = float(parts[-1])
        months.append(pd.Period(year=int(parts[1]), month=_SEASON_CENTRE[parts[0].upper()], freq="M"))
        values.append(np.nan if anom <= _CPC_MISSING else anom)
    s = pd.Series(values, index=pd.PeriodIndex(months, freq="M", name="month"), dtype=float, name="anom")
    return s.sort_index()


parse_cpc_oni = parse_cpc_seasonal


def parse_psl_monthly(text: str) -> pd.Series:
    """PSL monthly timeseries text -> monthly series, NaN where the file's missing value appears.

    The format is a ``start_year end_year`` line, one ``YYYY v1 .. v12`` row per
    year, a line holding the missing-value flag, then free-text notes.
    """
    lines = [ln.split() for ln in text.splitlines() if ln.strip()]
    start, end = (int(x) for x in lines[0][:2])
    if len(lines) < end - start + 3:
        raise ValueError(f"PSL text ends before its {start}-{end} rows and missing-value line")
    rows = []
    for year, parts in zip(range(start, end + 1), lines[1:]):
        if int(parts[0]) != year or len(parts) < 13:
            raise ValueError(f"PSL row for {year} malformed: {' '.join(parts)!r}")
        rows.append([float(x) for x in parts[1:13]])
    missing = float(lines[end - start + 2][0])
    values = np.array(rows, dtype=float).ravel()
    values[np.isclose(values, missing)] = np.nan
    index = pd.period_range(f"{start}-01", f"{end}-12", freq="M", name="month")
    return pd.Series(values, index=index)


def recentre(series: pd.Series, base: tuple[int, int] = (1991, 2020)) -> pd.Series:
    """Monthly anomalies re-expressed against the ``base`` years (inclusive).

    Each calendar month loses its mean over the base years, so the series
    becomes an anomaly from the 1991-2020 climatology whatever base period it
    was published on. Every calendar month needs a value in the base years.
    """
    s = _monthly(series)
    months = s.index.month
    in_base = (s.index.year >= base[0]) & (s.index.year <= base[1])
    clim = s[in_base].groupby(months[in_base]).mean()
    missing = sorted(set(range(1, 13)) - set(clim.dropna().index))
    if missing:
        raise ValueError(f"no {base[0]}-{base[1]} values for calendar months {missing}")
    return s - clim.reindex(months).to_numpy()


def running_mean_3(series: pd.Series) -> pd.Series:
    """Centred 3-month mean over consecutive calendar months; NaN unless all three exist."""
    return _monthly(series).rolling(3, center=True, min_periods=3).mean()


def _in_long_run(mask: np.ndarray, min_run: int) -> np.ndarray:
    """True where ``mask`` is True and part of a run of at least ``min_run`` Trues."""
    starts = mask & ~np.concatenate([[False], mask[:-1]])
    run_id = np.cumsum(starts)  # 1, 2, ... for the runs; meaningless where mask is False
    lengths = np.bincount(run_id[mask], minlength=len(mask) + 1)
    return mask & (lengths[run_id] >= min_run)


def enso_phases(oni: pd.Series, threshold: float = 0.5, min_run: int = 5) -> pd.Series:
    """NOAA episode rule on index values (already 3-month means): one phase label per month.

    Months are El Nino (La Nina) inside runs of at least ``min_run`` consecutive
    months with the index >= +threshold (<= -threshold), neutral otherwise. A
    missing month breaks a run; months without a value are left out of the
    result, so ``label_times`` reports them as unknown.
    """
    s = _monthly(oni)
    x = s.to_numpy()
    with np.errstate(invalid="ignore"):
        warm = _in_long_run(x >= threshold - _THRESHOLD_TOL, min_run)
        cold = _in_long_run(x <= -threshold + _THRESHOLD_TOL, min_run)
    labels = np.where(warm, EL_NINO, np.where(cold, LA_NINA, NEUTRAL))
    out = pd.Series(labels, index=s.index, name="enso")
    return out[np.isfinite(x)]


def _at_months(t_hours, series: pd.Series, fill, dtype) -> np.ndarray:
    """Value of ``series`` (monthly, or timestamped) at the UTC calendar month of each time; ``fill`` where absent."""
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    finite = np.isfinite(t)
    months = geo.from_epoch_hours(np.where(finite, t, 0.0)).astype("datetime64[M]").astype(np.int64)
    s = series.dropna()
    if not isinstance(s.index, pd.PeriodIndex):
        s.index = pd.DatetimeIndex(s.index).to_period("M")
    s = s.sort_index()
    known = s.index.asfreq("M").asi8  # monthly period ordinals count months from 1970-01
    out = np.full(t.shape, fill, dtype=dtype)
    if len(known) == 0:
        return out
    pos = np.clip(np.searchsorted(known, months), 0, len(known) - 1)
    found = finite & (known[pos] == months)
    out[found] = s.to_numpy(dtype=dtype)[pos[found]]
    return out


def label_times(t_hours, phases: pd.Series) -> np.ndarray:
    """Phase of the calendar month (UTC) of each epoch-hour time; ``"unknown"`` where unlabelled."""
    return _at_months(t_hours, phases, UNKNOWN, object)


def index_at_times(t_hours, index: pd.Series) -> np.ndarray:
    """Index value of the calendar month (UTC) of each epoch-hour time; NaN where the index has none."""
    return _at_months(t_hours, index, np.nan, np.float64)


def enso_year(t_hours) -> np.ndarray:
    """July-June year of each epoch-hour time, named by its July (Jul 1997 - Jun 1998 -> 1997)."""
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    if not np.all(np.isfinite(t)):
        raise ValueError("enso_year needs finite times")
    months = geo.from_epoch_hours(t).astype("datetime64[M]").astype(np.int64)  # months since 1970-01
    return (months - 6) // 12 + 1970
