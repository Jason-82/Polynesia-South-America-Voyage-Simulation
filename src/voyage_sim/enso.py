"""ENSO climate-state labels for launch dates.

The grid reports success odds separately for El Nino, La Nina and neutral
years. The labels come from the Oceanic Nino Index (ONI): the 3-month running
mean of Nino 3.4 sea-surface temperature anomalies. NOAA's rule makes a month
part of an El Nino (La Nina) episode when it belongs to a run of at least five
consecutive overlapping 3-month seasons with ONI >= +0.5 (<= -0.5) degC.

Two NOAA text formats are read: CPC's ``oni.ascii.txt`` (ONI by season) and
PSL's year-plus-twelve-values format (any monthly index; a raw Nino 3.4 anomaly
series becomes ONI-like through ``running_mean_3``). Every series is indexed by
calendar month as a monthly ``pd.PeriodIndex``; a 3-month season is filed under
its centre month (DJF 1950 -> 1950-01, NDJ 1950 -> 1950-12).
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


def parse_cpc_oni(text: str) -> pd.Series:
    """CPC ``oni.ascii.txt`` -> ONI anomaly (degC) indexed by each season's centre month."""
    months, values = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].upper() not in _SEASON_CENTRE:
            continue  # header or blank line
        anom = float(parts[3])
        months.append(pd.Period(year=int(parts[1]), month=_SEASON_CENTRE[parts[0].upper()], freq="M"))
        values.append(np.nan if anom <= _CPC_MISSING else anom)
    s = pd.Series(values, index=pd.PeriodIndex(months, freq="M", name="month"), dtype=float, name="oni")
    return s.sort_index()


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
    """NOAA episode rule on ONI values (already 3-month means): one phase label per month.

    Months are El Nino (La Nina) inside runs of at least ``min_run`` consecutive
    months with ONI >= +threshold (<= -threshold), neutral otherwise. A missing
    month breaks a run; months without an ONI value are left out of the result,
    so ``label_times`` reports them as unknown.
    """
    s = _monthly(oni)
    x = s.to_numpy()
    with np.errstate(invalid="ignore"):
        warm = _in_long_run(x >= threshold - _THRESHOLD_TOL, min_run)
        cold = _in_long_run(x <= -threshold + _THRESHOLD_TOL, min_run)
    labels = np.where(warm, EL_NINO, np.where(cold, LA_NINA, NEUTRAL))
    out = pd.Series(labels, index=s.index, name="enso")
    return out[np.isfinite(x)]


def label_times(t_hours, phases: pd.Series) -> np.ndarray:
    """Phase of the calendar month (UTC) of each epoch-hour time; ``"unknown"`` where unlabelled."""
    t = np.atleast_1d(np.asarray(t_hours, dtype=float))
    finite = np.isfinite(t)
    months = geo.from_epoch_hours(np.where(finite, t, 0.0)).astype("datetime64[M]").astype(np.int64)
    p = phases.dropna()
    if not isinstance(p.index, pd.PeriodIndex):
        p.index = pd.DatetimeIndex(p.index).to_period("M")
    p = p.sort_index()
    known = p.index.asfreq("M").asi8  # monthly period ordinals count months from 1970-01
    out = np.full(t.shape, UNKNOWN, dtype=object)
    if len(known) == 0:
        return out
    pos = np.clip(np.searchsorted(known, months), 0, len(known) - 1)
    found = finite & (known[pos] == months)
    out[found] = p.to_numpy(dtype=object)[pos[found]]
    return out
