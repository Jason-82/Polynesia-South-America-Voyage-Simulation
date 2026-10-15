"""Success odds, durations and their intervals from simulated voyages.

A voyage's track does not depend on the sighting rule or the supply limit, so
one run answers every combination: success under rule r with S days of
supplies is ``hit_days[r] <= S`` (a sighting exactly at the limit counts),
optionally restricted to landfalls in chosen island groups or coast segments.

Voyages launched in the same season share its weather and ocean state, so
they are not independent trials. Intervals therefore come from a cluster
bootstrap that resamples whole July-June launch years (``enso.enso_year``, so
one El Nino or La Nina event is one cluster) and recomputes the pooled success
fraction (a ratio of sums); an ordinary bootstrap over voyages would be too
narrow whenever outcomes are correlated within a year. Because the record
holds only a handful of strong events, El Nino and La Nina cells also report
how many distinct events their voyages come from.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from . import enso, geo
from .simulate import DID_NOT_SAIL


def _rule_index(results, rule) -> int:
    if isinstance(rule, (int, np.integer)):
        return int(rule)
    return list(results.rule_names).index(rule)


def success(results, rule, supply_days, dest_groups=None) -> np.ndarray:
    """True where land was sighted within ``supply_days`` (inclusive) under ``rule``.

    ``rule`` is an index or a name in ``results.rule_names``. With
    ``dest_groups`` (group names), only landfalls in those groups count; a name
    that is not a group of any landmass raises ``ValueError``.
    """
    r = _rule_index(results, rule)
    days = np.asarray(results.hit_days)[r]
    with np.errstate(invalid="ignore"):
        ok = days <= supply_days  # NaN (no sighting) compares False
    if dest_groups is not None:
        if isinstance(dest_groups, str):
            dest_groups = [dest_groups]
        dest_groups = list(dest_groups)
        groups = np.asarray(results.landmass_groups, dtype=object)
        # A misspelt destination would otherwise read as zero odds in every cell.
        unknown = sorted(set(dest_groups) - set(groups.tolist()), key=str)
        if unknown:
            raise ValueError(f"destination groups {unknown} match no landmass group")
        in_dest = np.isin(groups, dest_groups)
        # Index -1 (no landmass) reads the appended False.
        ok &= np.append(in_dest, False)[np.asarray(results.hit_landmass)[r]]
    return ok


def cluster_bootstrap(success, clusters, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    """Success fraction with a percentile interval from resampling whole clusters.

    Each replicate draws as many clusters as there are, with replacement, and
    takes total successes over total voyages in the drawn clusters. Returns
    ``p, lo, hi, n, n_clusters``; ``lo`` and ``hi`` are NaN with fewer than two
    clusters, where the resampling has nothing to vary.
    """
    s = np.asarray(success, dtype=bool)
    n = s.size
    if n == 0:
        return dict(p=np.nan, lo=np.nan, hi=np.nan, n=0, n_clusters=0)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    inv = inv.ravel()
    k = int(inv.max()) + 1
    hits = np.bincount(inv, weights=s, minlength=k)
    counts = np.bincount(inv, minlength=k).astype(float)
    p = float(s.mean())
    if k < 2:
        return dict(p=p, lo=np.nan, hi=np.nan, n=n, n_clusters=k)
    draw = np.random.default_rng(seed).integers(0, k, size=(n_boot, k))
    p_star = hits[draw].sum(axis=1) / counts[draw].sum(axis=1)
    alpha = 100.0 * (1.0 - level) / 2.0
    lo, hi = np.percentile(p_star, [alpha, 100.0 - alpha])
    return dict(p=p, lo=float(lo), hi=float(hi), n=n, n_clusters=k)


def duration_summary(days) -> dict:
    """Median and 90th percentile (linear interpolation) of the finite values, and their count."""
    d = np.asarray(days, dtype=float).ravel()
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(median=np.nan, p90=np.nan, n=0)
    median, p90 = np.percentile(d, [50.0, 90.0])
    return dict(median=float(median), p90=float(p90), n=int(d.size))


def launch_labels(results, phases: pd.Series | None = None, index: pd.Series | None = None) -> pd.DataFrame:
    """Per-voyage labels for ``summarize``, from each voyage's launch time (UTC).

    Columns: ``voyage_id``; ``month`` (1-12); ``enso``, the phase of the launch
    month in ``phases`` (``enso.enso_phases`` output; ``"unknown"`` where it has
    none, or without it); ``year``, the July-June year (``enso.enso_year``) that
    clusters the bootstrap; ``index``, the launch month's index value (NaN where
    missing, or without ``index``); and ``event``, the El Nino or La Nina event
    the launch falls in, named by phase and first month (``"el_nino:1997-05"``),
    missing outside one.
    """
    t = np.asarray(results.launch_t, dtype=float)
    n = t.size
    if phases is None:
        phase = np.full(n, enso.UNKNOWN, dtype=object)
        event = np.full(n, None, dtype=object)
    else:
        phase = enso.label_times(t, phases)
        event = enso.label_times(t, _events(phases))
        event[event == enso.UNKNOWN] = None
    return pd.DataFrame(
        {
            "voyage_id": np.asarray(results.voyage_id),
            "month": geo.month_of(t),
            "enso": phase,
            "year": enso.enso_year(t),
            "index": np.full(n, np.nan) if index is None else enso.index_at_times(t, index),
            "event": event,
        }
    )


def _events(phases: pd.Series) -> pd.Series:
    """Event name of each El Nino or La Nina month: its phase and the first month of its run.

    An event is a maximal run of consecutive months with the same non-neutral
    phase; a month missing from ``phases`` ends it, as it does in ``enso_phases``.
    """
    p = phases.dropna()
    if not isinstance(p.index, pd.PeriodIndex):
        p.index = pd.DatetimeIndex(p.index).to_period("M")
    p = p.sort_index()
    p.index = p.index.asfreq("M")
    phase = p.to_numpy(dtype=object)
    new = np.ones(len(p), dtype=bool)
    new[1:] = (phase[1:] != phase[:-1]) | (np.diff(p.index.asi8) != 1)
    first = np.maximum.accumulate(np.where(new, np.arange(len(p)), 0))
    names = np.array([f"{ph}:{start}" for ph, start in zip(phase[first], p.index[first])], dtype=object)
    keep = np.isin(phase, [enso.EL_NINO, enso.LA_NINA])
    return pd.Series(names[keep], index=p.index[keep], dtype=object)


def summarize(
    results,
    labels: pd.DataFrame,
    rules: Sequence,
    supplies: Iterable[float],
    dest_groups=None,
    by: Sequence[str] = ("month", "enso"),
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> pd.DataFrame:
    """One row per (rule, supply limit, ``by`` cell): odds, interval and days at sea.

    ``labels`` has one row per voyage, in results order or with a ``voyage_id``
    column to align on, holding the ``by`` columns and ``year``, the bootstrap
    cluster. Pass the July-June year of each launch there (``enso.enso_year``,
    as ``launch_labels`` does), so one El Nino or La Nina event is one cluster.

    ``n_years`` counts the distinct clusters behind a cell. ``n_events`` counts
    the distinct values of the labels' ``event`` column among the cell's
    voyages in El Nino and La Nina cells (``enso`` among ``by``); it is NA in
    other cells and without that column. ``share_not_sailed`` is the share of
    the cell's voyages that gave up waiting in port (``DID_NOT_SAIL``); they
    count as failures. Durations are the days to first sighting of the
    successful voyages in the cell. Rows are ordered by rule, supply limit,
    then sorted cells; every cell reuses ``seed``, so reruns give identical
    intervals.
    """
    by = [by] if isinstance(by, str) else list(by)
    n = len(np.asarray(results.voyage_id))
    if "voyage_id" in labels.columns:
        labels = labels.set_index("voyage_id").loc[np.asarray(results.voyage_id)].reset_index()
    if len(labels) != n:
        raise ValueError(f"labels has {len(labels)} rows for {n} voyages")
    years = labels["year"].to_numpy()
    if by:
        groups = labels.groupby(by, sort=True, dropna=False, observed=True).indices
        cells = [(k if isinstance(k, tuple) else (k,), idx) for k, idx in groups.items()]
    else:
        cells = [((), np.arange(n))]

    not_sailed = np.asarray(results.status) == DID_NOT_SAIL
    events = labels["event"].to_numpy(dtype=object) if "event" in labels.columns else None
    cell_counts = []
    for key, idx in cells:
        n_events = pd.NA
        if events is not None and "enso" in by and key[by.index("enso")] in (enso.EL_NINO, enso.LA_NINA):
            e = events[idx]
            n_events = len(set(e[pd.notna(e)]))
        cell_counts.append((n_events, float(not_sailed[idx].mean()) if len(idx) else np.nan))

    rows = []
    for rule in rules:
        r = _rule_index(results, rule)
        days = np.asarray(results.hit_days)[r]
        for supply in supplies:
            ok = success(results, r, supply, dest_groups)
            for (key, idx), (n_events, share_not_sailed) in zip(cells, cell_counts):
                ci = cluster_bootstrap(ok[idx], years[idx], n_boot=n_boot, seed=seed, level=level)
                dur = duration_summary(days[idx][ok[idx]])
                rows.append(
                    (results.rule_names[r], supply, *key, ci["p"], ci["lo"], ci["hi"], ci["n"],
                     ci["n_clusters"], n_events, share_not_sailed, dur["median"], dur["p90"])
                )
    columns = ["rule", "supply_days", *by, "p", "lo", "hi", "n", "n_years", "n_events", "share_not_sailed",
               "median_days", "p90_days"]
    df = pd.DataFrame(rows, columns=columns)
    df["n_events"] = df["n_events"].astype("Int64")
    return df
