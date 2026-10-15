from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from voyage_sim import analyze, enso, geo
from voyage_sim.simulate import AT_SEA, DID_NOT_SAIL, END_ALL_SIGHTED

NAN = np.nan
GROUPS = ["Marquesas", "Marquesas", "Coast: Ecuador"]  # landmass index -> group


def _results():
    """Eight voyages, two sighting rules; landmasses 0-1 are Marquesas, 2 is Ecuador."""
    return SimpleNamespace(
        voyage_id=np.arange(100, 108),
        launch_t=np.zeros(8),
        status=np.zeros(8, dtype=int),
        rule_names=["r20", "r40"],
        landmass_groups=GROUPS,
        hit_days=np.array(
            [
                [10.0, 30.0, NAN, 45.0, 5.0, NAN, 60.0, 61.0],
                [8.0, 25.0, 70.0, 40.0, 4.0, 80.0, 55.0, 58.0],
            ]
        ),
        hit_landmass=np.array([[0, 2, -1, 1, 0, -1, 2, 2], [0, 2, 1, 1, 0, 2, 2, 2]]),
    )


def _labels():
    return pd.DataFrame(
        {
            "voyage_id": np.arange(100, 108),
            "month": [1, 1, 1, 1, 2, 2, 2, 2],
            "year": [2000, 2000, 2001, 2001, 2000, 2000, 2001, 2001],
        }
    )


# --- success -------------------------------------------------------------------

def test_sighting_exactly_at_supply_limit_counts():
    res = SimpleNamespace(
        rule_names=["r40"],
        landmass_groups=GROUPS,
        hit_days=np.array([[30.0, 30.0 + 1e-9, 29.0, NAN]]),
        hit_landmass=np.array([[0, 0, 0, -1]]),
    )
    np.testing.assert_array_equal(analyze.success(res, 0, 30), [True, False, True, False])
    np.testing.assert_array_equal(analyze.success(res, "r40", 30.0), analyze.success(res, 0, 30))


def test_dest_groups_filter_and_no_landmass_never_matches_the_last_group():
    res = _results()
    ok = analyze.success(res, "r40", 60, dest_groups=["Coast: Ecuador"])
    np.testing.assert_array_equal(ok, [False, True, False, False, False, False, True, True])
    np.testing.assert_array_equal(analyze.success(res, "r40", 60, dest_groups="Coast: Ecuador"), ok)
    # Index -1 must not wrap around to the last landmass (an Ecuador coast segment).
    res.hit_landmass[0, 0] = -1
    assert not analyze.success(res, "r20", 60, dest_groups=["Coast: Ecuador"])[0]
    assert not analyze.success(res, "r20", 60, dest_groups=[]).any()


def test_unknown_destination_group_is_an_error_not_zero_odds():
    res = _results()
    with pytest.raises(ValueError, match="Equador"):
        analyze.success(res, "r40", 60, dest_groups=["Coast: Equador"])
    # Groups coded as indices rather than names must not silently match nothing.
    res.landmass_groups = np.array([0, 0, 1])
    with pytest.raises(ValueError):
        analyze.summarize(res, _labels(), ["r40"], [60], dest_groups="Coast: Ecuador", by=())


# --- cluster bootstrap -----------------------------------------------------------

def test_bootstrap_is_deterministic_and_reports_counts():
    rng = np.random.default_rng(1)
    s = rng.random(300) < 0.3
    years = np.repeat(np.arange(1990, 2000), 30)
    a = analyze.cluster_bootstrap(s, years, seed=7)
    assert a == analyze.cluster_bootstrap(s, years, seed=7)
    assert a != analyze.cluster_bootstrap(s, years, seed=8)
    assert a["n"] == 300 and a["n_clusters"] == 10
    assert a["p"] == pytest.approx(s.mean())
    assert a["lo"] <= a["p"] <= a["hi"]
    narrow = analyze.cluster_bootstrap(s, years, seed=7, level=0.5)
    assert a["lo"] < narrow["lo"] <= narrow["hi"] < a["hi"]


@pytest.mark.parametrize("p_true", [0.05, 0.3, 0.7])
def test_interval_contains_the_estimate(p_true):
    rng = np.random.default_rng(int(p_true * 100))
    s = rng.random(400) < p_true
    clusters = rng.integers(0, 15, 400)
    ci = analyze.cluster_bootstrap(s, clusters)
    assert ci["lo"] <= ci["p"] <= ci["hi"]


def test_interval_width_matches_the_analytic_sampling_sd_of_cluster_means():
    # Equal-size clusters: p* is the mean of k resampled cluster fractions, with
    # sd = std(q) / sqrt(k); a 95% interval spans about 2 * 1.96 sd.
    k, m = 20, 50
    q = np.round(np.linspace(0.0, 0.76, k) * m) / m
    s = (np.arange(m)[None, :] < (q * m)[:, None]).ravel()
    years = np.repeat(np.arange(k), m)
    ci = analyze.cluster_bootstrap(s, years, n_boot=20_000)
    sd = q.std() / np.sqrt(k)
    assert (ci["hi"] - ci["lo"]) / (2 * 1.959964 * sd) == pytest.approx(1.0, abs=0.08)
    assert 0.5 * (ci["lo"] + ci["hi"]) == pytest.approx(ci["p"], abs=0.2 * sd)


def test_within_year_correlation_widens_the_interval():
    k, m = 20, 50
    years = np.repeat(np.arange(k), m)
    # Whole years succeed or fail together: effectively 20 trials, not 1000.
    correlated = np.repeat(np.arange(k) < 8, m)
    clustered = analyze.cluster_bootstrap(correlated, years)
    iid = analyze.cluster_bootstrap(correlated, np.arange(k * m))
    assert clustered["p"] == iid["p"] == pytest.approx(0.4)
    assert clustered["hi"] - clustered["lo"] > 4 * (iid["hi"] - iid["lo"])
    # Independent outcomes: clustering by year changes the width little.
    independent = np.random.default_rng(3).random(k * m) < 0.4
    w_year = np.ptp([analyze.cluster_bootstrap(independent, years)[x] for x in ("lo", "hi")])
    w_iid = np.ptp([analyze.cluster_bootstrap(independent, np.arange(k * m))[x] for x in ("lo", "hi")])
    assert 0.7 < w_year / w_iid < 1.4


def test_bootstrap_pools_voyages_over_clusters_of_unequal_size():
    # A ratio of sums over the drawn clusters: one successful voyage alone in its year weighs one voyage of
    # 901, not a tenth of the sample (as a mean of per-year fractions would make it).
    s = np.concatenate([[True], np.tile(np.arange(100) < 10, 9)])
    years = np.concatenate([[0], np.repeat(np.arange(1, 10), 100)])
    ci = analyze.cluster_bootstrap(s, years, n_boot=4000, seed=0)
    assert ci["p"] == pytest.approx(91 / 901) and ci["hi"] < 0.11
    # Exactly the reference resampling, with the same draws.
    hits, counts = np.bincount(years, weights=s), np.bincount(years).astype(float)
    draw = np.random.default_rng(0).integers(0, 10, size=(4000, 10))
    assert (ci["lo"], ci["hi"]) == tuple(np.percentile(hits[draw].sum(1) / counts[draw].sum(1), [2.5, 97.5]))


def test_bootstrap_degenerate_inputs():
    one = analyze.cluster_bootstrap([True, False, True], [1999, 1999, 1999])
    assert one["p"] == pytest.approx(2 / 3) and one["n_clusters"] == 1
    assert np.isnan(one["lo"]) and np.isnan(one["hi"])
    empty = analyze.cluster_bootstrap(np.zeros(0, bool), np.zeros(0, int))
    assert empty["n"] == 0 and empty["n_clusters"] == 0
    assert all(np.isnan(empty[x]) for x in ("p", "lo", "hi"))


# --- durations -------------------------------------------------------------------

def test_duration_summary_uses_finite_values_and_linear_percentiles():
    d = analyze.duration_summary(np.r_[np.arange(1.0, 11.0), NAN, np.inf])
    assert d == {"median": pytest.approx(5.5), "p90": pytest.approx(9.1), "n": 10}
    empty = analyze.duration_summary([NAN])
    assert empty["n"] == 0 and np.isnan(empty["median"]) and np.isnan(empty["p90"])


# --- summarize -------------------------------------------------------------------

def _row(df, rule, supply, month):
    hit = df[(df.rule == rule) & (df.supply_days == supply) & (df.month == month)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_summarize_hand_worked_cells():
    df = analyze.summarize(_results(), _labels(), rules=["r20", 1], supplies=[30, 60], by=("month",))
    assert list(df.columns) == [
        "rule", "supply_days", "month", "p", "lo", "hi", "n", "n_years", "n_events", "share_not_sailed",
        "median_days", "p90_days",
    ]
    assert len(df) == 2 * 2 * 2
    assert list(df.rule) == ["r20"] * 4 + ["r40"] * 4
    assert list(df.supply_days) == [30, 30, 60, 60] * 2

    # r20, 30 days, January: sightings on days 10 and 30 (inclusive) of 4 voyages.
    r = _row(df, "r20", 30, 1)
    assert (r.p, r.n, r.n_years) == (0.5, 4, 2)
    assert (r.median_days, r.p90_days) == pytest.approx((20.0, 28.0))
    # One year all successes, the other none: replicates are 0, 0.5 or 1.
    assert (r.lo, r.hi) == (0.0, 1.0)

    r = _row(df, "r20", 60, 2)
    assert r.p == 0.5 and (r.median_days, r.p90_days) == pytest.approx((32.5, 54.5))

    r = _row(df, "r40", 60, 1)
    assert r.p == 0.75 and (r.median_days, r.p90_days) == pytest.approx((25.0, 37.0))

    r = _row(df, "r20", 30, 2)
    assert r.p == 0.25 and r.median_days == r.p90_days == 5.0


def test_summarize_destination_filter_and_overall_cell():
    df = analyze.summarize(
        _results(), _labels(), rules=["r40"], supplies=[60], dest_groups=["Coast: Ecuador"], by=()
    )
    assert list(df.columns) == [
        "rule", "supply_days", "p", "lo", "hi", "n", "n_years", "n_events", "share_not_sailed",
        "median_days", "p90_days",
    ]
    (r,) = df.itertuples()
    # Ecuador sightings within 60 days: voyages 1, 6, 7 (days 25, 55, 58).
    assert (r.p, r.n, r.n_years) == (3 / 8, 8, 2)
    assert r.median_days == 55.0
    # No ENSO cells: no event count.
    assert pd.isna(r.n_events) and r.share_not_sailed == 0.0


def test_summarize_default_month_and_enso_cells_keep_unlabelled_voyages():
    labels = _labels()
    labels["enso"] = ["el_nino", "neutral", None, "neutral", "la_nina", "la_nina", None, "neutral"]
    df = analyze.summarize(_results(), labels, rules=["r40"], supplies=[90])
    assert list(df.columns[2:4]) == ["month", "enso"]
    # Voyages without an ENSO label form their own cell instead of being dropped.
    assert df.n.sum() == 8 and df.enso.isna().sum() == 2
    cell = df[(df.month == 2) & (df.enso == "la_nina")].iloc[0]
    # Voyages 4 and 5 (both 2000): sightings on days 4 and 80, one year.
    assert (cell.p, cell.n, cell.n_years, cell.median_days) == (1.0, 2, 1, 42.0)
    # Categorical labels group the same way, in category order.
    labels["enso"] = pd.Categorical(
        labels.enso.fillna("unknown"), categories=["el_nino", "neutral", "la_nina", "unknown"]
    )
    cat = analyze.summarize(_results(), labels, rules=["r40"], supplies=[90])
    assert list(cat.enso.astype(str)) == ["el_nino", "neutral", "unknown", "neutral", "la_nina", "unknown"]
    np.testing.assert_array_equal(cat.n, [1, 2, 1, 1, 2, 1])


def test_summarize_is_deterministic_and_aligns_labels_by_voyage_id():
    kw = dict(rules=["r20", "r40"], supplies=[30, 60, 90], dest_groups=None, by=("month",))
    a = analyze.summarize(_results(), _labels(), **kw)
    pd.testing.assert_frame_equal(a, analyze.summarize(_results(), _labels(), **kw))
    shuffled = _labels().iloc[[5, 2, 7, 0, 3, 6, 1, 4]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, analyze.summarize(_results(), shuffled, **kw))
    with pytest.raises(ValueError):
        analyze.summarize(_results(), _labels().drop(columns="voyage_id").iloc[:7], **kw)


def test_share_not_sailed_counts_did_not_sail_as_failures():
    res = _results()
    # Voyages 1 and 2, both launched in January, never left port: each cell reports its own share.
    res.status = np.array([END_ALL_SIGHTED, DID_NOT_SAIL, DID_NOT_SAIL, AT_SEA, AT_SEA, AT_SEA, AT_SEA, AT_SEA])
    res.hit_days[:, [1, 2]] = NAN
    res.hit_landmass[:, [1, 2]] = -1
    df = analyze.summarize(res, _labels(), rules=["r20"], supplies=[90], by=("month",))
    assert list(df.share_not_sailed) == [0.5, 0.0]
    # They stay in the denominator: January has sightings by voyages 0 and 3 of 4.
    assert _row(df, "r20", 90, 1).p == 0.5


def test_summarize_without_voyages_gives_an_empty_overall_cell(recwarn):
    res = SimpleNamespace(voyage_id=np.zeros(0, int), launch_t=np.zeros(0), status=np.zeros(0, int),
                          rule_names=["r40"], landmass_groups=GROUPS, hit_days=np.zeros((1, 0)),
                          hit_landmass=np.zeros((1, 0), int))
    labels = analyze.launch_labels(res)
    (r,) = analyze.summarize(res, labels, ["r40"], [60], by=()).itertuples()
    assert (r.n, r.n_years) == (0, 0) and np.isnan(r.p) and np.isnan(r.share_not_sailed)
    assert analyze.summarize(res, labels, ["r40"], [60]).empty
    assert not recwarn.list


# --- launch labels and events ----------------------------------------------------

def _monthly_phases(start, codes):
    names = {"E": enso.EL_NINO, "L": enso.LA_NINA, "N": enso.NEUTRAL}
    return pd.Series([names[c] for c in codes], index=pd.period_range(start, periods=len(codes), freq="M"))


def test_launch_labels_from_launch_times():
    t = geo.to_epoch_hours(
        np.array(["1997-06-30T23:00", "1997-07-01", "1998-01-15", "1998-06-30", "2001-03-01"], dtype="datetime64[s]")
    )
    res = SimpleNamespace(voyage_id=np.arange(10, 15), launch_t=t)
    phases = _monthly_phases("1997-06", "NEEEEEEEEN")  # El Nino Jul 1997 - Feb 1998
    index = pd.Series([0.3, 0.7, 2.4], index=pd.period_range("1997-06", periods=3, freq="M"))
    lab = analyze.launch_labels(res, phases, index)
    assert list(lab.columns) == ["voyage_id", "month", "enso", "enso_source", "year", "index", "event"]
    assert list(lab.enso_source) == ["index", "index", "index", "unknown", "unknown"]
    assert list(lab.voyage_id) == [10, 11, 12, 13, 14]
    assert list(lab.month) == [6, 7, 1, 6, 3]
    assert list(lab.enso) == ["neutral", "el_nino", "el_nino", "unknown", "unknown"]
    assert list(lab.year) == [1996, 1997, 1997, 1997, 2000]
    np.testing.assert_allclose(lab["index"], [0.3, 0.7, NAN, NAN, NAN])
    assert list(lab.event.fillna("-")) == ["-", "el_nino:1997-07", "el_nino:1997-07", "-", "-"]
    # Labels from two indices are told apart by source.
    source = pd.Series(["hadisst"] * 4 + ["cpc"] * 6, index=phases.index)
    assert list(analyze.launch_labels(res, phases, index, source).enso_source) == [
        "hadisst", "hadisst", "cpc", "unknown", "unknown"]
    bare = analyze.launch_labels(res)
    assert (bare.enso == "unknown").all() and bare["index"].isna().all() and bare.event.isna().all()
    assert (bare.enso_source == "unknown").all()
    pd.testing.assert_series_equal(bare.year, lab.year)


def test_events_are_maximal_same_phase_runs_of_consecutive_months():
    phases = _monthly_phases("2000-01", "EEELLNEEE")
    # A missing month (dropped from the series) also ends a run.
    phases = pd.concat([phases, _monthly_phases("2000-11", "EE")])
    ev = analyze._events(phases)
    assert list(ev.index.astype(str)) == [
        "2000-01", "2000-02", "2000-03", "2000-04", "2000-05", "2000-07", "2000-08", "2000-09", "2000-11", "2000-12",
    ]
    assert list(ev) == ["el_nino:2000-01"] * 3 + ["la_nina:2000-04"] * 2 + ["el_nino:2000-07"] * 3 + [
        "el_nino:2000-11"
    ] * 2
    assert analyze._events(phases.iloc[:0]).empty


def test_summarize_counts_distinct_events_in_el_nino_and_la_nina_cells():
    # Eight voyages, one per month from Jan 2000: an El Nino (Jan-Mar), a La
    # Nina (Apr-May), neutral (Jun), then a second El Nino (Jul-Aug).
    t = geo.to_epoch_hours(pd.date_range("2000-01-01", periods=8, freq="MS").to_numpy() + np.timedelta64(9, "D"))
    res = _results()
    res.launch_t = t
    lab = analyze.launch_labels(res, _monthly_phases("2000-01", "EEELLNEE"))
    df = analyze.summarize(res, lab, rules=["r40"], supplies=[90], by=("enso",))
    counts = dict(zip(df.enso, df.n_events))
    assert counts["el_nino"] == 2 and counts["la_nina"] == 1 and pd.isna(counts["neutral"])
    assert df.n_events.dtype == "Int64"
    # Cells split by month see one event each; without enso among the cells, none are counted.
    by_month = analyze.summarize(res, lab, rules=["r40"], supplies=[90], by=("month", "enso"))
    assert list(by_month.n_events.fillna(-1)) == [1, 1, 1, 1, 1, -1, 1, 1]
    assert analyze.summarize(res, lab, rules=["r40"], supplies=[90], by=("month",)).n_events.isna().all()
    # Labels without an event column report no event counts.
    no_events = analyze.summarize(res, lab.drop(columns="event"), rules=["r40"], supplies=[90], by=("enso",))
    assert no_events.n_events.isna().all()
