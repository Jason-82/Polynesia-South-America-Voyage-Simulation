from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from voyage_sim import analyze

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
        "rule", "supply_days", "month", "p", "lo", "hi", "n", "n_years", "median_days", "p90_days",
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
        "rule", "supply_days", "p", "lo", "hi", "n", "n_years", "median_days", "p90_days",
    ]
    (r,) = df.itertuples()
    # Ecuador sightings within 60 days: voyages 1, 6, 7 (days 25, 55, 58).
    assert (r.p, r.n, r.n_years) == (3 / 8, 8, 2)
    assert r.median_days == 55.0


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
