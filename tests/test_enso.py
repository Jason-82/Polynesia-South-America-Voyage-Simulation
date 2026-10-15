import numpy as np
import pandas as pd
import pytest

from voyage_sim import enso, geo

CPC_SNIPPET = """\
 SEAS  YR   TOTAL   ANOM
  DJF 1950  24.72  -1.53
  JFM 1950  25.17  -1.34
  SON 1950  25.53  -0.81
  OND 1950  25.48  -0.84
  NDJ 1950  25.53  -0.82
  DJF 1951  25.24  -0.82
  JFM 1951  25.71  -0.54
  FMA 1951  -99.9  -99.9
"""

PSL_SNIPPET = """\
 1950 1951
 1950   -1.53   -1.34   -1.16   -1.18   -1.07   -0.85   -0.54   -0.42   -0.39   -0.44   -0.60   -0.80
 1951   -0.82   -0.54   -0.17    0.18    0.36    0.58    0.70    0.89  -99.99  -99.99  -99.99  -99.99
  -99.99
  Nino 3.4 anomaly, 1991-2020 base period
  2024 values are preliminary
"""


def _months(start, values):
    idx = pd.period_range(start, periods=len(values), freq="M")
    return pd.Series(values, index=idx, dtype=float)


def test_cpc_seasons_filed_under_centre_month_across_year_boundary():
    oni = enso.parse_cpc_oni(CPC_SNIPPET)
    assert isinstance(oni.index, pd.PeriodIndex)
    # DJF 1950 is Dec 1949 - Feb 1950, centred on Jan 1950.
    assert oni[pd.Period("1950-01", "M")] == pytest.approx(-1.53)
    assert oni[pd.Period("1950-02", "M")] == pytest.approx(-1.34)
    # NDJ 1950 (Nov 1950 - Jan 1951) is Dec 1950, and DJF 1951 follows it as Jan 1951.
    assert oni[pd.Period("1950-12", "M")] == pytest.approx(-0.82)
    assert oni[pd.Period("1951-01", "M")] == pytest.approx(-0.82)
    assert oni[pd.Period("1950-10", "M")] == pytest.approx(-0.81)
    assert np.isnan(oni[pd.Period("1951-03", "M")])
    assert len(oni) == 8 and oni.index.is_monotonic_increasing


def test_psl_parses_rows_missing_flag_and_ignores_footer():
    s = enso.parse_psl_monthly(PSL_SNIPPET)
    assert len(s) == 24
    assert s.index[0] == pd.Period("1950-01", "M") and s.index[-1] == pd.Period("1951-12", "M")
    assert s[pd.Period("1950-12", "M")] == pytest.approx(-0.80)
    assert s[pd.Period("1951-08", "M")] == pytest.approx(0.89)
    assert s[pd.Period("1951-09", "M"):].isna().all()
    assert s[: pd.Period("1951-08", "M")].notna().all()


def test_psl_rejects_a_missing_year_row():
    broken = PSL_SNIPPET.replace(" 1951   -0.82", " 1952   -0.82")
    with pytest.raises(ValueError):
        enso.parse_psl_monthly(broken)
    # A truncated download (rows, flag and footer cut off) is a ValueError too.
    with pytest.raises(ValueError):
        enso.parse_psl_monthly("\n".join(PSL_SNIPPET.splitlines()[:2]))


def test_running_mean_is_centred_and_needs_all_three_months():
    s = _months("2000-01", [1, 2, 3, 4, 6, 8])
    centred = enso.running_mean_3(s)
    np.testing.assert_allclose(centred.to_numpy(), [np.nan, 2, 3, 13 / 3, 6, np.nan])
    # A missing month (absent from the index, or NaN) blanks every window that spans it.
    gappy = s.drop(pd.Period("2000-03", "M"))
    m = enso.running_mean_3(gappy)
    assert len(m) == 6
    assert m.iloc[1:4].isna().all() and m.iloc[4] == pytest.approx(6.0)
    # Timestamp-indexed input is read by calendar month.
    ts = s.copy()
    ts.index = pd.date_range("2000-01-01", periods=6, freq="MS") + pd.Timedelta(days=14)
    pd.testing.assert_series_equal(enso.running_mean_3(ts), centred)


def test_noaa_rule_needs_five_consecutive_months_at_or_beyond_threshold():
    values = (
        [0.5, 0.9, 1.0, 0.6]          # 4 warm months: neutral
        + [0.0]
        + [0.5, 0.5, 0.9, 1.2, 0.5]   # 5 warm months, two exactly at +0.5: El Nino
        + [0.49]
        + [0.49, 0.8, 0.9, 1.0, 0.7]  # 0.49 does not count: only 4 warm months
        + [-0.5, -0.7, -1.0, -0.8, -0.6, -0.5]  # 6 cold months: La Nina
        + [-0.4]
    )
    phases = enso.enso_phases(_months("2000-01", values))
    expected = (
        ["neutral"] * 5 + ["el_nino"] * 5 + ["neutral"] * 6 + ["la_nina"] * 6 + ["neutral"]
    )
    assert list(phases) == expected


def test_missing_month_breaks_a_run_and_is_left_unlabelled():
    values = [0.8, 0.9, 1.0, np.nan, 1.1, 0.9, 0.7, 0.6, 0.5]
    phases = enso.enso_phases(_months("2000-01", values))
    assert pd.Period("2000-04", "M") not in phases.index
    assert list(phases) == ["neutral"] * 3 + ["el_nino"] * 5


def test_threshold_survives_float_rounding_of_three_month_means():
    # (0.12 + 0.95 + 0.43) / 3 is 0.49999999999999994 in floating point.
    oni = _months("2000-01", [(0.12 + 0.95 + 0.43) / 3] * 5)
    assert (enso.enso_phases(oni) == "el_nino").all()
    assert (enso.enso_phases(-oni) == "la_nina").all()
    assert (enso.enso_phases(oni, min_run=6) == "neutral").all()


def test_label_times_uses_utc_calendar_month():
    phases = pd.Series(
        ["neutral", "el_nino", "la_nina"], index=pd.period_range("2000-01", periods=3, freq="M")
    )
    t = geo.to_epoch_hours(
        np.array(
            ["2000-01-31T23:00", "2000-02-01T00:00", "2000-03-15T12:00", "1999-12-31T23:59", "2000-04-01T00:00"],
            dtype="datetime64[s]",
        )
    )
    t = np.append(t, np.nan)
    labels = enso.label_times(t, phases)
    assert list(labels) == ["neutral", "el_nino", "la_nina", "unknown", "unknown", "unknown"]
    # A phase table keyed by timestamps gives the same labels.
    stamped = phases.copy()
    stamped.index = pd.DatetimeIndex(["2000-01-01", "2000-02-01", "2000-03-01"])
    assert list(enso.label_times(t, stamped)) == list(labels)
    # Before 1970 epoch hours are negative; one hour before the epoch is December 1969.
    early = pd.Series(["la_nina", "el_nino"], index=pd.period_range("1969-12", periods=2, freq="M"))
    assert list(enso.label_times([-1.0, 0.0, -744.0 - 1e-3], early)) == ["la_nina", "el_nino", "unknown"]


def test_cpc_to_launch_labels_end_to_end():
    text = " SEAS  YR   TOTAL   ANOM\n" + "".join(
        f"  {s} 1997  27.0  {a:.2f}\n"
        for s, a in zip(
            ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"],
            [-0.5, -0.4, -0.1, 0.3, 0.8, 1.2, 1.6, 1.9, 2.1, 2.3, 2.4, 2.2],
        )
    )
    phases = enso.enso_phases(enso.parse_cpc_oni(text))
    t = geo.to_epoch_hours(np.array(["1997-03-10", "1997-05-01", "1997-12-20"], dtype="datetime64[s]"))
    assert list(enso.label_times(t, phases)) == ["neutral", "el_nino", "el_nino"]
