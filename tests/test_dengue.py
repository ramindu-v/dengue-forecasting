"""
Tests for the dengue forecasting pipeline.   Run:  pytest
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dengue as D  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return D.load_cases()


@pytest.fixture(scope="module")
def base(panel):
    return D.base_features(panel)


def test_panel_is_complete(panel):
    weeks = panel.week_start.nunique()
    assert panel.district.nunique() == 26
    assert len(panel) == weeks * 26                       # one row per district per week
    assert panel.cases.isna().sum() == 0
    assert (panel.cases >= 0).all()
    gaps = panel.week_start.drop_duplicates().sort_values().diff().dropna()
    assert (gaps == pd.Timedelta(weeks=1)).all()          # regular weekly grid


def test_lag_features_use_only_the_past(base):
    col = base[base.district == "Colombo"].reset_index(drop=True)
    assert col.loc[10, "y_lag1"] == pytest.approx(col.loc[9, "y"])
    assert col.loc[10, "y_lag0"] == pytest.approx(col.loc[10, "y"])
    assert col.loc[10, "y_mean4"] == pytest.approx(col.loc[7:10, "y"].mean())


@pytest.mark.parametrize("h", D.HORIZONS)
def test_target_is_h_weeks_ahead(base, h):
    df = D.horizon_features(base, h)
    col = df[df.district == "Kandy"].reset_index(drop=True)
    assert col.loc[100, "target_cases"] == col.loc[100 + h, "cases"]
    assert col.loc[100, "target_week"] == col.loc[100 + h, "week_start"]
    # same week last year, relative to the TARGET week
    assert col.loc[100, "y_last_year"] == pytest.approx(col.loc[100 + h - 52, "y"])


@pytest.mark.parametrize("h", D.HORIZONS)
def test_no_future_information_in_features(base, h):
    """Changing future cases must not change any feature at the forecast origin."""
    df1 = D.horizon_features(base, h)
    origin = pd.Timestamp("2020-06-06")
    future = base.copy()
    future.loc[future.week_start > origin, ["cases", "y"]] = 0
    df2 = D.horizon_features(D.base_features(future[["district", "week_start", "cases"]]), h)
    feats = D.feature_columns(df1)
    a = df1[df1.week_start == origin][feats].reset_index(drop=True)
    b = df2[df2.week_start == origin][feats].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_endemic_channel_matches_single_week(panel):
    week = pd.Timestamp("2024-06-29")
    table = D.channel_table(panel)
    a = table[table.week_start == week].set_index("district")[["typical", "upper"]].sort_index()
    b = D.channel_for_week(panel, week).set_index("district").sort_index()
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_names=False)
    assert (b.upper >= b.typical).all()


def test_alert_levels():
    assert D.alert_level(10, 10, upper=20) == "Normal"
    assert D.alert_level(18, 25, upper=20) == "Elevated"
    assert D.alert_level(60, 70, upper=20) == "High"
    assert D.alert_level(5, 5, upper=np.nan) == "Unknown"
    assert D.alert_value(10, 30, 0) == pytest.approx(10)
    assert D.alert_value(10, 30, 1) == pytest.approx(30)


def test_model_beats_naive_baseline(base):
    """Short backtest on one year: the model must beat 'same as this week'."""
    res = D.backtest(base, test_years=(2024,), use_weather=False)
    mae = D.score(res)
    assert (mae["point"] < mae["naive"]).all()
    assert (res["low"] <= res["point"]).all() and (res["point"] <= res["high"]).all()


def test_outputs_exist_and_are_consistent():
    out = ROOT / "outputs"
    fc = pd.read_csv(out / "forecast_latest.csv")
    assert set(fc.horizon) == set(D.HORIZONS)
    assert fc.groupby("horizon").district.nunique().eq(26).all()
    assert fc.alert.isin(["High", "Elevated", "Normal", "Unknown"]).all()
