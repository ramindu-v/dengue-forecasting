"""
dengue.py
Core of the project: data loading, feature engineering, models, backtesting and alerts.

Imported by train.py, the Streamlit app, the notebook and the tests, so every part of the
project uses exactly the same logic.

Forecasting set-up
  - We stand at week t (the latest week with reported cases) and forecast week t + h,
    for h = 1, 2, 3, 4 weeks ahead ("direct" strategy: one model per horizon).
  - Every feature uses only information available at week t (no future weather, no future cases),
    so the backtest is an honest simulation of real use.
  - One "global" LightGBM model per horizon is trained on all 26 districts together, so small
    districts borrow strength from big ones.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"

HORIZONS = [1, 2, 3, 4]
QUANTILES = {"low": 0.1, "high": 0.9}       # 80% prediction interval
SEED = 42

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, verbose=-1,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cases(path: Path = DATA / "dengue_weekly.csv") -> pd.DataFrame:
    """Weekly cases as a complete district x week panel (long format)."""
    df = pd.read_csv(path, parse_dates=["week_start"])
    wide = df.pivot(index="week_start", columns="district", values="cases")
    full_index = pd.date_range(wide.index.min(), wide.index.max(), freq="7D")
    wide = wide.reindex(full_index)
    # A handful of single missing reports: fill by linear interpolation (max 2 weeks)
    wide = wide.interpolate(limit=2, limit_area="inside").round()
    wide.index.name = "week_start"
    return (wide.stack().rename("cases").reset_index()
            .sort_values(["district", "week_start"]).reset_index(drop=True))


def load_weather(path: Path = DATA / "weather_daily.csv", weeks: pd.DatetimeIndex | None = None):
    """Daily NASA POWER weather -> weekly totals/means aligned to the case weeks. None if missing."""
    if not path.exists():
        return None
    w = pd.read_csv(path, parse_dates=["date"])
    if weeks is None:
        weeks = pd.date_range("2006-12-23", w.date.max(), freq="7D")
    # Assign each day to the case week that contains it: [week_start, week_start + 6 days]
    pos = np.searchsorted(weeks.values, w.date.values, side="right") - 1
    w = w[pos >= 0].assign(week_start=weeks[pos[pos >= 0]])
    weekly = (w.groupby(["district", "week_start"])
               .agg(rain_mm=("rain_mm", "sum"), temp_c=("temp_c", "mean"),
                    humidity_pct=("humidity_pct", "mean"), days=("date", "size"))
               .reset_index())
    return weekly[weekly.days == 7].drop(columns="days")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def _per_district(s: pd.Series, d: pd.Series, fn):
    return s.groupby(d, observed=True).transform(fn)


def base_features(panel: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features known at week t (shared by all horizons)."""
    df = panel.copy()
    if weather is not None:
        df = df.merge(weather, on=["district", "week_start"], how="left")

    d = df["district"]
    df["y"] = np.log1p(df["cases"])
    g = df.groupby(d, observed=True)["y"]

    for lag in range(8):                                  # last 8 weeks (lag 0 = this week)
        df[f"y_lag{lag}"] = g.shift(lag)
    for win in (4, 8, 13, 26):
        df[f"y_mean{win}"] = g.transform(lambda s: s.rolling(win).mean())
    df["y_std8"] = g.transform(lambda s: s.rolling(8).std())
    df["y_trend4"] = df["y_lag0"] - df["y_lag4"]          # growing or shrinking?
    df["y_hist_mean"] = g.transform(lambda s: s.expanding(26).mean())  # district's usual level

    national = df.groupby("week_start")["cases"].transform("sum")
    df["nat_lag0"] = np.log1p(national)
    nat_series = df.drop_duplicates("week_start").set_index("week_start")["nat_lag0"]
    df["nat_trend4"] = df["week_start"].map(nat_series - nat_series.shift(4))

    if weather is not None:
        r = df.groupby(d, observed=True)
        # Mosquitoes need 2-8 weeks after rain (breeding + virus incubation), so use lagged windows
        df["rain_0_3"] = r["rain_mm"].transform(lambda s: s.rolling(4).sum())
        df["rain_4_7"] = r["rain_mm"].transform(lambda s: s.shift(4).rolling(4).sum())
        df["rain_8_11"] = r["rain_mm"].transform(lambda s: s.shift(8).rolling(4).sum())
        df["temp_0_3"] = r["temp_c"].transform(lambda s: s.rolling(4).mean())
        df["temp_4_7"] = r["temp_c"].transform(lambda s: s.shift(4).rolling(4).mean())
        df["hum_0_3"] = r["humidity_pct"].transform(lambda s: s.rolling(4).mean())
        df["hum_4_7"] = r["humidity_pct"].transform(lambda s: s.shift(4).rolling(4).mean())

    df["district"] = df["district"].astype("category")
    return df


def horizon_features(base: pd.DataFrame, h: int) -> pd.DataFrame:
    """Add the horizon-specific columns: target week calendar, last year's value, and the target."""
    df = base.copy()
    g = df.groupby("district", observed=True)["y"]
    df["target_week"] = df["week_start"] + pd.Timedelta(weeks=h)
    woy = df["target_week"].dt.isocalendar().week.astype(float)
    df["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    df["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)
    # Same week last year (relative to the TARGET week) and its neighbours
    df["y_last_year"] = g.shift(52 - h)
    df["y_last_year_mean3"] = g.transform(lambda s: s.shift(52 - h - 1).rolling(3).mean())
    df["target"] = g.shift(-h)              # log1p(cases) at week t + h
    df["target_cases"] = df.groupby("district", observed=True)["cases"].shift(-h)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"week_start", "target_week", "cases", "y", "target", "target_cases",
            "rain_mm", "temp_c", "humidity_pct"}
    return [c for c in df.columns if c not in skip]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_models(train_df: pd.DataFrame, features: list[str]) -> dict:
    """Point model (L2) + two quantile models for the 80% interval, all on the log-change."""
    data = train_df.dropna(subset=["target", "y_lag7", "y_mean26"])
    # Predict the CHANGE from this week (on log scale), not the level: tree models cannot
    # predict values above anything seen in training, so levels fail during record epidemics.
    X, y = data[features], data["target"] - data["y_lag0"]
    models = {"point": lgb.LGBMRegressor(objective="regression", **LGB_PARAMS).fit(X, y)}
    for name, q in QUANTILES.items():
        models[name] = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGB_PARAMS).fit(X, y)
    return models


def predict(models: dict, df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, m in models.items():
        out[name] = np.clip(np.expm1(df["y_lag0"] + m.predict(df[features])), 0, None)
    # Quantile models are trained separately, so make sure low <= point <= high
    out["low"] = np.minimum(out["low"], out["point"])
    out["high"] = np.maximum(out["high"], out["point"])
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Simple forecasts any model must beat."""
    return pd.DataFrame({
        "naive": df["cases"],                                   # 'same as this week'
        "mean4": np.expm1(df["y_mean4"]),                       # average of last 4 weeks (geometric)
        "seasonal_naive": np.expm1(df["y_last_year"]),          # same week last year
    }, index=df.index)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest(base: pd.DataFrame, test_years=(2022, 2023, 2024, 2025), use_weather=True) -> pd.DataFrame:
    """
    Expanding-window backtest. For each test year Y and horizon h: train on every row whose
    TARGET week is before 1 Jan Y, then forecast every target week inside Y.
    Returns one row per (district, origin week, horizon) with actuals, model and baseline forecasts.
    """
    results = []
    for h in HORIZONS:
        df = horizon_features(base, h)
        feats = feature_columns(df)
        if not use_weather:
            feats = [f for f in feats if not f.startswith(("rain_", "temp_", "hum_"))]
        for year in test_years:
            start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
            train_df = df[df["target_week"] < start]
            test_df = df[(df["target_week"] >= start) & (df["target_week"] < end)].dropna(subset=["target"])
            models = fit_models(train_df, feats)
            res = pd.concat([test_df[["district", "week_start", "target_week", "target_cases"]],
                             predict(models, test_df, feats), baselines(test_df)], axis=1)
            res["horizon"], res["test_year"] = h, year
            results.append(res)
    return pd.concat(results, ignore_index=True)


def score(results: pd.DataFrame, cols=("point", "naive", "mean4", "seasonal_naive"), by="horizon"):
    """Mean absolute error (cases per district-week) for each forecast column."""
    err = {c: (results[c] - results["target_cases"]).abs() for c in cols}
    return pd.DataFrame(err).groupby(results[by]).mean()


def interval_coverage(results: pd.DataFrame) -> float:
    inside = (results["target_cases"] >= results["low"]) & (results["target_cases"] <= results["high"])
    return float(inside.mean())


# ---------------------------------------------------------------------------
# Alerts: the "endemic channel"
# ---------------------------------------------------------------------------

def channel_table(panel: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """
    The classic epidemiological "endemic channel" for every district and week.

    For week w, look at the same 5-week window (w-2 ... w+2) in each of the previous `years`
    years (25 values). The median is the TYPICAL level for that time of year and the 75th
    percentile is the UPPER line: cases above it signal an unusual rise.
    """
    wide = panel.pivot(index="week_start", columns="district", values="cases")
    arr = wide.to_numpy(dtype=float)
    n = len(arr)
    typical = np.full_like(arr, np.nan)
    upper = np.full_like(arr, np.nan)
    offsets = [52 * k + j for k in range(1, years + 1) for j in range(-2, 3)]
    for i in range(52 * years + 2, n + 52):           # also allows future weeks (i >= n)
        if i < n:
            past = np.stack([arr[i - o] for o in offsets])
            typical[i] = np.nanmedian(past, axis=0)
            upper[i] = np.nanpercentile(past, 75, axis=0)
    out = pd.DataFrame({
        "week_start": np.repeat(wide.index.values, wide.shape[1]),
        "district": np.tile(wide.columns.values, len(wide)),
        "typical": typical.ravel(), "upper": upper.ravel(),
    })
    return out


def channel_for_week(panel: pd.DataFrame, week: pd.Timestamp, years: int = 5) -> pd.DataFrame:
    """Endemic channel for any single week, including future weeks we are forecasting."""
    rows = []
    for k in range(1, years + 1):
        centre = week - pd.Timedelta(weeks=52 * k)
        lo, hi = centre - pd.Timedelta(weeks=2), centre + pd.Timedelta(weeks=2)
        rows.append(panel[(panel.week_start >= lo) & (panel.week_start <= hi)])
    hist = pd.concat(rows)
    return (hist.groupby("district", observed=True)["cases"]
                .agg(typical="median", upper=lambda s: s.quantile(0.75))
                .reset_index())


def alert_value(point, high, a: float):
    """
    The number compared with the upper line to raise an alert.

    a = 0 uses the central forecast; a = 1 uses the top of the 80% interval. Values in between
    trade a few more false alarms for catching more real rises (interpolated on the log scale).
    `a` is chosen on validation years in train.py, never on the test years.
    """
    lp, lh = np.log1p(point), np.log1p(high)
    return np.expm1(lp + a * (lh - lp))


def alert_level(point: float, alert_val: float, upper: float) -> str:
    """High = central forecast clearly above the usual level; Elevated = a rise is likely."""
    if np.isnan(upper):
        return "Unknown"
    if point > max(upper * 1.5, upper + 10):
        return "High"
    if alert_val > upper:
        return "Elevated"
    return "Normal"
