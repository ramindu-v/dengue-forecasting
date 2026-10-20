"""
train.py
Runs the whole modelling pipeline and saves everything the app needs.

    python train.py            # with weather (if data/weather_daily.csv exists)
    python train.py --fast     # skip the slow weather-vs-no-weather comparison

Outputs (folder `outputs/`)
  backtest.csv           every backtest forecast (2022-2025) with actuals and baselines
  metrics.json           headline numbers: MAE vs baselines, interval coverage, alert accuracy
  forecast_latest.csv    1-4 week forecasts for every district from the latest data, with alerts
  feature_importance.csv which inputs the models rely on
  history.csv            weekly cases + endemic channel, for the app's charts
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import dengue as D

OUT = Path(__file__).parent / "outputs"
MODELS = Path(__file__).parent / "models"
TEST_YEARS = (2022, 2023, 2024, 2025)
VALIDATION_YEARS = (2019, 2020, 2021)   # used only to choose the alert threshold
ALERT_GRID = np.round(np.arange(0, 1.01, 0.1), 1)


def alert_scores(res: pd.DataFrame, channel: pd.DataFrame, col: str) -> dict:
    """How well does a forecast column predict that a district-week will be above its upper line?"""
    m = res.merge(channel, left_on=["district", "target_week"], right_on=["district", "week_start"],
                  suffixes=("", "_ch")).dropna(subset=["upper"])
    actual = m["target_cases"] > m["upper"]
    pred = m[col] > m["upper"]
    tp, fp, fn = (actual & pred).sum(), (~actual & pred).sum(), (actual & ~pred).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(float(precision), 3), "recall": round(float(recall), 3),
            "f1": round(float(f1), 3), "actual_alert_rate": round(float(actual.mean()), 3)}


def main(fast: bool = False):
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    panel = D.load_cases()
    weather = D.load_weather(weeks=pd.DatetimeIndex(panel.week_start.unique()).sort_values())
    has_weather = weather is not None
    print(f"Cases: {panel.week_start.min().date()} -> {panel.week_start.max().date()}, "
          f"{panel.district.nunique()} districts | weather: {'yes' if has_weather else 'NO'}")

    base = D.base_features(panel, weather)
    channel = D.channel_table(panel)

    # ---- 1. Backtest ------------------------------------------------------
    print("Backtesting (expanding window, test years 2022-2025)...")
    res = D.backtest(base, TEST_YEARS, use_weather=has_weather)
    res.to_csv(OUT / "backtest.csv", index=False)

    # ---- Choose the alert threshold on VALIDATION years (no peeking at the test years) ----
    print("Choosing the alert threshold on validation years 2019-2021...")
    val = D.backtest(base, VALIDATION_YEARS, use_weather=has_weather)
    alert_a = {}
    for h in D.HORIZONS:
        v = val[val.horizon == h].copy()
        f1 = {}
        for a in ALERT_GRID:
            v["alert_val"] = D.alert_value(v["point"], v["high"], a)
            f1[a] = alert_scores(v, channel, "alert_val")["f1"]
        alert_a[h] = float(max(f1, key=f1.get))
    res["alert_val"] = D.alert_value(res["point"], res["high"], res["horizon"].map(alert_a))
    res.to_csv(OUT / "backtest.csv", index=False)

    mae = D.score(res)
    metrics = {
        "data": {"first_week": str(panel.week_start.min().date()),
                 "last_week": str(panel.week_start.max().date()),
                 "districts": int(panel.district.nunique()),
                 "rows": int(len(panel)), "weather": has_weather},
        "test_years": list(TEST_YEARS),
        "mae_by_horizon": mae.round(2).to_dict(orient="index"),
        "improvement_vs_naive_pct": ((1 - mae["point"] / mae["naive"]) * 100).round(1).to_dict(),
        "mae_by_year": D.score(res, by="test_year").round(2).to_dict(orient="index"),
        "interval_coverage_80": round(D.interval_coverage(res), 3),
        "alert_threshold_a": alert_a,
        "alerts": {f"h{h}": {"model": alert_scores(res[res.horizon == h], channel, "alert_val"),
                             "model_central_forecast": alert_scores(res[res.horizon == h], channel, "point"),
                             "naive": alert_scores(res[res.horizon == h], channel, "naive")}
                   for h in D.HORIZONS},
    }

    # ---- 2. Does weather help? (same backtest without weather features) ---
    if has_weather and not fast:
        print("Backtesting again WITHOUT weather for comparison...")
        res_nw = D.backtest(base, TEST_YEARS, use_weather=False)
        metrics["mae_without_weather"] = D.score(res_nw, cols=("point",))["point"].round(2).to_dict()

    # ---- 3. Final models on all data + latest forecast --------------------
    print("Training final models on all data...")
    latest_week = panel.week_start.max()
    forecasts, importances = [], []
    for h in D.HORIZONS:
        df = D.horizon_features(base, h)
        feats = D.feature_columns(df)
        models = D.fit_models(df, feats)
        joblib.dump({"models": models, "features": feats}, MODELS / f"model_h{h}.pkl")

        now = df[df.week_start == latest_week]
        pred = D.predict(models, now, feats)
        f = pd.concat([now[["district", "week_start", "target_week", "cases"]], pred], axis=1)
        f["horizon"] = h
        forecasts.append(f)
        importances.append(pd.DataFrame({"feature": feats, "horizon": h,
                                         "gain": models["point"].booster_.feature_importance("gain")}))

    fc = pd.concat(forecasts, ignore_index=True)
    ch = pd.concat([D.channel_for_week(panel, w).assign(target_week=w) for w in fc.target_week.unique()])
    fc = fc.merge(ch, on=["district", "target_week"], how="left")
    fc["alert_val"] = D.alert_value(fc["point"], fc["high"], fc["horizon"].map(alert_a))
    fc["alert"] = [D.alert_level(p, a, u) for p, a, u in zip(fc.point, fc.alert_val, fc.upper)]
    fc = fc.rename(columns={"week_start": "origin_week", "cases": "cases_latest"})
    fc.round(1).to_csv(OUT / "forecast_latest.csv", index=False)

    imp = pd.concat(importances)
    imp["share"] = imp.gain / imp.groupby("horizon").gain.transform("sum")
    imp.to_csv(OUT / "feature_importance.csv", index=False)

    hist = panel.merge(channel, on=["district", "week_start"], how="left")
    if has_weather:
        hist = hist.merge(weather, on=["district", "week_start"], how="left")
    hist.round(2).to_csv(OUT / "history.csv", index=False)

    metrics["latest_forecast"] = {
        "origin_week": str(latest_week.date()),
        "alerts_h4": fc[fc.horizon == 4].alert.value_counts().to_dict(),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- Summary ----------------------------------------------------------
    print("\nMAE (cases per district per week) by horizon:")
    print(mae.round(2).to_string())
    if "mae_without_weather" in metrics:
        print("Without weather:", metrics["mae_without_weather"])
    print("80% interval coverage:", metrics["interval_coverage_80"])
    for h in D.HORIZONS:
        a = metrics["alerts"][f"h{h}"]
        print(f"Alert F1 h={h}: model {a['model']['f1']} (recall {a['model']['recall']}) "
              f"vs naive {a['naive']['f1']} (recall {a['naive']['recall']}), threshold a={alert_a[h]}")
    print("Saved outputs/ and models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    main(**vars(ap.parse_args()))
