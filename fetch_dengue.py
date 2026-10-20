"""
fetch_dengue.py
Downloads the latest weekly dengue cases for Sri Lanka's districts and saves a clean CSV.

    python fetch_dengue.py

Source: the `denguedatahub` R package by Dr. Thiyanga S. Talagala (University of Sri Jayewardenepura),
which collects the Weekly Epidemiological Reports of the Epidemiology Unit, Ministry of Health,
Sri Lanka. https://github.com/thiyangt/denguedatahub

Output: data/dengue_weekly.csv  (week_start, report_year, report_week, district, cases)
"""

import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

URL = "https://raw.githubusercontent.com/thiyangt/denguedatahub/main/data/srilanka_weekly_data.rda"
OUT = Path(__file__).parent / "data" / "dengue_weekly.csv"


def align_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    A few reports start a day or two off the usual weekly grid. Snap every report to the
    nearest date on a regular 7-day grid so each district has exactly one row per week.
    """
    start = pd.to_datetime(df["start.date"], format="%m/%d/%Y")
    grid = pd.date_range(start.min(), start.max() + pd.Timedelta(days=7), freq="7D")
    nearest = np.abs(start.values[:, None] - grid.values[None, :]).argmin(axis=1)
    df = df.assign(week_start=grid[nearest])
    shift = (start - df["week_start"]).abs().dt.days.max()
    assert shift <= 3, f"A report is {shift} days off the weekly grid"
    assert not df.duplicated(["week_start", "district"]).any(), "Two reports map to the same week"
    return df


def main():
    print("Downloading", URL)
    with urllib.request.urlopen(URL, timeout=120) as r:
        raw = r.read()
    tmp = OUT.parent / "_srilanka_weekly_data.rda"
    tmp.write_bytes(raw)
    df = next(iter(pyreadr.read_r(str(tmp)).values()))
    tmp.unlink()

    df = align_weeks(df)
    out = (df[["week_start", "year", "week", "district", "cases"]]
           .rename(columns={"year": "report_year", "week": "report_week"})
           .astype({"report_year": int, "report_week": int, "cases": int})
           .sort_values(["week_start", "district"]))
    out.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    print(f"Saved {len(out):,} rows: {out.week_start.min().date()} -> {out.week_start.max().date()}, "
          f"{out.district.nunique()} districts")


if __name__ == "__main__":
    main()
