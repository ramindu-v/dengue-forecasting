"""
fetch_weather.py
Downloads daily weather for every district from NASA POWER (free, no API key).

    python fetch_weather.py

Output: data/weather_daily.csv  (date, district, rain_mm, temp_c, humidity_pct)

Variables
  PRECTOTCORR  precipitation (mm/day)
  T2M          mean air temperature at 2 m (°C)
  RH2M         relative humidity at 2 m (%)

Uses only the Python standard library, so it runs anywhere.
"""

import csv
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
COORDS = ROOT / "data" / "district_coords.csv"
OUT = ROOT / "data" / "weather_daily.csv"

START, END = "20061201", "20260531"
PARAMS = {"PRECTOTCORR": "rain_mm", "T2M": "temp_c", "RH2M": "humidity_pct"}
URL = ("https://power.larc.nasa.gov/api/temporal/daily/point"
       "?parameters={params}&community=AG&latitude={lat}&longitude={lon}"
       "&start={start}&end={end}&format=JSON")


def fetch(lat, lon, retries=3):
    url = URL.format(params=",".join(PARAMS), lat=lat, lon=lon, start=START, end=END)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)["properties"]["parameter"]
        except Exception as e:  # network hiccup -> wait and retry
            print(f"    attempt {attempt} failed: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download weather for {lat}, {lon}")


def main():
    with open(COORDS, newline="", encoding="utf-8") as f:
        districts = list(csv.DictReader(f))

    rows = []
    for i, d in enumerate(districts, 1):
        print(f"[{i:>2}/{len(districts)}] {d['district']}")
        data = fetch(d["lat"], d["lon"])
        for day in data["PRECTOTCORR"]:
            values = {}
            for api_name, col in PARAMS.items():
                v = data[api_name].get(day)
                values[col] = "" if v is None or v <= -999 else v   # -999 = missing
            rows.append({"date": f"{day[:4]}-{day[4:6]}-{day[6:]}", "district": d["district"], **values})
        time.sleep(1)  # be polite to the free API

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "district", *PARAMS.values()])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows):,} rows to {OUT}")


if __name__ == "__main__":
    main()
