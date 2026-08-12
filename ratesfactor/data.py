import numpy as np
import pandas as pd
import requests


def fetch_fred_rates(series_id, api_key, start_date, end_date):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "observations" not in payload:
        raise ValueError(f"FRED response missing observations for {series_id}: {payload}")

    df = pd.DataFrame.from_dict(payload["observations"])
    df = df.drop(columns=["realtime_start", "realtime_end"], errors="ignore")
    df = df[df["value"] != "."].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def fetch_treasury_rates(series, api_key, start_date, end_date):
    rates = pd.DataFrame()

    for tenor, series_id in series:
        curve_point = fetch_fred_rates(series_id, api_key, start_date, end_date)
        curve_point = curve_point.rename(columns={"value": tenor})
        rates = pd.concat([rates, curve_point[[tenor]]], axis=1)

    rates = rates.sort_index().dropna()
    return rates


class RatesData:
    def __init__(self, rates_pct):
        self.rates_pct = rates_pct.copy()
        self.rates_pct.index = pd.to_datetime(self.rates_pct.index).normalize()
        self.rates_decimal = self.rates_pct / 100

        self.dates = np.array(self.rates_pct.index)
        self.tenors = np.array(self.rates_pct.columns, dtype=float)
        self.latest_date = pd.Timestamp(self.dates[-1])
        self.daily_changes_bp = self.rates_pct.diff().dropna() * 100

    def get_curve(self, date=None, units="pct"):
        if date is None:
            date = self.latest_date

        date = pd.Timestamp(date).normalize()
        if date not in self.rates_pct.index:
            raise ValueError(f"Invalid date: {date}")

        if units == "decimal":
            return self.rates_decimal.loc[date]
        if units == "pct":
            return self.rates_pct.loc[date]
        raise ValueError("units must be 'pct' or 'decimal'")

