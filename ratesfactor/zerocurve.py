import math
from pathlib import Path

import numpy as np
import pandas as pd

from .pricing import year_fraction


CURVE_UNIVERSE_COLUMNS = [
    "instrument_id",
    "instrument_type",
    "cusip",
    "settlement_date",
    "maturity_date",
    "coupon_rate",
    "coupon_frequency",
    "day_count",
    "face_value",
    "clean_price",
    "accrued_interest",
    "dirty_price",
    "quote_date",
    "notes",
]


def demo_curve_universe():
    return pd.DataFrame({
        "instrument_id": [
            "UST-BILL-1M",
            "UST-BILL-3M",
            "UST-BILL-6M",
            "UST-NOTE-1Y",
            "UST-NOTE-2Y",
            "UST-NOTE-3Y",
            "UST-NOTE-5Y",
            "UST-NOTE-7Y",
            "UST-NOTE-10Y",
            "UST-BOND-20Y",
            "UST-BOND-30Y",
        ],
        "instrument_type": [
            "bill",
            "bill",
            "bill",
            "note",
            "note",
            "note",
            "note",
            "note",
            "note",
            "bond",
            "bond",
        ],
        "cusip": [
            "912797AA1",
            "912797AB9",
            "912797AC7",
            "91282CAA1",
            "91282CAB9",
            "91282CAC7",
            "91282CAD5",
            "91282CAE3",
            "91282CAF0",
            "912810AA1",
            "912810AB9",
        ],
        "settlement_date": ["2026-07-12"] * 11,
        "maturity_date": [
            "2026-08-12",
            "2026-10-12",
            "2027-01-12",
            "2027-07-12",
            "2028-07-12",
            "2029-07-12",
            "2031-07-12",
            "2033-07-12",
            "2036-07-12",
            "2046-07-12",
            "2056-07-12",
        ],
        "coupon_rate": [0.0, 0.0, 0.0, 0.039, 0.04, 0.041, 0.0425, 0.044, 0.045, 0.047, 0.0475],
        "coupon_frequency": [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2],
        "day_count": ["ACT/360", "ACT/360", "ACT/360"] + ["ACT/ACT"] * 8,
        "face_value": [100.0] * 11,
        "clean_price": [99.63, 98.91, 97.86, 99.78, 99.74, 99.63, 99.35, 99.06, 98.72, 97.95, 97.41],
        "accrued_interest": [0.0] * 11,
        "dirty_price": [99.63, 98.91, 97.86, 99.78, 99.74, 99.63, 99.35, 99.06, 98.72, 97.95, 97.41],
        "quote_date": ["2026-07-12"] * 11,
        "notes": [
            "Short bill proxy; zero coupon.",
            "Bill used for front-end discount factor.",
            "Bill used for 6M discount factor.",
            "Short coupon note.",
            "On-the-run style 2Y note proxy.",
            "3Y note proxy.",
            "5Y note proxy.",
            "7Y note proxy.",
            "10Y note proxy.",
            "20Y bond proxy.",
            "30Y bond proxy.",
        ],
    })


def curve_universe_template():
    return demo_curve_universe().loc[:, CURVE_UNIVERSE_COLUMNS].copy()


def load_curve_universe(path_or_buffer):
    raw = pd.read_excel(path_or_buffer, sheet_name="Curve Universe")
    return clean_curve_universe(raw)


def clean_curve_universe(raw):
    universe = raw.copy()
    universe = universe.dropna(how="all")
    universe.columns = [str(col).strip() for col in universe.columns]

    required = [
        "instrument_id",
        "instrument_type",
        "settlement_date",
        "maturity_date",
        "coupon_rate",
        "coupon_frequency",
        "day_count",
        "face_value",
        "clean_price",
        "accrued_interest",
    ]
    missing = [col for col in required if col not in universe.columns]
    if missing:
        raise ValueError(f"Curve universe is missing required columns: {missing}")

    if "dirty_price" not in universe.columns:
        universe["dirty_price"] = np.nan
    if "quote_date" not in universe.columns:
        universe["quote_date"] = pd.NaT
    if "cusip" not in universe.columns:
        universe["cusip"] = ""
    if "notes" not in universe.columns:
        universe["notes"] = ""

    universe["settlement_date"] = pd.to_datetime(universe["settlement_date"], errors="coerce")
    universe["maturity_date"] = pd.to_datetime(universe["maturity_date"], errors="coerce")
    universe["quote_date"] = pd.to_datetime(universe["quote_date"], errors="coerce")

    numeric_cols = ["coupon_rate", "coupon_frequency", "face_value", "clean_price", "accrued_interest", "dirty_price"]
    for col in numeric_cols:
        universe[col] = pd.to_numeric(universe[col], errors="coerce")

    universe["dirty_price"] = universe["dirty_price"].fillna(
        universe["clean_price"] + universe["accrued_interest"]
    )
    universe["day_count"] = universe["day_count"].fillna("ACT/ACT")
    universe["instrument_type"] = universe["instrument_type"].str.lower().str.strip()

    universe = universe.dropna(
        subset=[
            "settlement_date",
            "maturity_date",
            "coupon_rate",
            "coupon_frequency",
            "face_value",
            "dirty_price",
        ]
    )
    universe = universe[universe["maturity_date"] > universe["settlement_date"]].copy()
    universe["time_to_maturity"] = [
        year_fraction(start, end, day_count)
        for start, end, day_count in zip(
            universe["settlement_date"],
            universe["maturity_date"],
            universe["day_count"],
        )
    ]
    universe = universe[universe["time_to_maturity"] > 0].copy()
    universe = universe.sort_values("time_to_maturity").reset_index(drop=True)

    if universe.empty:
        raise ValueError("No valid rows found in curve construction universe.")

    return universe.loc[:, CURVE_UNIVERSE_COLUMNS + ["time_to_maturity"]]


def log_interpolate(t, d1, d2, t1, t2):
    log_d1 = math.log(d1)
    log_d2 = math.log(d2)
    log_d = log_d1 + ((log_d2 - log_d1) / (t2 - t1)) * (t - t1)
    return math.exp(log_d)


def discount_factor_at(t, discount_factors):
    t_vals = sorted(discount_factors.keys())
    if not t_vals:
        raise ValueError("No discount factors are available yet.")

    for t_known in t_vals:
        if np.isclose(t, t_known):
            return discount_factors[t_known]

    if t < t_vals[0]:
        return discount_factors[t_vals[0]]
    if t > t_vals[-1]:
        return discount_factors[t_vals[-1]]

    for idx in range(len(t_vals) - 1):
        t1 = t_vals[idx]
        t2 = t_vals[idx + 1]
        if t1 < t < t2:
            return log_interpolate(t, discount_factors[t1], discount_factors[t2], t1, t2)

    raise ValueError(f"Could not interpolate discount factor for t={t}")


def bootstrap_zero_curve(curve_universe):
    universe = clean_curve_universe(curve_universe)
    discount_factors = {}
    rows = []

    for _, row in universe.iterrows():
        coupon = float(row["coupon_rate"])
        maturity = float(row["time_to_maturity"])
        settlement_date = row["settlement_date"]
        maturity_date = row["maturity_date"]
        dirty_price = float(row["dirty_price"])
        face_value = float(row["face_value"])
        frequency = int(row["coupon_frequency"])
        day_count = row["day_count"]

        if frequency == 0 or coupon == 0:
            df = dirty_price / face_value
        else:
            if 12 % frequency != 0:
                raise ValueError(f"coupon_frequency must divide 12 for {row['instrument_id']}")

            months_step = 12 // frequency
            known_coupon_pv = 0.0
            final_cashflow = face_value * (1 + coupon / frequency)
            cashflow_date = maturity_date

            while cashflow_date > settlement_date:
                if cashflow_date != maturity_date:
                    t = year_fraction(settlement_date, cashflow_date, day_count)
                    coupon_cashflow = face_value * coupon / frequency
                    known_coupon_pv += coupon_cashflow * discount_factor_at(t, discount_factors)

                cashflow_date -= pd.DateOffset(months=months_step)

            df = (dirty_price - known_coupon_pv) / final_cashflow

        if df <= 0:
            raise ValueError(f"Bootstrapped non-positive discount factor for {row['instrument_id']}")

        discount_factors[maturity] = df
        zero_rate_cc = -math.log(df) / maturity
        rows.append({
            "instrument_id": row["instrument_id"],
            "instrument_type": row["instrument_type"],
            "maturity_date": row["maturity_date"],
            "time_to_maturity": maturity,
            "discount_factor": df,
            "zero_rate_cc": zero_rate_cc,
            "dirty_price": dirty_price,
        })

    zero_curve = pd.DataFrame(rows).sort_values("time_to_maturity").reset_index(drop=True)
    return zero_curve


def zero_curve_to_rate_curve(zero_curve, tenors):
    tenors = np.asarray(tenors, dtype=float)
    times = zero_curve["time_to_maturity"].to_numpy(dtype=float)
    log_dfs = np.log(zero_curve["discount_factor"].to_numpy(dtype=float))
    interp_log_dfs = np.interp(tenors, times, log_dfs)
    zero_rates = -interp_log_dfs / tenors
    return pd.Series(zero_rates, index=tenors)


def bundled_curve_universe_template_path():
    return Path(__file__).resolve().parents[1] / "templates" / "curve_construction_universe_template.xlsx"
