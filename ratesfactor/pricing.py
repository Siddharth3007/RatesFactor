import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline


def year_fraction(start_date, end_date, day_count="ACT/ACT"):
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    days = (end_date - start_date).days

    if day_count == "ACT/ACT":
        if days == 0:
            return 0.0

        total = 0.0
        current = start_date
        while current < end_date:
            next_year = pd.Timestamp(year=current.year + 1, month=1, day=1)
            period_end = min(end_date, next_year)
            days_in_year = 366 if current.is_leap_year else 365
            total += (period_end - current).days / days_in_year
            current = period_end
        return total
    if day_count == "ACT/365.25":
        return days / 365.25
    if day_count == "ACT/365":
        return days / 365
    if day_count == "ACT/360":
        return days / 360
    raise ValueError("Day count convention must be ACT/ACT, ACT/365.25, ACT/365, or ACT/360")


def _curve_interpolator(rate_curve):
    return CubicSpline(
        rate_curve.index.astype(float).to_numpy(),
        rate_curve.to_numpy(dtype=float),
    )


def _infer_maturity_date(row, settlement_date):
    if "maturity_date" in row.index and pd.notna(row["maturity_date"]):
        return pd.Timestamp(row["maturity_date"])

    maturity_years = float(row["maturity"])
    months = int(round(maturity_years * 12))
    return pd.Timestamp(settlement_date) + pd.DateOffset(months=months)


def bond_value(
    face_value,
    coupon,
    settlement_date,
    maturity_date,
    curve_fn,
    frequency,
    day_count="ACT/ACT",
):
    if 12 % int(frequency) != 0:
        raise ValueError("Frequency must divide 12")

    settlement_date = pd.Timestamp(settlement_date)
    maturity_date = pd.Timestamp(maturity_date)
    frequency = int(frequency)

    pv = 0.0
    months_step = 12 // frequency
    i = 0

    while maturity_date - pd.DateOffset(months=i * months_step) > settlement_date:
        cf_date = maturity_date - pd.DateOffset(months=i * months_step)
        t = year_fraction(settlement_date, cf_date, day_count)
        cashflow = (coupon / frequency) * face_value

        if i == 0:
            cashflow += face_value

        rate = float(curve_fn(t))
        discount_factor = 1 / pow(1 + rate / frequency, frequency * t)
        pv += cashflow * discount_factor
        i += 1

    if i == 0:
        return {
            "dirty_value": 0.0,
            "clean_value": 0.0,
            "accrued_interest": 0.0,
        }

    next_coupon_date = maturity_date - pd.DateOffset(months=(i - 1) * months_step)
    last_coupon_date = maturity_date - pd.DateOffset(months=i * months_step)

    accrued_interest = 0.0
    if last_coupon_date < settlement_date < next_coupon_date:
        accrual_time_fraction = year_fraction(last_coupon_date, settlement_date, day_count)
        coupon_time_fraction = year_fraction(last_coupon_date, next_coupon_date, day_count)
        accrued_interest = (accrual_time_fraction / coupon_time_fraction) * (coupon / frequency) * face_value

    dirty_value = pv
    clean_value = dirty_value - accrued_interest

    return {
        "dirty_value": dirty_value,
        "clean_value": clean_value,
        "accrued_interest": accrued_interest,
    }


def bond_pricer(face_value, coupon, maturity, rate_curve, spline, frequency):
    settlement_date = pd.Timestamp("today").normalize()
    maturity_date = settlement_date + pd.DateOffset(months=int(round(float(maturity) * 12)))
    return bond_value(face_value, coupon, settlement_date, maturity_date, spline, frequency)["dirty_value"]


def value_bond_row(row, rate_curve, settlement_date=None):
    if settlement_date is None:
        if "settlement_date" in row.index and pd.notna(row["settlement_date"]):
            settlement_date = row["settlement_date"]
        else:
            settlement_date = pd.Timestamp(rate_curve.name) if rate_curve.name is not None else pd.Timestamp("today")

    settlement_date = pd.Timestamp(settlement_date)
    maturity_date = _infer_maturity_date(row, settlement_date)
    day_count = row.get("day_count", "ACT/ACT")
    curve_fn = _curve_interpolator(rate_curve)

    values = bond_value(
        float(row["face_value"]),
        float(row["coupon"]),
        settlement_date,
        maturity_date,
        curve_fn,
        int(row["frequency"]),
        day_count=day_count,
    )
    position = float(row.get("positions", 1))
    return {key: value * position for key, value in values.items()}


def portfolio_valuation(portfolio, rate_curve, settlement_date=None):
    rows = []
    for _, row in portfolio.iterrows():
        rows.append(value_bond_row(row, rate_curve, settlement_date=settlement_date))
    if not rows:
        return {
            "dirty_value": 0.0,
            "clean_value": 0.0,
            "accrued_interest": 0.0,
        }
    return pd.DataFrame(rows).sum().to_dict()


def portfolio_value(portfolio, rate_curve, settlement_date=None):
    return portfolio_valuation(portfolio, rate_curve, settlement_date=settlement_date)["dirty_value"]
