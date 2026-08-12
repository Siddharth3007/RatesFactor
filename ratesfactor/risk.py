import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .pricing import portfolio_value, value_bond_row


def bump_curve(rates_data, date, maturity, bump=0.0001):
    rate_curve = rates_data.get_curve(date, "decimal")
    rate_curve_up = rate_curve.copy()
    rate_curve_down = rate_curve.copy()
    rate_curve_up.loc[maturity] += bump
    rate_curve_down.loc[maturity] -= bump
    return rate_curve_up, rate_curve_down


def parallel_bump_curve(rates_data, date, bump=0.0001):
    rate_curve = rates_data.get_curve(date, "decimal")
    return rate_curve + bump, rate_curve - bump


def make_delta_ladder(portfolio, rates_data, date):
    date = pd.Timestamp(date)
    delta_ladder = []

    for maturity in rates_data.tenors:
        rates_up, rates_down = bump_curve(rates_data, date, maturity)
        price_up = portfolio_value(portfolio, rates_up, settlement_date=date)
        price_down = portfolio_value(portfolio, rates_down, settlement_date=date)
        delta_ladder.append((price_down - price_up) / 2)

    return np.array(delta_ladder)


def calc_dv01(portfolio, rates_data, date):
    date = pd.Timestamp(date)
    rates_up, rates_down = parallel_bump_curve(rates_data, date)
    price_up = portfolio_value(portfolio, rates_up, settlement_date=date)
    price_down = portfolio_value(portfolio, rates_down, settlement_date=date)
    return (price_down - price_up) / 2


def calc_effective_duration(portfolio, rates_data, date, bump=0.0001):
    date = pd.Timestamp(date)
    base_curve = rates_data.get_curve(date, "decimal")
    rates_up, rates_down = parallel_bump_curve(rates_data, date, bump=bump)
    value_base = portfolio_value(portfolio, base_curve, settlement_date=date)
    if value_base == 0:
        return np.nan

    value_up = portfolio_value(portfolio, rates_up, settlement_date=date)
    value_down = portfolio_value(portfolio, rates_down, settlement_date=date)
    return (value_down - value_up) / (2 * value_base * bump)


def calc_convexity(portfolio, rates_data, date, bump=0.0001):
    date = pd.Timestamp(date)
    base_curve = rates_data.get_curve(date, "decimal")
    rates_up, rates_down = parallel_bump_curve(rates_data, date, bump=bump)
    value_base = portfolio_value(portfolio, base_curve, settlement_date=date)
    if value_base == 0:
        return np.nan

    value_up = portfolio_value(portfolio, rates_up, settlement_date=date)
    value_down = portfolio_value(portfolio, rates_down, settlement_date=date)
    return (value_down + value_up - 2 * value_base) / (value_base * bump**2)


def curve_rate_for_maturity(rate_curve, maturity):
    spline = CubicSpline(
        rate_curve.index.astype(float).to_numpy(),
        rate_curve.to_numpy(dtype=float),
    )
    return float(spline(float(maturity)))


def line_item_bond_analytics(portfolio, rates_data, date):
    date = pd.Timestamp(date)
    base_curve = rates_data.get_curve(date, "decimal")
    rows = []

    for _, row in portfolio.iterrows():
        bond_df = pd.DataFrame([row])
        values = value_bond_row(row, base_curve, settlement_date=date)
        dirty_value = values["dirty_value"]
        face_position = float(row["face_value"]) * float(row.get("positions", 1))
        price_per_100 = np.nan if face_position == 0 else dirty_value / face_position * 100
        dv01 = calc_dv01(bond_df, rates_data, date)
        duration = calc_effective_duration(bond_df, rates_data, date)
        convexity = calc_convexity(bond_df, rates_data, date)
        krd = make_delta_ladder(bond_df, rates_data, date)
        abs_krd_sum = np.abs(krd).sum()
        largest_idx = int(np.argmax(np.abs(krd))) if len(krd) else 0
        largest_krd_bucket = rates_data.tenors[largest_idx] if len(krd) else np.nan
        largest_krd_share = np.nan if abs_krd_sum == 0 else abs(krd[largest_idx]) / abs_krd_sum

        rows.append({
            "bond": row.get("bond", ""),
            "face_value": row["face_value"],
            "positions": row.get("positions", 1),
            "coupon": row["coupon"],
            "maturity": row["maturity"],
            "maturity_date": row.get("maturity_date", pd.NaT),
            "curve_rate": curve_rate_for_maturity(base_curve, row["maturity"]),
            "clean_value": values["clean_value"],
            "accrued_interest": values["accrued_interest"],
            "dirty_value": dirty_value,
            "price_per_100": price_per_100,
            "dv01": dv01,
            "effective_duration": duration,
            "convexity": convexity,
            "largest_krd_bucket": largest_krd_bucket,
            "largest_krd_share": largest_krd_share,
        })

    return pd.DataFrame(rows)


def key_rate_risk_concentration(delta_ladder, tenors):
    delta_ladder = np.asarray(delta_ladder, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    abs_ladder = np.abs(delta_ladder)

    if len(abs_ladder) == 0 or abs_ladder.sum() == 0:
        return {
            "largest_bucket": np.nan,
            "largest_dv01": 0.0,
            "share_total_abs_krd": 0.0,
            "total_abs_krd": 0.0,
        }

    idx = int(np.argmax(abs_ladder))
    return {
        "largest_bucket": float(tenors[idx]),
        "largest_dv01": float(delta_ladder[idx]),
        "share_total_abs_krd": float(abs_ladder[idx] / abs_ladder.sum()),
        "total_abs_krd": float(abs_ladder.sum()),
    }


def dv01_hedge(portfolio, hedge_instrument, rates_data, date):
    dv01_portfolio = calc_dv01(portfolio, rates_data, date)
    dv01_hedge_instrument = calc_dv01(hedge_instrument, rates_data, date)
    return -dv01_portfolio / dv01_hedge_instrument
