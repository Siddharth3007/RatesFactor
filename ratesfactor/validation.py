import pandas as pd
import numpy as np

from .data import RatesData
from .portfolio import toy_portfolio
from .pricing import portfolio_valuation, value_bond_row
from .risk import calc_dv01
from .zerocurve import bootstrap_zero_curve, clean_curve_universe, curve_universe_template, zero_curve_to_rate_curve


def _par_yield_proxy_curve(curve_universe, tenors):
    universe = curve_universe.copy()
    universe["settlement_date"] = pd.to_datetime(universe["settlement_date"])
    universe["quote_date"] = pd.to_datetime(universe["quote_date"])
    rates = []
    maturities = []

    for _, row in universe.iterrows():
        t = float(row["time_to_maturity"])
        if int(row["coupon_frequency"]) == 0 or float(row["coupon_rate"]) == 0:
            discount_factor = float(row["dirty_price"]) / float(row["face_value"])
            rate = discount_factor ** (-1 / t) - 1
        else:
            rate = float(row["coupon_rate"])
        maturities.append(t)
        rates.append(rate)

    curve = pd.Series(rates, index=maturities).sort_index()
    return pd.Series(data=np.interp(tenors, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)), index=tenors)


def demo_pricing_error_summary():
    """Compare fitted/par-yield proxy pricing against the bundled bootstrapped curve."""
    curve_universe = clean_curve_universe(curve_universe_template())
    zero_curve = bootstrap_zero_curve(curve_universe)
    settlement_date = pd.Timestamp(curve_universe["settlement_date"].iloc[0])
    tenors = pd.Index([1 / 12, 0.25, 0.50, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)

    bootstrap_curve = zero_curve_to_rate_curve(zero_curve, tenors)
    par_proxy_curve = _par_yield_proxy_curve(curve_universe, tenors)
    bootstrap_curve.name = settlement_date
    par_proxy_curve.name = settlement_date

    portfolio = toy_portfolio(settlement_date=settlement_date, day_count="ACT/ACT")
    rows = []
    for _, bond in portfolio.iterrows():
        par_values = value_bond_row(bond, par_proxy_curve, settlement_date=settlement_date)
        boot_values = value_bond_row(bond, bootstrap_curve, settlement_date=settlement_date)
        face = float(bond["face_value"]) * float(bond.get("positions", 1))
        par_price = par_values["dirty_value"] / face * 100
        boot_price = boot_values["dirty_value"] / face * 100
        rows.append({
            "bond": bond["bond"],
            "maturity": float(bond["maturity"]),
            "par_proxy_price": par_price,
            "bootstrap_price": boot_price,
            "price_delta": boot_price - par_price,
        })

    rates_data_par = RatesData(pd.DataFrame([par_proxy_curve * 100], index=[settlement_date]))
    rates_data_boot = RatesData(pd.DataFrame([bootstrap_curve * 100], index=[settlement_date]))
    par_dv01 = calc_dv01(portfolio, rates_data_par, settlement_date)
    boot_dv01 = calc_dv01(portfolio, rates_data_boot, settlement_date)
    par_value = portfolio_valuation(portfolio, par_proxy_curve, settlement_date=settlement_date)["dirty_value"]
    boot_value = portfolio_valuation(portfolio, bootstrap_curve, settlement_date=settlement_date)["dirty_value"]

    table = pd.DataFrame(rows)
    return {
        "comparison_table": table,
        "max_abs_price_delta": float(table["price_delta"].abs().max()),
        "avg_abs_price_delta": float(table["price_delta"].abs().mean()),
        "portfolio_value_delta": float(boot_value - par_value),
        "dv01_delta": float(boot_dv01 - par_dv01),
    }
