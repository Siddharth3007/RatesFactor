import pandas as pd
import numpy as np

from .curves import shock_curve
from .portfolio import build_hedged_portfolio
from .pricing import portfolio_value


IRRBB_USD_SHOCKS_BP = {
    "parallel": 200,
    "short": 300,
    "long": 225,
}


def generate_irrbb_usd_scenarios(tenors):
    tenors = np.asarray(tenors, dtype=float)
    x = 4.0

    parallel = IRRBB_USD_SHOCKS_BP["parallel"]
    short = IRRBB_USD_SHOCKS_BP["short"] * np.exp(-tenors / x)
    long = IRRBB_USD_SHOCKS_BP["long"] * (1 - np.exp(-tenors / x))

    return {
        "irrbb_usd_parallel_up": np.full_like(tenors, parallel, dtype=float),
        "irrbb_usd_parallel_down": np.full_like(tenors, -parallel, dtype=float),
        "irrbb_usd_steepener": -0.65 * short + 0.90 * long,
        "irrbb_usd_flattener": 0.80 * short - 0.60 * long,
        "irrbb_usd_short_rate_up": short,
        "irrbb_usd_short_rate_down": -short,
    }


def run_scenario_analysis(portfolio, hedge_instruments, hedge_weights, base_curve, scenario_shocks_bp, settlement_date=None):
    shock_decimal = scenario_shocks_bp / 10000
    shocked_curve = shock_curve(base_curve, shock_decimal)
    hedged_portfolio = build_hedged_portfolio(portfolio, hedge_instruments, hedge_weights)

    unhedged_pnl = (
        portfolio_value(portfolio, shocked_curve, settlement_date=settlement_date)
        - portfolio_value(portfolio, base_curve, settlement_date=settlement_date)
    )
    hedged_pnl = (
        portfolio_value(hedged_portfolio, shocked_curve, settlement_date=settlement_date)
        - portfolio_value(hedged_portfolio, base_curve, settlement_date=settlement_date)
    )
    pnl_reduction = abs(unhedged_pnl) - abs(hedged_pnl)
    return unhedged_pnl, hedged_pnl, pnl_reduction


def run_multi_scenario_analysis(portfolio, hedge_instruments, hedge_weights, base_curve, scenarios, settlement_date=None):
    results = {
        "scenario_name": [],
        "unhedged_pnl": [],
        "hedged_pnl": [],
        "pnl_reduction": [],
    }

    for scenario_name, scenario_shocks_bp in scenarios.items():
        unhedged_pnl, hedged_pnl, pnl_reduction = run_scenario_analysis(
            portfolio,
            hedge_instruments,
            hedge_weights,
            base_curve,
            scenario_shocks_bp,
            settlement_date=settlement_date,
        )
        results["scenario_name"].append(scenario_name)
        results["unhedged_pnl"].append(unhedged_pnl)
        results["hedged_pnl"].append(hedged_pnl)
        results["pnl_reduction"].append(pnl_reduction)

    df = pd.DataFrame(results)
    df["abs_unhedged_pnl"] = df["unhedged_pnl"].abs()
    df["abs_hedged_pnl"] = df["hedged_pnl"].abs()
    df["hedge_helped"] = df["abs_hedged_pnl"] < df["abs_unhedged_pnl"]
    df["pnl_reduction_pct"] = df["pnl_reduction"] / df["abs_unhedged_pnl"]
    return df.sort_values("abs_unhedged_pnl", ascending=False)
