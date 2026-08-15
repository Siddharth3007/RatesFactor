import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ratesfactor.bootstrapper import bootstrap_zero_from_par
from ratesfactor.config import TREASURY_SERIES
from ratesfactor.data import RatesData, fetch_treasury_rates
from ratesfactor.hedging import (
    compute_pca_hedge_weights,
    hedge_diagnostics,
    run_pca_hedge_backtest,
    summarize_backtest_results,
)
from ratesfactor.portfolio import build_hedged_portfolio, mild_mismatch_hedge_instruments, toy_portfolio
from ratesfactor.pca import fit_pca, fit_rolling_pca
from ratesfactor.var import backtest_historical_var_table, compute_historical_var, compute_parametric_var


def load_fred_api_key():
    if os.getenv("FRED_API_KEY"):
        return os.environ["FRED_API_KEY"]

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as handle:
            secrets = tomllib.load(handle)
        if secrets.get("FRED_API_KEY"):
            return secrets["FRED_API_KEY"]

    raise RuntimeError("Set FRED_API_KEY or add .streamlit/secrets.toml before building the demo artifact.")


def build_demo_state():
    api_key = load_fred_api_key()

    holdings_as_of_ts = pd.Timestamp("2026-08-15")
    history_years = 5
    day_count = "ACT/ACT"
    lookback = 252
    max_backtest_days = 750
    pca_stride = 10
    var_backtest_lookback = 300
    ridge_lambda = 0.10
    alpha = 0.05
    n_components = 3

    start_date = holdings_as_of_ts - pd.DateOffset(years=history_years)
    rates_pct = fetch_treasury_rates(
        TREASURY_SERIES,
        api_key,
        start_date.strftime("%Y-%m-%d"),
        holdings_as_of_ts.strftime("%Y-%m-%d"),
    )
    zero_rates_pct = bootstrap_zero_from_par(rates_pct, frequency=2, dc_conv=day_count)
    rates_data = RatesData(zero_rates_pct)

    portfolio = toy_portfolio(settlement_date=holdings_as_of_ts, day_count=day_count)
    hedge_instruments = mild_mismatch_hedge_instruments(holdings_as_of_ts, day_count)
    cost_bps = np.array([0.35, 0.45, 0.65])

    curve_as_of_date = pd.Timestamp(rates_data.latest_date)
    base_curve = rates_data.get_curve(curve_as_of_date, units="decimal")
    latest_pca = fit_pca(rates_data.daily_changes_bp.iloc[-lookback:], n_components=3)
    static_pca = fit_pca(rates_data.daily_changes_bp, n_components=3)

    hedge_weights = compute_pca_hedge_weights(
        portfolio,
        hedge_instruments,
        rates_data,
        curve_as_of_date,
        latest_pca,
        n_components=n_components,
        ridge_lambda=ridge_lambda,
    )
    hedged_portfolio = build_hedged_portfolio(portfolio, hedge_instruments, hedge_weights)
    hedge_diag = hedge_diagnostics(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        curve_as_of_date,
        latest_pca,
        n_components=n_components,
    )

    rolling_pca = fit_rolling_pca(
        rates_data.daily_changes_bp,
        lookback=lookback,
        n_components=3,
        max_windows=max_backtest_days + 1,
        stride=pca_stride,
    )
    backtest_results = run_pca_hedge_backtest(
        portfolio,
        hedge_instruments,
        rates_data,
        rolling_pca,
        cost_bps,
        lookback=lookback,
        n_components=n_components,
        ridge_lambda=ridge_lambda,
        max_backtest_days=max_backtest_days,
    )
    results_df, summary, hedge_labels, hedge_weight_cols, _ = summarize_backtest_results(
        backtest_results,
        hedge_instruments,
    )

    hist_var = compute_historical_var(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        alpha=alpha,
        lookback=lookback,
    )
    param_var = compute_parametric_var(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        latest_pca,
        alpha=alpha,
        lookback=lookback,
        n_components=n_components,
    )
    var_backtest = backtest_historical_var_table(results_df, alpha=alpha, lookback=var_backtest_lookback)

    return {
        "portfolio": portfolio,
        "rates_data": rates_data,
        "curve_as_of_date": curve_as_of_date,
        "base_curve": base_curve,
        "zero_curve": None,
        "curve_universe": None,
        "pricing_curve_label": "FRED CMT-implied zero curve",
        "hedge_instruments": hedge_instruments,
        "cost_bps": cost_bps,
        "n_components": n_components,
        "latest_pca": latest_pca,
        "static_pca": static_pca,
        "hedge_weights": hedge_weights,
        "hedged_portfolio": hedged_portfolio,
        "hedge_diag": hedge_diag,
        "results_df": results_df,
        "summary": summary,
        "hedge_labels": hedge_labels,
        "hedge_weight_cols": hedge_weight_cols,
        "lookback": lookback,
        "var_backtest_lookback": var_backtest_lookback,
        "pca_stride": pca_stride,
        "alpha": alpha,
        "max_backtest_days": max_backtest_days,
        "curve_fit_method": "NSS",
        "hist_var": hist_var,
        "param_var": param_var,
        "var_backtest": var_backtest,
        "is_precomputed_demo": True,
    }


def main():
    output_path = PROJECT_ROOT / "assets" / "demo_run.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state = build_demo_state()
    with output_path.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {output_path}")
    print(f"Backtest rows: {len(state['results_df'])}")
    print(f"Curve as-of date: {state['curve_as_of_date'].date()}")


if __name__ == "__main__":
    main()
