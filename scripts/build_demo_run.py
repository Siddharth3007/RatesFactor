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
from ratesfactor.portfolio import (
    build_hedged_portfolio,
    load_custom_hedge_instruments,
    load_standard_holdings,
    mild_mismatch_hedge_instruments,
    toy_portfolio,
)
from ratesfactor.pca import fit_pca, fit_rolling_pca
from ratesfactor.var import backtest_historical_var_table, compute_historical_var, compute_parametric_var


HISTORY_YEARS = 5
DAY_COUNT = "ACT/ACT"
LOOKBACK = 252
MAX_BACKTEST_DAYS = 750
PCA_STRIDE = 10
VAR_BACKTEST_LOOKBACK = 300
RIDGE_LAMBDA = 0.10
ALPHA = 0.05


def load_fred_api_key():
    if os.getenv("FRED_API_KEY"):
        return os.environ["FRED_API_KEY"]

    secrets_path = PROJECT_ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as handle:
            secrets = tomllib.load(handle)
        if secrets.get("FRED_API_KEY"):
            return secrets["FRED_API_KEY"]

    raise RuntimeError("Set FRED_API_KEY or add .streamlit/secrets.toml before building the demo artifacts.")


def load_rates_data(api_key, holdings_as_of_ts):
    start_date = holdings_as_of_ts - pd.DateOffset(years=HISTORY_YEARS)
    rates_pct = fetch_treasury_rates(
        TREASURY_SERIES,
        api_key,
        start_date.strftime("%Y-%m-%d"),
        holdings_as_of_ts.strftime("%Y-%m-%d"),
    )
    zero_rates_pct = bootstrap_zero_from_par(rates_pct, frequency=2, dc_conv=DAY_COUNT)
    return RatesData(zero_rates_pct)


def build_state(
    *,
    demo_label,
    demo_description,
    holdings_as_of_ts,
    portfolio,
    hedge_instruments,
    cost_bps,
    target_notional,
    api_key,
):
    rates_data = load_rates_data(api_key, holdings_as_of_ts)
    n_components = min(3, len(hedge_instruments))

    curve_as_of_date = pd.Timestamp(rates_data.latest_date)
    base_curve = rates_data.get_curve(curve_as_of_date, units="decimal")
    latest_pca = fit_pca(rates_data.daily_changes_bp.iloc[-LOOKBACK:], n_components=3)
    static_pca = fit_pca(rates_data.daily_changes_bp, n_components=3)

    hedge_weights = compute_pca_hedge_weights(
        portfolio,
        hedge_instruments,
        rates_data,
        curve_as_of_date,
        latest_pca,
        n_components=n_components,
        ridge_lambda=RIDGE_LAMBDA,
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
        lookback=LOOKBACK,
        n_components=3,
        max_windows=MAX_BACKTEST_DAYS + 1,
        stride=PCA_STRIDE,
    )
    backtest_results = run_pca_hedge_backtest(
        portfolio,
        hedge_instruments,
        rates_data,
        rolling_pca,
        cost_bps,
        lookback=LOOKBACK,
        n_components=n_components,
        ridge_lambda=RIDGE_LAMBDA,
        max_backtest_days=MAX_BACKTEST_DAYS,
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
        alpha=ALPHA,
        lookback=LOOKBACK,
    )
    param_var = compute_parametric_var(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        latest_pca,
        alpha=ALPHA,
        lookback=LOOKBACK,
        n_components=n_components,
    )
    var_backtest = backtest_historical_var_table(results_df, alpha=ALPHA, lookback=VAR_BACKTEST_LOOKBACK)

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
        "lookback": LOOKBACK,
        "var_backtest_lookback": VAR_BACKTEST_LOOKBACK,
        "pca_stride": PCA_STRIDE,
        "alpha": ALPHA,
        "max_backtest_days": MAX_BACKTEST_DAYS,
        "curve_fit_method": "NSS",
        "hist_var": hist_var,
        "param_var": param_var,
        "var_backtest": var_backtest,
        "target_notional": target_notional,
        "demo_label": demo_label,
        "demo_description": demo_description,
        "is_precomputed_demo": True,
    }


def build_toy_demo_state(api_key):
    holdings_as_of_ts = pd.Timestamp("2026-08-15")
    portfolio = toy_portfolio(settlement_date=holdings_as_of_ts, day_count=DAY_COUNT)
    hedge_instruments = mild_mismatch_hedge_instruments(holdings_as_of_ts, DAY_COUNT)
    cost_bps = np.array([0.35, 0.45, 0.65])

    return build_state(
        demo_label="Demo 1: Toy Treasury book",
        demo_description=(
            "Default toy Treasury portfolio, middle-end 3Y/7Y/20Y hedge universe, "
            "750-day hedge backtest, fixed rebalance stride, and 300-day VaR backtest."
        ),
        holdings_as_of_ts=holdings_as_of_ts,
        portfolio=portfolio,
        hedge_instruments=hedge_instruments,
        cost_bps=cost_bps,
        target_notional=float(portfolio["face_value"].sum()),
        api_key=api_key,
    )


def build_ief_demo_state(api_key):
    holdings_as_of_ts = pd.Timestamp("2026-08-19")
    portfolio = load_standard_holdings(
        PROJECT_ROOT / "downloaded_holdings" / "IEF_dashboard_standard_holdings_2026-08-19.xlsx",
        as_of_date=holdings_as_of_ts,
        target_notional=100_000_000,
        day_count=DAY_COUNT,
    )
    hedge_instruments, cost_bps = load_custom_hedge_instruments(
        PROJECT_ROOT / "downloaded_holdings" / "IEF_hedge_universe_2026-08-19.xlsx",
        settlement_date=holdings_as_of_ts,
        day_count=DAY_COUNT,
    )

    return build_state(
        demo_label="Demo 2: IEF holdings + custom hedge universe",
        demo_description=(
            "IEF-style Treasury holdings with a custom 5Y/7Y/10Y/20Y hedge universe, "
            "holdings/risk as-of 2026-08-19, 750-day hedge backtest, fixed rebalance stride, "
            "and 300-day VaR backtest."
        ),
        holdings_as_of_ts=holdings_as_of_ts,
        portfolio=portfolio,
        hedge_instruments=hedge_instruments,
        cost_bps=cost_bps,
        target_notional=100_000_000,
        api_key=api_key,
    )


def build_demo_state():
    # Backward-compatible helper for callers that expect a single default demo.
    api_key = load_fred_api_key()
    return build_toy_demo_state(api_key)


def main():
    api_key = load_fred_api_key()
    output_dir = PROJECT_ROOT / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    demo_specs = [
        ("demo_run.pkl", build_toy_demo_state),
        ("demo_run_ief.pkl", build_ief_demo_state),
    ]

    for file_name, builder in demo_specs:
        output_path = output_dir / file_name
        state = builder(api_key)
        with output_path.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Wrote {output_path}")
        print(f"  Demo: {state['demo_label']}")
        print(f"  Backtest rows: {len(state['results_df'])}")
        print(f"  Curve as-of date: {state['curve_as_of_date'].date()}")


if __name__ == "__main__":
    main()
