import re

import numpy as np
import pandas as pd

from .portfolio import build_hedged_portfolio
from .pricing import portfolio_value
from .risk import make_delta_ladder


def hedge_design_matrix(portfolio, hedge_instruments, rates_data, date, pca, n_components=3):
    date = pd.Timestamp(date)
    portfolio_ladder = make_delta_ladder(portfolio, rates_data, date)
    hedge_ladders = []

    for i in range(len(hedge_instruments)):
        hedge_instrument = hedge_instruments.iloc[[i]]
        hedge_ladders.append(make_delta_ladder(hedge_instrument, rates_data, date))

    hedge_ladders = np.array(hedge_ladders)
    v = pca.components_[:n_components]
    A = v @ hedge_ladders.T
    y = -(v @ portfolio_ladder.T)
    return A, y, portfolio_ladder, hedge_ladders


def solve_regularized_hedge(A, y, ridge_lambda=1e-4):
    if ridge_lambda <= 0:
        h, *_ = np.linalg.lstsq(A, y, rcond=None)
        return h

    n_hedges = A.shape[1]
    gram = A.T @ A
    scale = np.trace(gram) / n_hedges
    penalty = ridge_lambda * scale if scale > 0 else ridge_lambda
    return np.linalg.solve(gram + penalty * np.eye(n_hedges), A.T @ y)


def compute_pca_hedge_weights(
    portfolio,
    hedge_instruments,
    rates_data,
    date,
    pca,
    n_components=3,
    ridge_lambda=1e-4,
):
    A, y, *_ = hedge_design_matrix(portfolio, hedge_instruments, rates_data, date, pca, n_components)
    return solve_regularized_hedge(A, y, ridge_lambda=ridge_lambda)


def hedge_diagnostics(
    portfolio,
    hedge_instruments,
    hedge_weights,
    rates_data,
    date,
    pca,
    n_components=3,
    coverage_multiplier=1.5,
):
    A, y, portfolio_ladder, _ = hedge_design_matrix(portfolio, hedge_instruments, rates_data, date, pca, n_components)
    residual = A @ hedge_weights - y
    condition_number = np.linalg.cond(A) if A.size else np.nan
    portfolio_notional = float((portfolio["face_value"] * portfolio.get("positions", 1)).abs().sum())
    gross_hedge_notional = float((hedge_instruments["face_value"].values * np.abs(hedge_weights)).sum())
    hedge_notional_ratio = np.nan if portfolio_notional == 0 else gross_hedge_notional / portfolio_notional

    portfolio_exposure = (portfolio["face_value"] * portfolio.get("positions", 1)).abs()
    hedge_exposure = hedge_instruments["face_value"].abs()
    portfolio_wam = float((portfolio["maturity"] * portfolio_exposure).sum() / portfolio_exposure.sum())
    hedge_wam = float((hedge_instruments["maturity"] * hedge_exposure).sum() / hedge_exposure.sum())
    max_hedge_maturity = float(hedge_instruments["maturity"].max())
    top_krd_idx = int(np.argmax(np.abs(portfolio_ladder)))
    top_krd_bucket = float(rates_data.tenors[top_krd_idx])
    maturity_ratio = np.inf if hedge_wam == 0 else portfolio_wam / hedge_wam
    top_bucket_outside_coverage = top_krd_bucket > max_hedge_maturity * coverage_multiplier

    warnings = []
    severity = "Good"

    if top_bucket_outside_coverage:
        severity = "Warning"
        warnings.append(
            "The portfolio's largest key-rate DV01 bucket is materially beyond the longest hedge instrument. "
            "Consider using a hedge universe with longer-dated instruments."
        )

    return {
        "severity": severity,
        "warnings": warnings,
        "condition_number": float(condition_number),
        "gross_hedge_notional": gross_hedge_notional,
        "portfolio_notional": portfolio_notional,
        "hedge_notional_ratio": float(hedge_notional_ratio),
        "portfolio_wam": portfolio_wam,
        "hedge_wam": hedge_wam,
        "max_hedge_maturity": max_hedge_maturity,
        "top_krd_bucket": top_krd_bucket,
        "factor_residual_norm": float(np.linalg.norm(residual)),
        "target_factor_exposure_norm": float(np.linalg.norm(y)),
    }


# Notebook-compatible alias.
pca_hedge = compute_pca_hedge_weights


def make_hedge_label(row):
    if "bond" in row.index:
        label = str(row["bond"])
    else:
        label = f"{row['maturity']}Y"

    label = label.lower()
    for token in ["treasury", "bill", "note", "bond", "hedge"]:
        label = label.replace(token, "")
    label = re.sub(r"[^a-z0-9]+", "_", label).strip("_")
    return label or f"{row['maturity']}y"


def hedge_column_names(hedge_instruments):
    labels = [make_hedge_label(hedge_instruments.iloc[i]) for i in range(len(hedge_instruments))]
    return labels, [f"h_{label}" for label in labels], [f"turnover_{label}" for label in labels]


def run_pca_hedge_backtest(
    portfolio,
    hedge_instruments,
    rates_data,
    rolling_pca,
    cost_bps,
    lookback=252,
    n_components=3,
    ridge_lambda=1e-4,
    max_backtest_days=None,
):
    if len(cost_bps) != len(hedge_instruments):
        raise ValueError("cost_bps must have the same length as hedge_instruments")

    results = {}
    prev_h = None
    dates = np.array(list(rates_data.daily_changes_bp.index))

    backtest_dates = list(zip(dates[lookback - 1 : -1], dates[lookback:]))
    if max_backtest_days is not None:
        backtest_dates = backtest_dates[-int(max_backtest_days):]

    for date, next_date in backtest_dates:
        date = pd.Timestamp(date)
        next_date = pd.Timestamp(next_date)

        h = compute_pca_hedge_weights(
            portfolio,
            hedge_instruments,
            rates_data,
            date,
            rolling_pca[date]["pca"],
            n_components=n_components,
            ridge_lambda=ridge_lambda,
        )
        hedged_portfolio = build_hedged_portfolio(portfolio, hedge_instruments, h)

        rate_curve_next = rates_data.get_curve(next_date, "decimal")
        rate_curve_curr = rates_data.get_curve(date, "decimal")

        unhedged_pnl = (
            portfolio_value(portfolio, rate_curve_next, settlement_date=next_date)
            - portfolio_value(portfolio, rate_curve_curr, settlement_date=date)
        )
        hedged_pnl = (
            portfolio_value(hedged_portfolio, rate_curve_next, settlement_date=next_date)
            - portfolio_value(hedged_portfolio, rate_curve_curr, settlement_date=date)
        )

        turnover = abs(h) if prev_h is None else abs(h - prev_h)
        prev_h = h.copy()

        traded_notional = turnover * hedge_instruments["face_value"].values
        transaction_cost = (cost_bps * traded_notional).sum() / 10000
        net_hedged_pnl = hedged_pnl - transaction_cost

        results[date] = {
            "unhedged_pnl": unhedged_pnl,
            "hedged_pnl": hedged_pnl,
            "net_hedged_pnl": net_hedged_pnl,
            "pnl_reduction": abs(unhedged_pnl) - abs(hedged_pnl),
            "net_pnl_reduction": abs(unhedged_pnl) - abs(net_hedged_pnl),
            "hedge_weights": h,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
        }

    return results


def summarize_backtest_results(results, hedge_instruments):
    results_df = pd.DataFrame.from_dict(results, orient="index")
    results_df.index = pd.to_datetime(results_df.index)
    results_df = results_df.sort_index()
    results_df.index.name = "date"

    hedge_labels, hedge_weight_cols, turnover_cols = hedge_column_names(hedge_instruments)

    hedge_weights_df = pd.DataFrame(
        results_df["hedge_weights"].to_list(),
        index=results_df.index,
        columns=hedge_weight_cols,
    )
    results_df = pd.concat([results_df.drop(columns=["hedge_weights"]), hedge_weights_df], axis=1)

    if "turnover" in results_df.columns:
        turnover_df = pd.DataFrame(
            results_df["turnover"].to_list(),
            index=results_df.index,
            columns=turnover_cols,
        )
        results_df = pd.concat([results_df.drop(columns=["turnover"]), turnover_df], axis=1)

    results_df["abs_unhedged_pnl"] = results_df["unhedged_pnl"].abs()
    results_df["abs_gross_hedged_pnl"] = results_df["hedged_pnl"].abs()
    results_df["abs_net_hedged_pnl"] = results_df["net_hedged_pnl"].abs()
    results_df["gross_hedge_helped"] = results_df["abs_gross_hedged_pnl"] < results_df["abs_unhedged_pnl"]
    results_df["net_hedge_helped"] = results_df["abs_net_hedged_pnl"] < results_df["abs_unhedged_pnl"]
    results_df["gross_abs_pnl_reduction"] = results_df["abs_unhedged_pnl"] - results_df["abs_gross_hedged_pnl"]
    results_df["net_abs_pnl_reduction"] = results_df["abs_unhedged_pnl"] - results_df["abs_net_hedged_pnl"]
    results_df["cum_unhedged_pnl"] = results_df["unhedged_pnl"].cumsum()
    results_df["cum_gross_hedged_pnl"] = results_df["hedged_pnl"].cumsum()
    results_df["cum_net_hedged_pnl"] = results_df["net_hedged_pnl"].cumsum()
    results_df["cum_transaction_cost"] = results_df["transaction_cost"].cumsum()

    summary = pd.Series({
        "num_days": len(results_df),
        "avg_abs_unhedged_pnl": results_df["abs_unhedged_pnl"].mean(),
        "avg_abs_gross_hedged_pnl": results_df["abs_gross_hedged_pnl"].mean(),
        "avg_abs_net_hedged_pnl": results_df["abs_net_hedged_pnl"].mean(),
        "unhedged_pnl_vol": results_df["unhedged_pnl"].std(),
        "gross_hedged_pnl_vol": results_df["hedged_pnl"].std(),
        "net_hedged_pnl_vol": results_df["net_hedged_pnl"].std(),
        "worst_unhedged_pnl": results_df["unhedged_pnl"].min(),
        "worst_gross_hedged_pnl": results_df["hedged_pnl"].min(),
        "worst_net_hedged_pnl": results_df["net_hedged_pnl"].min(),
        "gross_hit_rate": results_df["gross_hedge_helped"].mean(),
        "net_hit_rate": results_df["net_hedge_helped"].mean(),
        "total_unhedged_pnl": results_df["unhedged_pnl"].sum(),
        "total_gross_hedged_pnl": results_df["hedged_pnl"].sum(),
        "total_net_hedged_pnl": results_df["net_hedged_pnl"].sum(),
        "total_transaction_cost": results_df["transaction_cost"].sum(),
        "avg_daily_transaction_cost": results_df["transaction_cost"].mean(),
    })

    summary["gross_avg_abs_pnl_reduction_pct"] = 1 - summary["avg_abs_gross_hedged_pnl"] / summary["avg_abs_unhedged_pnl"]
    summary["net_avg_abs_pnl_reduction_pct"] = 1 - summary["avg_abs_net_hedged_pnl"] / summary["avg_abs_unhedged_pnl"]
    summary["gross_pnl_vol_reduction_pct"] = 1 - summary["gross_hedged_pnl_vol"] / summary["unhedged_pnl_vol"]
    summary["net_pnl_vol_reduction_pct"] = 1 - summary["net_hedged_pnl_vol"] / summary["unhedged_pnl_vol"]
    return results_df, summary, hedge_labels, hedge_weight_cols, turnover_cols
