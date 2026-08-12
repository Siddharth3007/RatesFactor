import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from .portfolio import build_hedged_portfolio
from .risk import make_delta_ladder
from .scenarios import run_scenario_analysis


def compute_historical_var(portfolio, hedge_instruments, hedge_weights, rates_data, alpha=0.05, lookback=None):
    latest_date = rates_data.latest_date
    base_curve = rates_data.rates_decimal.loc[latest_date]
    daily_changes_bp = rates_data.daily_changes_bp.copy()

    if lookback is not None:
        daily_changes_bp = daily_changes_bp.iloc[-lookback:]

    hedged_pnls = []
    unhedged_pnls = []

    for _, row in daily_changes_bp.iterrows():
        scenario_shocks_bp = np.array(row)
        unhedged_pnl, hedged_pnl, _ = run_scenario_analysis(
            portfolio,
            hedge_instruments,
            hedge_weights,
            base_curve,
            scenario_shocks_bp,
            settlement_date=latest_date,
        )
        unhedged_pnls.append(unhedged_pnl)
        hedged_pnls.append(hedged_pnl)

    hedged_pnls = np.array(hedged_pnls)
    unhedged_pnls = np.array(unhedged_pnls)
    loss_hedged = -hedged_pnls
    loss_unhedged = -unhedged_pnls

    var_hedged = np.quantile(loss_hedged, 1 - alpha)
    var_unhedged = np.quantile(loss_unhedged, 1 - alpha)
    es_hedged = np.mean(loss_hedged[loss_hedged >= var_hedged])
    es_unhedged = np.mean(loss_unhedged[loss_unhedged >= var_unhedged])

    return pd.DataFrame({
        "portfolio": ["unhedged", "hedged"],
        "VaR": [var_unhedged, var_hedged],
        "Expected Shortfall": [es_unhedged, es_hedged],
        "worst_pnl": [unhedged_pnls.min(), hedged_pnls.min()],
        "best_pnl": [unhedged_pnls.max(), hedged_pnls.max()],
        "avg_pnl": [unhedged_pnls.mean(), hedged_pnls.mean()],
        "pnl_vol": [unhedged_pnls.std(), hedged_pnls.std()],
    })


def kupiec_unconditional_coverage(actual_breaches, days, alpha):
    if days <= 0:
        return np.nan, np.nan

    actual_breaches = int(actual_breaches)
    days = int(days)
    breach_rate = actual_breaches / days

    def log_likelihood(probability):
        if probability <= 0:
            return 0.0 if actual_breaches == 0 else -np.inf
        if probability >= 1:
            return 0.0 if actual_breaches == days else -np.inf
        return (
            (days - actual_breaches) * np.log(1 - probability)
            + actual_breaches * np.log(probability)
        )

    null_ll = log_likelihood(alpha)
    fitted_ll = log_likelihood(breach_rate)
    lr_stat = -2 * (null_ll - fitted_ll)
    if not np.isfinite(lr_stat):
        return np.nan, np.nan
    p_value = 1 - chi2.cdf(lr_stat, df=1)
    return float(lr_stat), float(p_value)


def backtest_historical_var_from_pnl(pnl_series, alpha=0.05, lookback=252):
    pnl_series = pd.Series(pnl_series).dropna()
    records = []

    for idx in range(lookback, len(pnl_series)):
        pnl_window = pnl_series.iloc[idx - lookback : idx]
        var_t = np.quantile(-pnl_window.to_numpy(dtype=float), 1 - alpha)
        realized_loss = -float(pnl_series.iloc[idx])
        records.append({
            "date": pnl_series.index[idx],
            "VaR": var_t,
            "realized_loss": realized_loss,
            "breach": realized_loss > var_t,
        })

    if not records:
        return {
            "days": 0,
            "expected_breaches": 0.0,
            "actual_breaches": 0,
            "breach_rate": np.nan,
            "expected_breach_rate": alpha,
            "latest_var": np.nan,
            "kupiec_lr": np.nan,
            "kupiec_p_value": np.nan,
            "kupiec_pass_5pct": False,
        }

    backtest = pd.DataFrame(records).set_index("date")
    days = len(backtest)
    actual_breaches = int(backtest["breach"].sum())
    kupiec_lr, kupiec_p_value = kupiec_unconditional_coverage(actual_breaches, days, alpha)
    return {
        "days": days,
        "expected_breaches": alpha * days,
        "actual_breaches": actual_breaches,
        "breach_rate": float(backtest["breach"].mean()),
        "expected_breach_rate": alpha,
        "latest_var": float(backtest["VaR"].iloc[-1]),
        "kupiec_lr": kupiec_lr,
        "kupiec_p_value": kupiec_p_value,
        "kupiec_pass_5pct": bool(kupiec_p_value >= 0.05) if not np.isnan(kupiec_p_value) else False,
    }


def backtest_historical_var_table(results_df, alpha=0.05, lookback=252):
    series_map = {
        "unhedged": "unhedged_pnl",
        "gross_hedged": "hedged_pnl",
        "net_hedged": "net_hedged_pnl",
    }
    rows = []

    for portfolio_name, column in series_map.items():
        if column not in results_df.columns:
            continue
        row = backtest_historical_var_from_pnl(results_df[column], alpha=alpha, lookback=lookback)
        row["portfolio"] = portfolio_name
        row["VaR Method"] = "Rolling historical"
        rows.append(row)

    columns = [
        "portfolio",
        "VaR Method",
        "days",
        "expected_breaches",
        "actual_breaches",
        "breach_rate",
        "expected_breach_rate",
        "latest_var",
        "kupiec_lr",
        "kupiec_p_value",
        "kupiec_pass_5pct",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def compute_parametric_var(
    portfolio,
    hedge_instruments,
    hedge_weights,
    rates_data,
    pca,
    alpha=0.05,
    lookback=252,
    n_components=3,
):
    date = rates_data.latest_date
    daily_changes_bp = rates_data.daily_changes_bp.copy()
    if lookback is not None:
        daily_changes_bp = daily_changes_bp.iloc[-lookback:]

    factor_scores = pca.transform(daily_changes_bp)[:, :n_components]
    factor_cov = np.cov(factor_scores, rowvar=False)
    V = pca.components_[:n_components]

    hedged_portfolio = build_hedged_portfolio(portfolio, hedge_instruments, hedge_weights)
    portfolios = {"Unhedged": portfolio, "Hedged": hedged_portfolio}

    z = norm.ppf(1 - alpha)
    phi = norm.pdf(z)
    results = {}

    for name, port in portfolios.items():
        dv01_ladder = make_delta_ladder(port, rates_data, date)
        factor_exposures = V @ dv01_ladder
        pnl_vol = np.sqrt(factor_exposures.T @ factor_cov @ factor_exposures)
        results[name] = {
            "VaR": z * pnl_vol,
            "Expected Shortfall": phi * pnl_vol / alpha,
            "pnl_vol": pnl_vol,
            "factor_exposures": factor_exposures,
            "alpha": alpha,
            "confidence_level": 1 - alpha,
            "lookback": lookback,
            "n_components": n_components,
        }

    return results
