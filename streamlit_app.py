import inspect
import os
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from ratesfactor.attribution import pca_pnl_attribution
from ratesfactor.config import PRESET_SCENARIOS_BP, TREASURY_SERIES
from ratesfactor.curves import compute_fwd, count_turning_points, cubic_spline_yield, fit_nss_grid, nss_yield
from ratesfactor.data import RatesData, fetch_treasury_rates
from ratesfactor.hedging import (
    compute_pca_hedge_weights,
    hedge_diagnostics,
    run_pca_hedge_backtest,
    summarize_backtest_results,
)
from ratesfactor.plots import backtest_figure, curve_figure, pca_loadings_figure, transaction_cost_figure
from ratesfactor.portfolio import (
    build_hedged_portfolio,
    front_end_hedge_instruments,
    load_custom_hedge_instruments,
    load_portfolio,
    long_duration_hedge_instruments,
    mild_mismatch_hedge_instruments,
    portfolio_summary_stats,
)
from ratesfactor.pca import fit_pca, fit_rolling_pca
from ratesfactor.pricing import portfolio_valuation, portfolio_value
from ratesfactor.risk import (
    calc_convexity,
    calc_dv01,
    calc_effective_duration,
    key_rate_risk_concentration,
    line_item_bond_analytics,
    make_delta_ladder,
)
from ratesfactor.scenarios import generate_irrbb_usd_scenarios, run_multi_scenario_analysis, run_scenario_analysis
from ratesfactor.templates import (
    curve_construction_universe_template,
    dataframe_to_csv_download,
    dataframe_to_xlsx_download,
    hedge_instruments_template,
    standard_holdings_template,
    tlt_holdings_template,
)
from ratesfactor.var import backtest_historical_var_table, compute_historical_var, compute_parametric_var
from ratesfactor.zerocurve import bootstrap_zero_curve, load_curve_universe, zero_curve_to_rate_curve


st.set_page_config(page_title="RatesFactor", layout="wide")


@st.cache_data(show_spinner=False)
def cached_fetch_rates(api_key, start_date, end_date):
    return fetch_treasury_rates(TREASURY_SERIES, api_key, start_date, end_date)


def fit_rolling_pca_for_rebalance(daily_changes_bp, lookback, n_components, max_windows, stride):
    kwargs = {
        "lookback": int(lookback),
        "n_components": int(n_components),
        "max_windows": int(max_windows),
    }
    if "stride" in inspect.signature(fit_rolling_pca).parameters:
        kwargs["stride"] = int(stride)
        return fit_rolling_pca(daily_changes_bp, **kwargs)

    rolling_pca = fit_rolling_pca(daily_changes_bp, **kwargs)
    if int(stride) <= 1:
        return rolling_pca

    return {
        date: pca_result
        for idx, (date, pca_result) in enumerate(rolling_pca.items())
        if idx % int(stride) == 0
    }


@st.cache_data(show_spinner=False)
def cached_template_download(template_name):
    if template_name == "Standard holdings":
        return (
            dataframe_to_xlsx_download(standard_holdings_template()),
            "ratesfactor_standard_holdings_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    if template_name == "TLT-style holdings":
        return (
            dataframe_to_csv_download(tlt_holdings_template()),
            "ratesfactor_tlt_style_holdings_template.csv",
            "text/csv",
        )
    if template_name == "Custom hedge instruments":
        return (
            dataframe_to_xlsx_download(hedge_instruments_template()),
            "ratesfactor_hedge_instruments_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    return (
        dataframe_to_xlsx_download(curve_construction_universe_template()),
        "ratesfactor_curve_construction_universe_template.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def format_money(x):
    return f"${x:,.0f}"


def format_tenor(x):
    if pd.isna(x):
        return "n/a"
    return f"{float(x):g}Y"


METRIC_HELP = {
    "Weighted avg maturity": "Face-value-weighted average maturity of the active portfolio.",
    "Weighted avg coupon": "Face-value-weighted average coupon across the active portfolio.",
    "Market / dirty value": "Present value including accrued interest, using the selected pricing curve source.",
    "DV01": "Dollar change in portfolio value for a 1 bp parallel move in rates.",
    "Effective duration": "Approximate percentage price sensitivity to a small parallel yield-curve shift.",
    "Convexity": "Second-order sensitivity of portfolio value to rate moves.",
    "Largest KRD bucket": "Treasury tenor where the portfolio has the largest absolute key-rate DV01.",
    "Largest KRD share": "Largest absolute key-rate DV01 divided by total absolute key-rate DV01.",
    "Suitability": "Default warning based on hedge-universe coverage. The threshold is a practical 1.5x heuristic, not a theoretical boundary.",
    "Condition number": "Higher values mean hedge instruments are close to collinear, making the hedge solve numerically unstable.",
    "Residual factor norm": "||Ah - y||: factor exposure left unneutralized after applying the hedge weights.",
    "Net hit rate": "Percent of days where net hedged P&L had smaller absolute magnitude than unhedged P&L.",
    "Net vol reduction": "Percent reduction in daily P&L volatility after transaction costs.",
    "Net P&L reduction": "Percent reduction in average absolute daily P&L after transaction costs.",
    "Gross P&L reduction": "Percent reduction in average absolute daily P&L before transaction costs.",
    "Forward turning points": "Number of local direction changes in the fitted forward curve. Useful as a curve-shape diagnostic, not a trading signal.",
}


def metric_with_help(container, label, value, key=None):
    return container.metric(label, value, help=METRIC_HELP.get(key or label))


def formatted_scenario_table(df):
    display_df = df.copy()
    for col in ["unhedged_pnl", "hedged_pnl", "pnl_reduction"]:
        if col in display_df:
            display_df[col] = display_df[col].map(format_money)
    return display_df.rename(
        columns={
            "scenario_name": "Scenario",
            "unhedged_pnl": "Unhedged P&L",
            "hedged_pnl": "Hedged P&L",
            "pnl_reduction": "Abs P&L Reduction",
        }
    )


def formatted_summary_table(summary):
    display = summary.to_frame("Value").reset_index().rename(columns={"index": "Metric"})
    pct_metrics = [name for name in display["Metric"] if "pct" in str(name) or "rate" in str(name)]
    money_metrics = [name for name in display["Metric"] if "cost" in str(name) or "pnl" in str(name)]

    def format_value(row):
        value = row["Value"]
        metric = row["Metric"]
        if metric in pct_metrics:
            return f"{value:.2%}"
        if metric in money_metrics:
            return format_money(value)
        return f"{value:,.4f}" if isinstance(value, float) else value

    display["Value"] = display.apply(format_value, axis=1)
    display["Metric"] = display["Metric"].str.replace("_", " ").str.title()
    return display


def formatted_var_table(df):
    display_df = df.copy()
    money_cols = ["VaR", "Expected Shortfall", "worst_pnl", "best_pnl", "avg_pnl", "pnl_vol"]
    for col in money_cols:
        if col in display_df:
            display_df[col] = display_df[col].map(format_money)
    return display_df


def formatted_parametric_var_table(param_var):
    display_df = pd.DataFrame(param_var).T.reset_index().rename(columns={"index": "Portfolio"})
    money_cols = ["VaR", "Expected Shortfall", "pnl_vol"]
    for col in money_cols:
        if col in display_df:
            display_df[col] = display_df[col].map(format_money)
    if "factor_exposures" in display_df:
        display_df["factor_exposures"] = display_df["factor_exposures"].map(
            lambda values: ", ".join(f"{float(x):,.2f}" for x in values)
        )
    if "alpha" in display_df:
        display_df["alpha"] = display_df["alpha"].map(lambda x: f"{float(x):.2%}")
    if "confidence_level" in display_df:
        display_df["confidence_level"] = display_df["confidence_level"].map(lambda x: f"{float(x):.2%}")
    return display_df


def formatted_attribution_table(df):
    display_df = df.copy()
    display_df["pnl"] = display_df["pnl"].map(format_money)
    return display_df.rename(columns={"component": "Component", "pnl": "P&L"})


def linearization_validation_table(unhedged_attr, hedged_attr, unhedged_value, hedged_value):
    rows = []
    for label, attr, value in [
        ("Unhedged", unhedged_attr, unhedged_value),
        ("Hedged", hedged_attr, hedged_value),
    ]:
        total_linearized = float(attr.loc[attr["component"] == "total_linearized", "pnl"].iloc[0])
        full_reprice = float(attr.loc[attr["component"] == "full_reprice", "pnl"].iloc[0])
        error = float(attr.loc[attr["component"] == "linearization_error", "pnl"].iloc[0])
        error_bp = np.nan if value == 0 else abs(error) / abs(value) * 10_000
        rows.append({
            "Portfolio": label,
            "Linearized P&L": total_linearized,
            "Full Reprice P&L": full_reprice,
            "Linearization Error": error,
            "Error (bp of value)": error_bp,
        })
    return pd.DataFrame(rows)


def line_item_display_table(line_item_df):
    display_df = line_item_df.copy()
    display_df = display_df[
        [
            "bond",
            "face_value",
            "positions",
            "coupon",
            "maturity",
            "maturity_date",
            "curve_rate",
            "clean_value",
            "accrued_interest",
            "dirty_value",
            "price_per_100",
            "dv01",
            "effective_duration",
            "convexity",
            "largest_krd_bucket",
            "largest_krd_share",
        ]
    ]

    display_df["face_value"] = display_df["face_value"].map(lambda x: f"${x:,.0f}")
    display_df["positions"] = display_df["positions"].map(lambda x: f"{x:,.2f}")
    display_df["coupon"] = display_df["coupon"].map(lambda x: f"{x:.2%}")
    display_df["maturity"] = display_df["maturity"].map(lambda x: f"{x:.2f}Y")
    display_df["maturity_date"] = pd.to_datetime(display_df["maturity_date"]).dt.strftime("%Y-%m-%d")
    display_df["curve_rate"] = display_df["curve_rate"].map(lambda x: f"{x:.2%}")
    display_df["clean_value"] = display_df["clean_value"].map(lambda x: f"${x:,.0f}")
    display_df["accrued_interest"] = display_df["accrued_interest"].map(lambda x: f"${x:,.0f}")
    display_df["dirty_value"] = display_df["dirty_value"].map(lambda x: f"${x:,.0f}")
    display_df["price_per_100"] = display_df["price_per_100"].map(lambda x: f"{x:.2f}")
    display_df["dv01"] = display_df["dv01"].map(lambda x: f"${x:,.2f}")
    display_df["effective_duration"] = display_df["effective_duration"].map(lambda x: f"{x:.3f}")
    display_df["convexity"] = display_df["convexity"].map(lambda x: f"{x:.3f}")
    display_df["largest_krd_bucket"] = display_df["largest_krd_bucket"].map(format_tenor)
    display_df["largest_krd_share"] = display_df["largest_krd_share"].map(lambda x: f"{x:.2%}")

    return display_df.rename(
        columns={
            "bond": "Bond",
            "face_value": "Face Value",
            "positions": "Position",
            "coupon": "Coupon",
            "maturity": "Mat (Y)",
            "maturity_date": "Maturity Date",
            "curve_rate": "Curve Rate",
            "clean_value": "Clean Value",
            "accrued_interest": "Accrued Interest",
            "dirty_value": "Dirty Value",
            "price_per_100": "Price / $100",
            "dv01": "DV01",
            "effective_duration": "Eff. Duration",
            "convexity": "Convexity",
            "largest_krd_bucket": "Largest KRD Bucket",
            "largest_krd_share": "Largest KRD Share",
        }
    )


def choose_hedge_universe(name, settlement_date, day_count):
    if name == "Front-end: 6M + 2Y":
        return front_end_hedge_instruments(settlement_date, day_count), np.array([0.20, 0.25]), 2
    if name == "Middle-end: 3Y + 7Y + 20Y":
        return mild_mismatch_hedge_instruments(settlement_date, day_count), np.array([0.35, 0.45, 0.65]), 3
    return long_duration_hedge_instruments(settlement_date, day_count), np.array([0.25, 0.35, 0.50, 0.75]), 3


def get_fred_api_key():
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        return os.getenv("FRED_API_KEY", "")


st.title("RatesFactor")
st.caption("Treasury curve risk, PCA hedging, scenarios, VaR, and attribution")

with st.sidebar:
    st.header("Inputs")

    api_key = get_fred_api_key()
    if api_key:
        st.caption("FRED API key loaded from local secrets/config.")

    source_type = st.selectbox(
        "Portfolio source",
        ["toy", "standard", "tlt"],
        format_func=lambda x: {
            "toy": "Toy portfolio",
            "standard": "Standard holdings template",
            "tlt": "iShares TLT holdings CSV",
        }[x],
    )

    uploaded_file = None
    if source_type == "standard":
        uploaded_file = st.file_uploader("Upload standard holdings .xlsx", type=["xlsx"])
    elif source_type == "tlt":
        uploaded_file = st.file_uploader("Upload iShares TLT holdings .csv", type=["csv"])

    holdings_as_of_date = st.date_input("Holdings / risk as-of date", value=date(2026, 7, 12))
    target_notional = st.number_input("Target portfolio notional", min_value=100_000, value=10_000_000, step=100_000)

    hedge_universe_name = st.selectbox(
        "Hedge universe",
        [
            "Front-end: 6M + 2Y",
            "Middle-end: 3Y + 7Y + 20Y",
            "Long-duration: 2Y + 5Y + 10Y + 30Y",
            "Custom hedge template",
        ],
    )
    hedge_file = None
    if hedge_universe_name == "Custom hedge template":
        hedge_file = st.file_uploader("Upload custom hedge instruments .xlsx", type=["xlsx"])

    with st.expander("Advanced settings"):
        day_count = st.selectbox("Day count convention", ["ACT/ACT", "ACT/365.25", "ACT/365", "ACT/360"])
        history_years = st.number_input("Curve history window (years)", min_value=1, max_value=15, value=5, step=1)
        alpha = st.selectbox("VaR significance level", [0.05, 0.01], format_func=lambda x: f"{x:.0%} tail / {(1-x):.0%} VaR")
        lookback = st.number_input("PCA / VaR lookback days", min_value=60, max_value=1000, value=252, step=21)
        max_backtest_days = st.number_input(
            "Rolling hedge backtest days",
            min_value=30,
            max_value=1000,
            value=750,
            step=30,
            help="Number of daily P&L observations replayed in the rolling hedge backtest.",
        )
        pca_stride = st.number_input(
            "Rolling PCA rebalance stride",
            min_value=1,
            max_value=60,
            value=10,
            step=1,
            help="Recompute PCA hedge weights every N backtest days and hold the previous hedge between rebalances.",
        )
        var_backtest_lookback = st.number_input(
            "VaR backtest lookback days",
            min_value=20,
            max_value=750,
            value=300,
            step=25,
            help="Rolling P&L window used only for the Kupiec VaR breach test. Keep it below the hedge backtest days to show results.",
        )
        ridge_lambda = st.number_input(
            "Hedge ridge regularization",
            min_value=0.0,
            max_value=10.0,
            value=0.01,
            step=0.001,
            format="%.6f",
            help="Higher values reduce unstable hedge weights but allow more residual factor exposure.",
        )
        st.caption("Ridge λ trades off factor-neutrality against hedge stability: higher values shrink hedge weights but leave more residual exposure.")
        curve_fit_method = st.selectbox("Displayed curve fit", ["NSS", "Cubic spline"])
        curve_source = st.selectbox(
            "Pricing curve source",
            ["fred", "demo_bootstrap", "uploaded_bootstrap"],
            format_func=lambda x: {
                "fred": "FRED fitted Treasury curve",
                "demo_bootstrap": "Use filled demo bootstrap template",
                "uploaded_bootstrap": "Upload curve construction universe",
            }[x],
        )
        curve_universe_file = None
        if curve_source == "uploaded_bootstrap":
            curve_universe_file = st.file_uploader("Upload curve construction universe .xlsx", type=["xlsx"])
        st.caption(
            "Bootstrapped curve modes use dirty prices from the curve construction universe for the latest pricing curve; "
            "FRED history is still used for PCA, historical shocks, and backtests."
        )

    with st.expander("Download input templates"):
        st.caption("Template files are prepared only when requested to keep app startup fast.")
        prepare_template = st.checkbox("Prepare a template download")
        if prepare_template:
            template_name = st.selectbox(
                "Template",
                [
                    "Standard holdings",
                    "TLT-style holdings",
                    "Custom hedge instruments",
                    "Curve construction universe",
                ],
            )
            template_data, template_file_name, template_mime = cached_template_download(template_name)
            st.download_button(
                "Download selected template",
                data=template_data,
                file_name=template_file_name,
                mime=template_mime,
            )

run_disabled = (
    (not api_key)
    or (source_type != "toy" and uploaded_file is None)
    or (hedge_universe_name == "Custom hedge template" and hedge_file is None)
    or (curve_source == "uploaded_bootstrap" and curve_universe_file is None)
)
run = st.sidebar.button("Run RatesFactor", type="primary", disabled=run_disabled)
if "run_state" in st.session_state and not run:
    st.sidebar.caption("Showing the last completed run. Click Run RatesFactor to refresh after changing inputs.")

if (not run) and ("run_state" not in st.session_state):
    st.info("Choose inputs in the sidebar and click Run RatesFactor.")
    if not api_key:
        st.warning("A FRED API key is required to fetch Treasury curve history.")
    if source_type != "toy" and uploaded_file is None:
        st.warning("Upload a holdings file or select the toy portfolio.")
    if hedge_universe_name == "Custom hedge template" and hedge_file is None:
        st.warning("Upload a custom hedge instruments file or choose a built-in hedge universe.")
    if curve_source == "uploaded_bootstrap" and curve_universe_file is None:
        st.warning("Upload a curve construction universe file or choose the FRED/demo curve option.")
    st.stop()


if run:
    holdings_as_of_ts = pd.Timestamp(holdings_as_of_date)
    start_date = holdings_as_of_ts - pd.DateOffset(years=history_years)
    end_date = holdings_as_of_ts

    with st.spinner("Loading portfolio and Treasury curve history..."):
        _, portfolio = load_portfolio(
            source_type,
            path_or_buffer=uploaded_file,
            as_of_date=holdings_as_of_ts,
            target_notional=target_notional,
            day_count=day_count,
        )
        rates_pct = cached_fetch_rates(api_key, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        rates_data = RatesData(rates_pct)

    curve_as_of_date = pd.Timestamp(rates_data.latest_date)
    base_curve = rates_data.get_curve(curve_as_of_date, units="decimal")
    zero_curve = None
    curve_universe = None
    pricing_curve_label = "FRED fitted Treasury curve"

    with st.spinner("Preparing pricing curve..."):
        if curve_source == "demo_bootstrap":
            curve_universe = curve_construction_universe_template()
            zero_curve = bootstrap_zero_curve(curve_universe)
            pricing_curve_label = "Demo bootstrapped zero curve"
        elif curve_source == "uploaded_bootstrap":
            curve_universe = load_curve_universe(curve_universe_file)
            zero_curve = bootstrap_zero_curve(curve_universe)
            pricing_curve_label = "Uploaded bootstrapped zero curve"

        if zero_curve is not None:
            base_curve = zero_curve_to_rate_curve(zero_curve, rates_data.tenors)
            base_curve.name = curve_as_of_date
            rates_data.rates_decimal.loc[curve_as_of_date, rates_data.tenors] = base_curve.to_numpy(dtype=float)
            rates_data.rates_pct.loc[curve_as_of_date, rates_data.tenors] = base_curve.to_numpy(dtype=float) * 100.0

    if hedge_universe_name == "Custom hedge template":
        hedge_instruments, cost_bps = load_custom_hedge_instruments(
            hedge_file,
            settlement_date=holdings_as_of_ts,
            day_count=day_count,
        )
        n_components = min(3, len(hedge_instruments))
    else:
        hedge_instruments, cost_bps, n_components = choose_hedge_universe(
            hedge_universe_name,
            holdings_as_of_ts,
            day_count,
        )
    n_components = min(n_components, len(hedge_instruments), 3)

    with st.spinner("Running PCA, hedge construction, and risk analytics..."):
        latest_pca = fit_pca(rates_data.daily_changes_bp.iloc[-int(lookback):], n_components=3)
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

    with st.spinner("Running hedge backtest and VaR validation inputs..."):
        rolling_pca = fit_rolling_pca_for_rebalance(
            rates_data.daily_changes_bp,
            lookback=int(lookback),
            n_components=3,
            max_windows=int(max_backtest_days) + 1,
            stride=int(pca_stride),
        )
        backtest_results = run_pca_hedge_backtest(
            portfolio,
            hedge_instruments,
            rates_data,
            rolling_pca,
            cost_bps,
            lookback=int(lookback),
            n_components=n_components,
            ridge_lambda=ridge_lambda,
            max_backtest_days=int(max_backtest_days),
        )
        results_df, summary, hedge_labels, hedge_weight_cols, _ = summarize_backtest_results(
            backtest_results,
            hedge_instruments,
        )

    st.session_state.run_state = {
        "portfolio": portfolio,
        "rates_data": rates_data,
        "curve_as_of_date": curve_as_of_date,
        "base_curve": base_curve,
        "zero_curve": zero_curve,
        "curve_universe": curve_universe,
        "pricing_curve_label": pricing_curve_label,
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
        "lookback": int(lookback),
        "var_backtest_lookback": int(var_backtest_lookback),
        "pca_stride": int(pca_stride),
        "alpha": alpha,
        "max_backtest_days": int(max_backtest_days),
        "curve_fit_method": curve_fit_method,
    }

state = st.session_state.run_state
portfolio = state["portfolio"]
rates_data = state["rates_data"]
curve_as_of_date = state["curve_as_of_date"]
base_curve = state["base_curve"]
zero_curve = state["zero_curve"]
curve_universe = state["curve_universe"]
pricing_curve_label = state["pricing_curve_label"]
hedge_instruments = state["hedge_instruments"]
n_components = state["n_components"]
latest_pca = state["latest_pca"]
static_pca = state["static_pca"]
hedge_weights = state["hedge_weights"]
hedged_portfolio = state["hedged_portfolio"]
hedge_diag = state["hedge_diag"]
results_df = state["results_df"]
summary = state["summary"]
hedge_labels = state["hedge_labels"]
hedge_weight_cols = state["hedge_weight_cols"]
lookback = state["lookback"]
var_backtest_lookback = state["var_backtest_lookback"]
pca_stride = state["pca_stride"]
alpha = state["alpha"]
max_backtest_days = state["max_backtest_days"]
curve_fit_method = state["curve_fit_method"]

st.subheader("Run Context")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Curve as-of date", str(curve_as_of_date.date()))
col2.metric("Portfolio rows", len(portfolio))
col3.metric("Target notional", format_money(portfolio["face_value"].sum()))
col4.metric("Hedge instruments", len(hedge_instruments))
st.caption(f"Pricing curve source: {pricing_curve_label}")
st.caption(
    f"Rolling PCA hedge backtest observations: latest {int(max_backtest_days)} days; "
    f"PCA hedge weights are rebalanced every {int(pca_stride)} day(s)."
)

if hedge_diag["severity"] == "Warning":
    st.warning(
        "Hedge universe warning: "
        + " ".join(hedge_diag["warnings"])
    )

tabs = st.tabs([
    "Portfolio",
    "Curve & PCA",
    "Risk",
    "Hedge Backtest",
    "Scenarios",
    "VaR / ES",
    "Attribution",
])

with tabs[0]:
    st.subheader("Active Portfolio")
    portfolio_stats = portfolio_summary_stats(portfolio)
    portfolio_values = portfolio_valuation(portfolio, base_curve, settlement_date=curve_as_of_date)
    line_item_df = line_item_bond_analytics(portfolio, rates_data, curve_as_of_date)
    c1, c2, c3 = st.columns(3)
    metric_with_help(c1, "Weighted avg maturity", f"{portfolio_stats['weighted_avg_maturity']:.2f}Y")
    metric_with_help(c2, "Weighted avg coupon", f"{portfolio_stats['weighted_avg_coupon']:.2%}")
    metric_with_help(c3, "Market / dirty value", format_money(portfolio_values["dirty_value"]))
    st.caption(
        "Pricing caveat: FRED mode discounts off fitted CMT yields, not a fully bootstrapped market zero curve. "
        "Bootstrapped modes are available as a demo/user-uploaded curve-construction workflow. A fitted/par-yield proxy "
        "vs bootstrapped curve comparison is documented on GitHub in the README and Assumptions & Limitations."
    )
    st.dataframe(
        line_item_display_table(line_item_df),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Raw portfolio inputs"):
        st.dataframe(portfolio, use_container_width=True, hide_index=True)
    st.subheader("Hedge Instruments")
    hedge_display = hedge_instruments.copy()
    hedge_display["hedge_weight"] = hedge_weights
    st.dataframe(hedge_display, use_container_width=True, hide_index=True)
    st.subheader("Hedge Suitability Diagnostics")
    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Suitability", hedge_diag["severity"])
    metric_with_help(c2, "Condition number", f"{hedge_diag['condition_number']:.1f}")
    c3.metric("Hedge / portfolio notional", f"{hedge_diag['hedge_notional_ratio']:.2f}x")
    metric_with_help(c4, "Top portfolio KRD bucket", format_tenor(hedge_diag["top_krd_bucket"]), key="Largest KRD bucket")
    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Portfolio WAM", f"{hedge_diag['portfolio_wam']:.2f}Y", key="Weighted avg maturity")
    metric_with_help(c2, "Hedge universe WAM", f"{hedge_diag['hedge_wam']:.2f}Y", key="Weighted avg maturity")
    c3.metric("Max hedge maturity", f"{hedge_diag['max_hedge_maturity']:.2f}Y")
    metric_with_help(c4, "Residual factor norm", f"{hedge_diag['factor_residual_norm']:.2f}")
    st.caption(
        "Large hedge notionals, negative hedged market value, or sign-flipped hedged P&L can occur when the selected "
        "hedge universe does not cover the portfolio's key-rate exposure; use the suitability diagnostics before "
        "interpreting hedge results."
    )

with tabs[1]:
    curve_pct = rates_data.get_curve(curve_as_of_date, units="pct")
    tenors = rates_data.tenors
    yields = curve_pct.to_numpy(dtype=float)
    tenor_grid = np.linspace(tenors.min(), tenors.max(), 300)
    curve_diagnostics = {
        "Latest rolling explained variance": latest_pca.explained_variance_ratio_.round(4).tolist(),
        "Static explained variance": static_pca.explained_variance_ratio_.round(4).tolist(),
        "Displayed curve fit": curve_fit_method,
    }

    if curve_fit_method == "NSS":
        beta_best, tau_best, fit_yields_grid = None, None, None
        beta_best, tau_best, nss_rmse = fit_nss_grid(tenors, yields, grid_size=40)
        fit_yields_grid = nss_yield(tenor_grid, beta_best, tau_best[0], tau_best[1])
        fit_label = "NSS fitted curve"
        curve_diagnostics["NSS RMSE pct"] = float(nss_rmse)
    else:
        fit_yields_grid = cubic_spline_yield(tenors, yields, tenor_grid)
        fit_label = "Natural cubic spline"

    fitted_fwd = compute_fwd(tenor_grid, fit_yields_grid)
    curve_diagnostics["Fitted forward turning points"] = int(count_turning_points(fitted_fwd))

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        curve_figure(
            rates_data,
            curve_as_of_date,
            fitted_tenors=tenor_grid,
            fitted_yields=fit_yields_grid,
            fit_label=fit_label,
        ),
        use_container_width=True,
    )
    c2.plotly_chart(pca_loadings_figure(latest_pca, rates_data.tenors), use_container_width=True)
    st.caption(
        f"The displayed PCA loadings are from the latest rolling PCA fit using the most recent {int(lookback)} "
        "daily curve-change observations."
    )
    st.caption(
        "Rolling PCA factors are aligned by cosine similarity for economic continuity, so displayed PC labels may not "
        "strictly follow descending explained variance after alignment."
    )
    st.caption(
        "Static explained variance is fit once on the full rate-history sample. Latest rolling explained variance is "
        "fit on the most recent PCA window and is what drives the current hedge, so it can differ when the recent "
        "rate regime differs from the full sample."
    )
    st.write("PCA explained variance")
    explained_variance_df = pd.DataFrame({
        "Component": [f"PC{i + 1}" for i in range(3)],
        "Latest rolling explained variance": latest_pca.explained_variance_ratio_[:3] * 100,
        "Static explained variance": static_pca.explained_variance_ratio_[:3] * 100,
    })
    st.dataframe(
        explained_variance_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Latest rolling explained variance": st.column_config.NumberColumn(
                "Latest rolling explained variance",
                format="%.2f%%",
                help="Explained variance from the latest rolling PCA window used for current hedge construction.",
            ),
            "Static explained variance": st.column_config.NumberColumn(
                "Static explained variance",
                format="%.2f%%",
                help="Explained variance from a PCA fit once on the full historical curve-change sample.",
            ),
        },
    )
    st.write("Curve fit diagnostics")
    fit_diagnostics = [
        {"Metric": "Displayed curve fit", "Value": curve_fit_method},
        {"Metric": "Fitted forward turning points", "Value": curve_diagnostics["Fitted forward turning points"]},
    ]
    if "NSS RMSE pct" in curve_diagnostics:
        fit_diagnostics.append({"Metric": "NSS RMSE (%)", "Value": f"{curve_diagnostics['NSS RMSE pct']:.4f}"})
    st.dataframe(pd.DataFrame(fit_diagnostics), use_container_width=True, hide_index=True)
    if zero_curve is not None:
        st.subheader("Bootstrapped Zero Curve")
        st.caption(
            "Discount factors are bootstrapped from the curve construction universe and interpolated in log discount-factor space."
        )
        zero_display = zero_curve.copy()
        zero_display["zero_rate_pct"] = zero_display["zero_rate_cc"] * 100.0
        st.dataframe(
            zero_display[
                [
                    "instrument_id",
                    "instrument_type",
                    "time_to_maturity",
                    "discount_factor",
                    "zero_rate_pct",
                    "dirty_price",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Curve construction universe"):
            st.dataframe(curve_universe, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("DV01 and Key-Rate DV01")
    unhedged_values = portfolio_valuation(portfolio, base_curve, settlement_date=curve_as_of_date)
    hedged_values = portfolio_valuation(hedged_portfolio, base_curve, settlement_date=curve_as_of_date)
    unhedged_value = unhedged_values["dirty_value"]
    hedged_value = hedged_values["dirty_value"]
    unhedged_dv01 = calc_dv01(portfolio, rates_data, curve_as_of_date)
    hedged_dv01 = calc_dv01(hedged_portfolio, rates_data, curve_as_of_date)
    unhedged_duration = calc_effective_duration(portfolio, rates_data, curve_as_of_date)
    hedged_duration = calc_effective_duration(hedged_portfolio, rates_data, curve_as_of_date)
    unhedged_convexity = calc_convexity(portfolio, rates_data, curve_as_of_date)
    hedged_convexity = calc_convexity(hedged_portfolio, rates_data, curve_as_of_date)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unhedged value", format_money(unhedged_value))
    c2.metric("Hedged value", format_money(hedged_value))
    metric_with_help(c3, "Unhedged parallel DV01", format_money(unhedged_dv01), key="DV01")
    metric_with_help(c4, "Hedged parallel DV01", format_money(hedged_dv01), key="DV01")

    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Unhedged eff. duration", f"{unhedged_duration:.2f}", key="Effective duration")
    metric_with_help(c2, "Hedged eff. duration", f"{hedged_duration:.2f}", key="Effective duration")
    metric_with_help(c3, "Unhedged convexity", f"{unhedged_convexity:.2f}", key="Convexity")
    metric_with_help(c4, "Hedged convexity", f"{hedged_convexity:.2f}", key="Convexity")

    unhedged_ladder = make_delta_ladder(portfolio, rates_data, curve_as_of_date)
    hedged_ladder = make_delta_ladder(hedged_portfolio, rates_data, curve_as_of_date)
    unhedged_concentration = key_rate_risk_concentration(unhedged_ladder, rates_data.tenors)
    hedged_concentration = key_rate_risk_concentration(hedged_ladder, rates_data.tenors)

    st.subheader("Risk Concentration")
    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Unhedged top KRD bucket", format_tenor(unhedged_concentration["largest_bucket"]), key="Largest KRD bucket")
    metric_with_help(c2, "Unhedged top KRD share", f"{unhedged_concentration['share_total_abs_krd']:.2%}", key="Largest KRD share")
    metric_with_help(c3, "Hedged top KRD bucket", format_tenor(hedged_concentration["largest_bucket"]), key="Largest KRD bucket")
    metric_with_help(c4, "Hedged top KRD share", f"{hedged_concentration['share_total_abs_krd']:.2%}", key="Largest KRD share")

    ladder_df = pd.DataFrame({
        "Tenor": [format_tenor(x) for x in rates_data.tenors],
        "Unhedged Key-Rate DV01": unhedged_ladder,
        "Hedged Key-Rate DV01": hedged_ladder,
    })
    st.dataframe(
        ladder_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Unhedged Key-Rate DV01": st.column_config.NumberColumn("Unhedged Key-Rate DV01", format="$%.0f"),
            "Hedged Key-Rate DV01": st.column_config.NumberColumn("Hedged Key-Rate DV01", format="$%.0f"),
        },
    )
    st.bar_chart(ladder_df.set_index("Tenor"))
    st.write("Clean / dirty valuation")
    st.dataframe(
        pd.DataFrame(
            [
                {"portfolio": "unhedged", **unhedged_values},
                {"portfolio": "hedged", **hedged_values},
            ]
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "portfolio": st.column_config.TextColumn("Portfolio"),
            "clean_value": st.column_config.NumberColumn("Clean Value", format="$%.0f"),
            "dirty_value": st.column_config.NumberColumn("Dirty Value", format="$%.0f"),
            "accrued_interest": st.column_config.NumberColumn("Accrued Interest", format="$%.0f"),
        },
    )

with tabs[3]:
    st.subheader("Rolling PCA Hedge Backtest")
    st.info(
        "How to read this: strong hedge effectiveness is most believable when the hedge universe spans the portfolio's "
        "key-rate exposure and transaction costs remain small. Very high effectiveness can be a toy/demo artifact if "
        "the hedge instruments overlap closely with the portfolio tenors; weak or sign-flipped results usually point "
        "to hedge-universe mismatch, unstable weights, or residual factor exposure."
    )
    c1, c2, c3, c4 = st.columns(4)
    metric_with_help(c1, "Avg abs P&L reduction", f"{summary['gross_avg_abs_pnl_reduction_pct']:.2%}", key="Gross P&L reduction")
    metric_with_help(c2, "Net vol reduction", f"{summary['net_pnl_vol_reduction_pct']:.2%}")
    metric_with_help(c3, "Net hit rate", f"{summary['net_hit_rate']:.2%}")
    c4.metric("Total transaction cost", format_money(summary["total_transaction_cost"]))
    st.caption(
        "The backtest replays daily P&L through the selected rolling hedge window. PCA hedge weights are recalculated "
        f"every {int(pca_stride)} day(s) and held between rebalances. If the hedge universe is maturity-mismatched, "
        "check the Portfolio tab's hedge suitability diagnostics before interpreting results."
    )
    st.caption(
        "Transaction costs use a bps-on-notional-change assumption: daily cost = |change in hedge notional| x cost bps / 10,000. "
        "This is a configurable modeling assumption, not a universal execution-cost estimate."
    )
    st.plotly_chart(backtest_figure(results_df, hedge_weight_cols), use_container_width=True)
    st.caption(
        f"PCA hedge weights are recalculated every {int(pca_stride)} day(s) and held constant between rebalances, "
        "so the hedge-weight lines are expected to look stepwise. Large jumps usually point to a changing PCA regime, "
        "an ill-conditioned hedge universe, or a ridge regularization setting that is too low for the selected instruments."
    )
    st.plotly_chart(transaction_cost_figure(results_df), use_container_width=True)
    st.write("Rolling Hedge Backtest - Summary")
    st.dataframe(formatted_summary_table(summary), use_container_width=True, hide_index=True)
    with st.expander("Rolling Hedge Backtest - Daily Detail"):
        detail_df = results_df.reset_index().rename(columns={"index": "Date"})
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                "unhedged_pnl": st.column_config.NumberColumn("Unhedged P&L", format="$%.0f"),
                "hedged_pnl": st.column_config.NumberColumn("Gross Hedged P&L", format="$%.0f"),
                "net_hedged_pnl": st.column_config.NumberColumn("Net Hedged P&L", format="$%.0f"),
                "transaction_cost": st.column_config.NumberColumn("Transaction Cost", format="$%.2f"),
                "cum_transaction_cost": st.column_config.NumberColumn("Cumulative Transaction Cost", format="$%.0f"),
            },
        )

with tabs[4]:
    st.subheader("Scenario Analysis")
    irrbb_scenarios = generate_irrbb_usd_scenarios(rates_data.tenors)
    all_scenarios = {
        **PRESET_SCENARIOS_BP,
        **irrbb_scenarios,
    }
    st.caption("IRRBB USD scenarios use the Basel prescribed parallel, short-rate, steepener, flattener, and short-rate-down/up shock shapes.")
    scenario_name = st.selectbox("Single scenario", list(all_scenarios.keys()))
    unhedged_pnl, hedged_pnl, pnl_reduction = run_scenario_analysis(
        portfolio,
        hedge_instruments,
        hedge_weights,
        base_curve,
        all_scenarios[scenario_name],
        settlement_date=curve_as_of_date,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Unhedged P&L", format_money(unhedged_pnl))
    c2.metric("Hedged P&L", format_money(hedged_pnl))
    c3.metric("Abs P&L reduction", format_money(pnl_reduction))

    multi_scenarios = run_multi_scenario_analysis(
        portfolio,
        hedge_instruments,
        hedge_weights,
        base_curve,
        all_scenarios,
        settlement_date=curve_as_of_date,
    )
    st.dataframe(formatted_scenario_table(multi_scenarios), use_container_width=True, hide_index=True)
    st.bar_chart(multi_scenarios.set_index("scenario_name")[["unhedged_pnl", "hedged_pnl"]])

with tabs[5]:
    st.subheader("VaR / Expected Shortfall")
    st.info(
        "How to read this: Historical VaR replays realized curve shocks, while parametric PCA VaR assumes normal factor "
        "moves. A reasonable VaR backtest should have breach frequency close to the selected tail rate; a low Kupiec "
        "p-value suggests the breach rate is inconsistent with the target VaR level."
    )
    hist_var = compute_historical_var(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        alpha=alpha,
        lookback=int(lookback),
    )
    param_var = compute_parametric_var(
        portfolio,
        hedge_instruments,
        hedge_weights,
        rates_data,
        latest_pca,
        alpha=alpha,
        lookback=int(lookback),
        n_components=n_components,
    )
    st.write("Historical simulation VaR / ES")
    st.dataframe(formatted_var_table(hist_var), use_container_width=True, hide_index=True)
    st.write("Historical VaR Backtest")
    var_backtest = backtest_historical_var_table(results_df, alpha=alpha, lookback=int(var_backtest_lookback))
    if var_backtest.empty or var_backtest["days"].sum() == 0:
        st.info("Not enough backtest observations after the selected VaR backtest lookback window.")
    else:
        st.caption(
            "Rolling historical VaR is computed from prior P&L observations and compared with the next realized daily loss. "
            "Kupiec p-value tests unconditional coverage; low values suggest breach frequency differs from the target tail rate. "
            f"The VaR/ES model lookback is {int(lookback)} days; this Kupiec panel uses a shorter "
            f"{int(var_backtest_lookback)}-day rolling P&L window."
        )
        st.dataframe(
            var_backtest,
            use_container_width=True,
            hide_index=True,
            column_config={
                "expected_breaches": st.column_config.NumberColumn("expected_breaches", format="%.2f"),
                "breach_rate": st.column_config.NumberColumn("breach_rate", format="%.2%"),
                "expected_breach_rate": st.column_config.NumberColumn("expected_breach_rate", format="%.2%"),
                "latest_var": st.column_config.NumberColumn("latest_var", format="$%.0f"),
                "kupiec_lr": st.column_config.NumberColumn("Kupiec LR", format="%.3f"),
                "kupiec_p_value": st.column_config.NumberColumn("Kupiec p-value", format="%.3f"),
                "kupiec_pass_5pct": st.column_config.CheckboxColumn("Kupiec pass 5%"),
            },
        )
    st.write("Parametric PCA VaR / ES")
    st.dataframe(formatted_parametric_var_table(param_var), use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Latest One-Day PCA P&L Attribution")
    date_t = pd.Timestamp(rates_data.dates[-2])
    date_t1 = pd.Timestamp(rates_data.dates[-1])
    pca_for_attribution = latest_pca
    unhedged_attr = pca_pnl_attribution(portfolio, rates_data, date_t, date_t1, pca_for_attribution, n_components=3)
    hedged_attr = pca_pnl_attribution(hedged_portfolio, rates_data, date_t, date_t1, pca_for_attribution, n_components=3)
    unhedged_value_latest = portfolio_value(portfolio, rates_data.get_curve(date_t, "decimal"), settlement_date=date_t)
    hedged_value_latest = portfolio_value(hedged_portfolio, rates_data.get_curve(date_t, "decimal"), settlement_date=date_t)
    validation_df = linearization_validation_table(
        unhedged_attr,
        hedged_attr,
        unhedged_value_latest,
        hedged_value_latest,
    )
    st.caption(
        "Validation artifact: PCA attribution is a linearized P&L approximation. The table below compares the "
        "linearized attribution sum with full bond repricing for the latest one-day move."
    )
    v1, v2, v3 = st.columns(3)
    avg_error_bp = validation_df["Error (bp of value)"].mean()
    max_abs_error = validation_df["Linearization Error"].abs().max()
    v1.metric("Avg linearization error", f"{avg_error_bp:.2f} bp", help="Average absolute attribution error as bp of portfolio value.")
    v2.metric("Max error ($)", format_money(max_abs_error), help="Largest absolute difference between linearized P&L and full-reprice P&L.")
    v3.metric("Attribution date", f"{date_t.date()} to {date_t1.date()}")
    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Linearized P&L": st.column_config.NumberColumn("Linearized P&L", format="$%.0f"),
            "Full Reprice P&L": st.column_config.NumberColumn("Full Reprice P&L", format="$%.0f"),
            "Linearization Error": st.column_config.NumberColumn("Linearization Error", format="$%.0f"),
            "Error (bp of value)": st.column_config.NumberColumn("Error (bp of value)", format="%.2f"),
        },
    )
    st.bar_chart(validation_df.set_index("Portfolio")[["Linearized P&L", "Full Reprice P&L"]])
    c1, c2 = st.columns(2)
    c1.write("Unhedged")
    c1.dataframe(formatted_attribution_table(unhedged_attr), use_container_width=True, hide_index=True)
    c2.write("Hedged")
    c2.dataframe(formatted_attribution_table(hedged_attr), use_container_width=True, hide_index=True)
