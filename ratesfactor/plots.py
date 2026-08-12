import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def curve_figure(rates_data, date, fitted_tenors=None, fitted_yields=None, fit_label=None):
    curve = rates_data.get_curve(date, units="pct")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.index, y=curve.values, mode="markers", name="Observed Treasury points"))
    if fitted_tenors is not None and fitted_yields is not None:
        fig.add_trace(go.Scatter(x=fitted_tenors, y=fitted_yields, mode="lines", name=fit_label or "Fitted curve"))
    fig.update_layout(title=f"Treasury Curve - {pd.Timestamp(date).date()}", xaxis_title="Maturity", yaxis_title="Yield (%)")
    return fig


def pca_loadings_figure(pca, tenors):
    fig = go.Figure()
    for i in range(min(3, len(pca.components_))):
        pct = pca.explained_variance_ratio_[i] * 100
        fig.add_trace(go.Scatter(x=tenors, y=pca.components_[i], mode="lines+markers", name=f"PC{i + 1} ({pct:.1f}%)"))
    fig.update_layout(title="PCA Loadings", xaxis_title="Maturity", yaxis_title="Loading")
    return fig


def backtest_figure(results_df, hedge_weight_cols):
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Cumulative P&L", "Daily P&L", "PCA Hedge Weights", "Daily P&L Distribution"),
    )
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["cum_unhedged_pnl"], name="Unhedged"), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["cum_gross_hedged_pnl"], name="Gross Hedged"), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["cum_net_hedged_pnl"], name="Net Hedged"), row=1, col=1)

    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["unhedged_pnl"], name="Unhedged Daily", opacity=0.65), row=1, col=2)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["hedged_pnl"], name="Gross Hedged Daily", opacity=0.65), row=1, col=2)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["net_hedged_pnl"], name="Net Hedged Daily", opacity=0.65), row=1, col=2)

    for col in hedge_weight_cols:
        fig.add_trace(go.Scatter(x=results_df.index, y=results_df[col], name=col), row=2, col=1)

    fig.add_trace(go.Histogram(x=results_df["unhedged_pnl"], name="Unhedged Hist", opacity=0.55), row=2, col=2)
    fig.add_trace(go.Histogram(x=results_df["hedged_pnl"], name="Gross Hedged Hist", opacity=0.55), row=2, col=2)
    fig.add_trace(go.Histogram(x=results_df["net_hedged_pnl"], name="Net Hedged Hist", opacity=0.55), row=2, col=2)
    fig.update_layout(height=780, barmode="overlay", title="Hedge Backtest")
    return fig


def transaction_cost_figure(results_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df["cum_transaction_cost"], mode="lines", name="Cumulative Transaction Cost"))
    fig.update_layout(title="Cumulative Transaction Cost", xaxis_title="Date", yaxis_title="Cost ($)")
    return fig
