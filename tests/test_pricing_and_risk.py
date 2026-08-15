import pandas as pd
import pytest

from ratesfactor.bootstrapper import bootstrap_zero_from_par
from ratesfactor.data import RatesData
from ratesfactor.portfolio import toy_portfolio
from ratesfactor.pricing import bond_value, value_bond_row
from ratesfactor.risk import calc_dv01
from ratesfactor.validation import demo_pricing_error_summary


def flat_curve(rate):
    curve = pd.Series([rate, rate, rate, rate], index=[0.5, 1.0, 2.0, 5.0])
    curve.name = pd.Timestamp("2026-01-01")
    return curve


def test_bond_prices_near_par_when_coupon_matches_flat_curve():
    row = pd.Series({
        "bond": "2Y par note",
        "face_value": 100.0,
        "coupon": 0.05,
        "maturity": 2.0,
        "maturity_date": pd.Timestamp("2028-01-01"),
        "frequency": 2,
        "positions": 1.0,
        "day_count": "ACT/365",
    })

    values = value_bond_row(row, flat_curve(0.05), settlement_date="2026-01-01")

    assert abs(values["dirty_value"] - 100.0) < 0.01
    assert values["accrued_interest"] == 0.0


def test_accrued_interest_between_coupon_dates():
    values = bond_value(
        face_value=100.0,
        coupon=0.06,
        settlement_date=pd.Timestamp("2026-04-01"),
        maturity_date=pd.Timestamp("2027-01-01"),
        curve_fn=lambda _: 0.05,
        frequency=2,
        day_count="ACT/365",
    )

    assert 1.45 < values["accrued_interest"] < 1.55
    assert values["clean_value"] < values["dirty_value"]


def test_dv01_is_positive_for_long_fixed_rate_bonds():
    date = pd.Timestamp("2026-01-01")
    portfolio = toy_portfolio(settlement_date=date, day_count="ACT/ACT")
    rates_pct = pd.DataFrame([[4.0, 4.0, 4.0, 4.0]], index=[date], columns=[0.5, 2.0, 10.0, 30.0])
    rates_data = RatesData(rates_pct)

    assert calc_dv01(portfolio, rates_data, date) > 0


def test_demo_pricing_error_summary_is_bounded_and_nonempty():
    summary = demo_pricing_error_summary()

    assert len(summary["comparison_table"]) == 4
    assert summary["max_abs_price_delta"] > 0
    assert summary["avg_abs_price_delta"] > 0


def test_fred_par_bootstrap_returns_zero_rate_history():
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    par_yields = pd.DataFrame(
        [
            [4.00, 4.05, 4.10, 4.20, 4.35],
            [4.01, 4.06, 4.11, 4.21, 4.36],
        ],
        index=dates,
        columns=[1 / 12, 0.25, 0.5, 1.0, 2.0],
    )

    zero_rates = bootstrap_zero_from_par(par_yields, frequency=2, dc_conv="ACT/ACT")

    assert list(zero_rates.columns) == list(par_yields.columns)
    assert zero_rates.shape == par_yields.shape
    assert zero_rates.notna().all().all()
    assert zero_rates.iloc[0, 0] == pytest.approx(par_yields.iloc[0, 0])
    assert zero_rates.iloc[0, 1] == pytest.approx(par_yields.iloc[0, 1])
