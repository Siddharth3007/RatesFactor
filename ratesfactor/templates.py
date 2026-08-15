from io import BytesIO

import pandas as pd

from .zerocurve import curve_universe_template


def dataframe_to_xlsx_download(df):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buffer.getvalue()


def dataframe_to_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def standard_holdings_template():
    return pd.DataFrame({
        "Description": ["US TREASURY N/B", "US TREASURY N/B"],
        "Par Value": [2_000_000, 1_500_000],
        "Maturity": ["2030-05-15", "2035-08-15"],
        "Coupon (%)": [3.50, 4.25],
        "Frequency": [2, 2],
        "Settlement Date": ["2026-08-15", "2026-08-15"],
        "Day Count": ["ACT/ACT", "ACT/ACT"],
    })


def tlt_holdings_template():
    return pd.DataFrame({
        "Name": ["TREASURY BOND", "TREASURY BOND"],
        "Sector": ["Treasuries", "Treasuries"],
        "Asset Class": ["Fixed Income", "Fixed Income"],
        "Par Value": [2_000_000, 1_500_000],
        "Maturity": ["2053-08-15", "2054-11-15"],
        "Coupon (%)": [4.13, 4.75],
        "Settlement Date": ["2026-08-15", "2026-08-15"],
        "Day Count": ["ACT/ACT", "ACT/ACT"],
    })


def hedge_instruments_template():
    return pd.DataFrame({
        "Description": ["6M Treasury Bill Hedge", "2Y Treasury Note Hedge"],
        "Face Value": [1_000_000, 1_000_000],
        "Maturity": [0.50, 2.00],
        "Coupon (%)": [0.00, 4.00],
        "Frequency": [1, 2],
        "Cost (bps)": [0.20, 0.25],
        "Settlement Date": ["2026-08-15", "2026-08-15"],
        "Day Count": ["ACT/ACT", "ACT/ACT"],
    })


def curve_construction_universe_template():
    return curve_universe_template()
