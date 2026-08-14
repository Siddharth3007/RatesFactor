import pandas as pd


def _clean_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _date_from_maturity_years(settlement_date, maturity_years):
    return pd.Timestamp(settlement_date) + pd.DateOffset(months=int(round(float(maturity_years) * 12)))


def _with_pricing_conventions(df, settlement_date, day_count="ACT/ACT"):
    df = df.copy()
    df["settlement_date"] = pd.Timestamp(settlement_date)
    df["day_count"] = day_count
    if "maturity_date" not in df.columns:
        df["maturity_date"] = [
            _date_from_maturity_years(settlement_date, maturity)
            for maturity in df["maturity"]
        ]
    return df


def toy_portfolio(settlement_date=None, day_count="ACT/ACT"):
    settlement_date = pd.Timestamp(settlement_date)
    return pd.DataFrame({
        "bond": ["2Y Note", "5Y Note", "10Y Note", "30Y Bond"],
        "face_value": [2_500_000, 2_500_000, 2_500_000, 2_500_000],
        "coupon": [0.040, 0.042, 0.045, 0.047],
        "maturity": [2, 5, 10, 30],
        "maturity_date": [
            _date_from_maturity_years(settlement_date, maturity)
            for maturity in [2, 5, 10, 30]
        ],
        "frequency": [2, 2, 2, 2],
        "positions": [1, 1, 1, 1],
        "settlement_date": settlement_date,
        "day_count": day_count,
    })


def front_end_hedge_instruments(settlement_date=None, day_count="ACT/ACT"):
    settlement_date = pd.Timestamp(settlement_date)
    hedges = pd.DataFrame({
        "bond": ["6M Treasury Bill Hedge", "2Y Treasury Note Hedge"],
        "face_value": [1_000_000, 1_000_000],
        "coupon": [0.00, 0.04],
        "maturity": [0.50, 2.00],
        "frequency": [1, 2],
        "positions": [1, 1],
    })
    return _with_pricing_conventions(hedges, settlement_date, day_count)


def long_duration_hedge_instruments(settlement_date=None, day_count="ACT/ACT"):
    settlement_date = pd.Timestamp(settlement_date)
    hedges = pd.DataFrame({
        "bond": ["2Y Hedge Note", "5Y Hedge Note", "10Y Hedge Note", "30Y Hedge Bond"],
        "face_value": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        "coupon": [0.04, 0.04, 0.04, 0.04],
        "maturity": [2, 5, 10, 30],
        "frequency": [2, 2, 2, 2],
        "positions": [1, 1, 1, 1],
    })
    return _with_pricing_conventions(hedges, settlement_date, day_count)


def mild_mismatch_hedge_instruments(settlement_date=None, day_count="ACT/ACT"):
    settlement_date = pd.Timestamp(settlement_date)
    hedges = pd.DataFrame({
        "bond": ["3Y Hedge Note", "7Y Hedge Note", "20Y Hedge Bond"],
        "face_value": [1_000_000, 1_000_000, 1_000_000],
        "coupon": [0.04, 0.04, 0.04],
        "maturity": [3, 7, 20],
        "frequency": [2, 2, 2],
        "positions": [1, 1, 1],
    })
    return _with_pricing_conventions(hedges, settlement_date, day_count)


def load_tlt_holdings_csv(path_or_buffer, as_of_date, target_notional=10_000_000, day_count="ACT/ACT"):
    try:
        raw = pd.read_csv(path_or_buffer, skiprows=9)
    except Exception:
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        raw = pd.read_csv(path_or_buffer)

    required_cols = {"Name", "Sector", "Asset Class", "Par Value", "Maturity", "Coupon (%)"}
    if not required_cols.issubset(raw.columns):
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(0)
        raw = pd.read_csv(path_or_buffer)

    missing_cols = sorted(required_cols - set(raw.columns))
    if missing_cols:
        raise ValueError(f"TLT holdings file is missing required columns: {missing_cols}")

    tlt = raw[
        (raw["Name"] == "TREASURY BOND")
        & (raw["Sector"] == "Treasuries")
        & (raw["Asset Class"] == "Fixed Income")
    ].copy()

    tlt["face_value"] = _clean_number(tlt["Par Value"])
    tlt["coupon"] = _clean_number(tlt["Coupon (%)"]) / 100
    tlt["maturity_date"] = pd.to_datetime(tlt["Maturity"], errors="coerce")
    if "Day Count" in tlt.columns:
        tlt["day_count"] = tlt["Day Count"].fillna(day_count)
    else:
        tlt["day_count"] = day_count

    as_of_date = pd.Timestamp(as_of_date)
    tlt["maturity"] = (tlt["maturity_date"] - as_of_date).dt.days / 365.25
    tlt = tlt.dropna(subset=["face_value", "coupon", "maturity"])
    tlt = tlt[tlt["maturity"] > 0].copy()

    if target_notional is not None:
        total_face = tlt["face_value"].sum()
        if total_face <= 0:
            raise ValueError("No positive TLT Treasury par value after cleaning.")
        tlt["face_value"] = tlt["face_value"] * target_notional / total_face

    return pd.DataFrame({
        "bond": tlt["Name"] + " " + tlt["Coupon (%)"].astype(str) + "% " + tlt["Maturity"].astype(str),
        "face_value": tlt["face_value"],
        "coupon": tlt["coupon"],
        "maturity": tlt["maturity"],
        "maturity_date": tlt["maturity_date"],
        "frequency": 2,
        "positions": 1,
        "settlement_date": as_of_date,
        "day_count": tlt["day_count"],
    }).reset_index(drop=True)


def load_standard_holdings(path_or_buffer, as_of_date, target_notional=10_000_000, day_count="ACT/ACT"):
    raw_df = pd.read_excel(path_or_buffer)

    if "Frequency" not in raw_df.columns:
        raw_df["Frequency"] = 2
    if "Day Count" in raw_df.columns:
        day_count_values = raw_df["Day Count"].fillna(day_count)
    else:
        day_count_values = day_count

    holdings = raw_df.loc[:, ["Description", "Coupon (%)", "Par Value", "Maturity", "Frequency"]].copy()
    holdings["Day Count"] = day_count_values
    holdings["face_value"] = _clean_number(holdings["Par Value"])
    holdings["coupon"] = _clean_number(holdings["Coupon (%)"]) / 100
    holdings["maturity_date"] = pd.to_datetime(holdings["Maturity"], errors="coerce")
    holdings["frequency"] = pd.to_numeric(holdings["Frequency"], errors="coerce").fillna(2).astype(int)

    as_of_date = pd.Timestamp(as_of_date)
    holdings["maturity"] = (holdings["maturity_date"] - as_of_date).dt.days / 365.25
    holdings = holdings.dropna(subset=["face_value", "coupon", "maturity"])
    holdings = holdings[holdings["maturity"] > 0].copy()

    if target_notional is not None:
        total_face = holdings["face_value"].sum()
        if total_face <= 0:
            raise ValueError("No positive par value after cleaning standard holdings.")
        holdings["face_value"] = holdings["face_value"] * target_notional / total_face

    return pd.DataFrame({
        "bond": (
            holdings["Description"].astype(str)
            + " "
            + holdings["Coupon (%)"].astype(str)
            + "% "
            + holdings["Maturity"].astype(str)
        ),
        "face_value": holdings["face_value"],
        "coupon": holdings["coupon"],
        "maturity": holdings["maturity"],
        "maturity_date": holdings["maturity_date"],
        "frequency": holdings["frequency"],
        "positions": 1,
        "settlement_date": as_of_date,
        "day_count": holdings["Day Count"],
    }).reset_index(drop=True)


def load_custom_hedge_instruments(path_or_buffer, settlement_date=None, day_count="ACT/ACT"):
    raw = pd.read_excel(path_or_buffer)
    settlement_date = pd.Timestamp(settlement_date)

    if "Frequency" not in raw.columns:
        raw["Frequency"] = 2
    if "Cost (bps)" not in raw.columns:
        raw["Cost (bps)"] = 0.25
    if "Day Count" not in raw.columns:
        raw["Day Count"] = day_count

    required = ["Description", "Face Value", "Maturity", "Coupon (%)", "Frequency", "Cost (bps)", "Day Count"]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Custom hedge template is missing required columns: {missing}")

    hedges = raw.loc[:, required].copy()
    hedges["face_value"] = _clean_number(hedges["Face Value"])
    hedges["coupon"] = _clean_number(hedges["Coupon (%)"]) / 100
    hedges["maturity"] = _clean_number(hedges["Maturity"])
    maturity_is_years = hedges["maturity"].notna()
    hedges["maturity_date"] = pd.NaT
    hedges.loc[~maturity_is_years, "maturity_date"] = pd.to_datetime(
        hedges.loc[~maturity_is_years, "Maturity"],
        errors="coerce",
    )
    has_maturity_date = hedges["maturity_date"].notna()
    hedges.loc[has_maturity_date, "maturity"] = (
        hedges.loc[has_maturity_date, "maturity_date"] - settlement_date
    ).dt.days / 365.25
    hedges.loc[~has_maturity_date, "maturity_date"] = [
        _date_from_maturity_years(settlement_date, maturity)
        for maturity in hedges.loc[~has_maturity_date, "maturity"]
    ]
    hedges["frequency"] = _clean_number(hedges["Frequency"]).fillna(2).astype(int)
    hedges["cost_bps"] = _clean_number(hedges["Cost (bps)"]).fillna(0.25)

    hedges = hedges.dropna(subset=["face_value", "coupon", "maturity"])
    hedges = hedges[(hedges["face_value"] > 0) & (hedges["maturity"] > 0)].copy()
    if hedges.empty:
        raise ValueError("No valid hedge instruments after cleaning the uploaded template.")

    hedge_instruments = pd.DataFrame({
        "bond": hedges["Description"].astype(str),
        "face_value": hedges["face_value"],
        "coupon": hedges["coupon"],
        "maturity": hedges["maturity"],
        "maturity_date": hedges["maturity_date"],
        "frequency": hedges["frequency"],
        "positions": 1,
        "settlement_date": settlement_date,
        "day_count": hedges["Day Count"],
    }).reset_index(drop=True)

    return hedge_instruments, hedges["cost_bps"].to_numpy(dtype=float)


def load_portfolio(source_type, path_or_buffer=None, as_of_date=None, target_notional=10_000_000, day_count="ACT/ACT"):
    as_of_date = pd.Timestamp(as_of_date)

    if source_type == "toy":
        return as_of_date, toy_portfolio(as_of_date, day_count)
    if source_type == "tlt":
        return as_of_date, load_tlt_holdings_csv(path_or_buffer, as_of_date, target_notional, day_count)
    if source_type == "standard":
        return as_of_date, load_standard_holdings(path_or_buffer, as_of_date, target_notional, day_count)
    raise ValueError("source_type must be 'toy', 'tlt', or 'standard'")


def build_hedged_portfolio(portfolio, hedge_instruments, hedge_weights):
    hedge_portfolio = hedge_instruments.copy()
    hedge_portfolio["positions"] = hedge_weights
    return pd.concat([portfolio, hedge_portfolio], ignore_index=True)


def portfolio_summary_stats(portfolio):
    exposure = portfolio["face_value"] * portfolio.get("positions", 1)
    abs_exposure = exposure.abs()
    total_abs_exposure = abs_exposure.sum()

    if total_abs_exposure == 0:
        return {
            "weighted_avg_maturity": 0.0,
            "weighted_avg_coupon": 0.0,
        }

    return {
        "weighted_avg_maturity": float((portfolio["maturity"] * abs_exposure).sum() / total_abs_exposure),
        "weighted_avg_coupon": float((portfolio["coupon"] * abs_exposure).sum() / total_abs_exposure),
    }
