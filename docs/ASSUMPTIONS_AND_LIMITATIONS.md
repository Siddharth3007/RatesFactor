# Assumptions and Limitations

RatesFactor is a research-grade fixed-income risk prototype. It is intended to demonstrate rates risk methodology and software engineering, not to replicate a production pricing or risk platform.

## Main Assumptions

- The risk universe is U.S. Treasury curve risk.
- Treasury curve data comes from FRED constant-maturity Treasury series.
- Portfolio instruments are treated as fixed-rate Treasury-like bonds.
- Coupon schedules are inferred from maturity date and coupon frequency.
- The default day-count convention is ACT/ACT.
- Hedge instruments are represented as cash Treasury-like instruments.
- Transaction costs are modeled as simple basis-point costs on hedge notional changes.
- PCA is fitted on daily Treasury yield changes.
- Historical VaR replays historical yield curve shocks.
- Parametric VaR assumes normally distributed PCA factor shocks.

## Current Limitations

### Pricing and Discounting

The largest methodological limitation is that the current pricing engine discounts cash flows using fitted Treasury curve rates rather than a fully bootstrapped zero/discount curve.

In production fixed-income pricing, cash flows should generally be discounted using zero rates or discount factors built from market instruments. Using fitted par/constant-maturity yields introduces approximation error into:

- Dirty value.
- Clean value.
- Accrued-interest-adjusted value.
- DV01.
- Key-rate DV01.
- Hedge weights.
- VaR and scenario P&L.

This is the highest-priority future improvement.

As a bounded demo sensitivity check, the bundled curve-construction universe compares fitted/par-yield proxy pricing with the bootstrapped zero-curve mode on the toy 2Y/5Y/10Y/30Y bonds. In that demo universe, the average absolute price difference is about **$2.07 per $100 face**, and the largest difference is about **$5.60 per $100 face** on the 30Y bond. This number is not a universal estimate; it is included to show the approximation can be measured rather than ignored.

The fitted/par-yield proxy price is found by treating coupon rates as par-yield-style curve points for coupon bonds and converting bills into discount-implied rates. The bootstrapped price is found by solving discount factors from dirty prices in maturity order and then discounting the same toy bond cash flows from the resulting zero curve.

### Coupon Schedule Realism

The app supports regular coupon schedules inferred by stepping backward from maturity date. It does not currently handle:

- Odd first coupons.
- Odd last coupons.
- Ex-coupon periods.
- Treasury holiday calendars.
- Business-day adjustment.
- Actual settlement conventions by instrument type.

### Treasury Market Data

FRED constant-maturity yields are useful for public prototyping, but they are not the same as a full CUSIP-level Treasury pricing dataset. A more realistic system would ingest:

- CUSIPs.
- Actual clean prices.
- Accrued interest.
- Issue dates.
- Maturity dates.
- Coupon schedules.
- Bid/ask or evaluated prices.

### PCA Interpretation

PCA factors are statistical objects. They are often interpreted as level, slope, and curvature, but this interpretation can change across regimes.

Rolling PCA factors are aligned using cosine similarity to reduce sign/order instability. This improves continuity, but it does not guarantee that factor identities are economically stable in every market regime.

### Hedge Construction

The hedge solve is a regularized least-squares factor hedge, not a constrained portfolio optimizer. It does not currently enforce:

- Maximum notional limits.
- Long-only or short-only constraints.
- Liquidity limits.
- Integer trade sizes.
- Funding constraints.
- Bid/ask spread by instrument.

The hedge suitability warning is a diagnostic, not a trading rule.

### VaR

Historical VaR and parametric PCA VaR are both simplified:

- Historical VaR depends heavily on the chosen lookback window.
- Parametric PCA VaR assumes normal factor shocks.
- The Kupiec test checks unconditional breach frequency, not independence or clustering of breaches.
- The current VaR framework does not model liquidity, spread, optionality, or nonlinear derivatives exposure.

### Backtest P&L

The hedge backtest focuses on rate-shock P&L and transaction costs. It does not yet include a full carry/rolldown decomposition.

## How to Interpret Results

The dashboard should be interpreted as a risk analytics prototype:

- Good for understanding curve exposure, hedge coverage, PCA factor behavior, and scenario sensitivity.
- Not sufficient for production trade booking, valuation, model validation, or regulatory reporting.

The most defensible project positioning is:

> A prototype Treasury risk analytics dashboard focused on key-rate risk, PCA factor hedging, transaction-cost-aware backtesting, and VaR/scenario analysis, with documented pricing limitations and future scope for zero-curve bootstrapping.
