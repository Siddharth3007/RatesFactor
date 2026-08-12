# Methodology

RatesFactor follows a rates risk workflow that starts with Treasury curve data, values a portfolio, decomposes curve exposure, constructs a PCA hedge, and evaluates the hedge under backtests, scenarios, and VaR.

## 1. Treasury Curve Data

The app pulls U.S. Treasury constant-maturity yield series from FRED across 11 tenors:

```text
1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y
```

Daily yield changes are computed in basis points and used for PCA, scenario generation, historical simulation VaR, and hedge backtesting.

## 2. Portfolio Normalization

All input modes are normalized into a common portfolio schema:

```text
bond, face_value, coupon, maturity, maturity_date, frequency, positions, settlement_date, day_count
```

Supported input modes:

- Built-in toy Treasury portfolio.
- Standard holdings Excel template.
- iShares TLT-style holdings CSV.
- Custom hedge instruments template.

## 3. Bond Valuation

The pricing engine generates future coupon cash flows from maturity date, settlement date, coupon frequency, and face value. It computes:

- Dirty value.
- Clean value.
- Accrued interest.
- Price per $100 face value.

The dashboard supports ACT/ACT, ACT/365.25, ACT/365, and ACT/360 year fractions. Treasury-style semiannual coupons are supported through the `frequency` input.

Current limitation: discounting is based on fitted Treasury curve rates, not a bootstrapped zero/discount curve. This is acceptable for a prototype risk dashboard, but it is not institutional-grade fixed-income pricing.

## 4. DV01 and Key-Rate DV01

Portfolio DV01 is computed using bump-and-reprice:

```text
DV01 = (V_down - V_up) / 2
```

where the curve is shocked up/down by 1 bp.

Key-rate DV01 is computed by shocking one Treasury tenor at a time and repricing the portfolio. This produces an 11-dimensional sensitivity ladder aligned to the Treasury curve tenors.

## 5. PCA Yield-Curve Factors

PCA is fitted on historical daily Treasury yield changes. The first three components are interpreted as approximate level, slope, and curvature factors.

Rolling PCA creates a sign/order instability problem because PCA components can flip sign or rotate between windows. RatesFactor aligns rolling PCA loadings by cosine similarity against reference factors to keep labels economically stable through time.

Important note: after cosine-similarity alignment, displayed PC labels may not strictly follow descending explained variance.

## 6. PCA Hedge Construction

The portfolio and each hedge instrument are converted into key-rate DV01 ladders. These ladders are projected into PCA factor space:

```text
portfolio factor exposure = V @ portfolio_krd
hedge factor exposure     = V @ hedge_krd
```

where `V` contains selected PCA loading vectors.

The hedge solve chooses hedge weights that reduce portfolio factor exposure:

```text
min_h || A h + y ||^2 + lambda ||h||^2
```

where:

- `y` is the portfolio PCA factor exposure.
- `A` is the matrix of hedge-instrument PCA factor exposures.
- `h` is the vector of hedge positions/weights.
- `lambda` is the ridge regularization parameter.

Ridge regularization improves numerical stability when hedge instruments are collinear or when the hedge universe poorly spans the portfolio risk.

## 7. Hedge Suitability Diagnostics

The app reports diagnostics to prevent blind interpretation of hedge outputs:

- Condition number.
- Hedge notional ratio.
- Portfolio weighted average maturity.
- Hedge universe weighted average maturity.
- Maximum hedge maturity.
- Top portfolio key-rate bucket.
- Residual factor norm.

Large hedge notionals, negative hedged market value, or sign-flipped hedged P&L may occur when the hedge universe does not cover the portfolio key-rate exposure.

## 8. Hedge Backtest

The rolling hedge backtest:

1. Uses rolling PCA windows.
2. Recomputes hedge weights through time.
3. Applies realized daily curve shocks.
4. Computes unhedged P&L, gross hedged P&L, net hedged P&L, and transaction costs.

Summary statistics include average absolute P&L reduction, volatility reduction, hit rate, and total transaction costs.

## 9. Scenario Analysis

Scenario analysis applies predefined curve shocks in basis points and fully reprices the unhedged and hedged portfolios.

The scenario set includes custom curve-shape shocks and Basel/IRRBB-style USD shocks:

- Parallel up/down.
- Bear/bull steepeners.
- Bear/bull flatteners.
- Belly shocks.
- Front-end and long-end shocks.
- Basel/IRRBB-style parallel, short-rate, steepener, and flattener scenarios.

## 10. VaR and Expected Shortfall

RatesFactor includes two VaR approaches:

### Historical Simulation VaR

Historical daily yield changes are replayed as curve shocks. The portfolio is repriced under each historical shock to generate a loss distribution.

### Parametric PCA VaR

Daily yield changes are projected into PCA factor scores. Factor covariance is estimated from historical factor scores. Portfolio factor exposure is combined with factor covariance to estimate P&L volatility:

```text
pnl_vol = sqrt(b.T @ factor_cov @ b)
```

Normal VaR and expected shortfall are then computed from this volatility.

## 11. VaR Backtesting

The historical VaR backtest uses rolling historical P&L windows:

1. Estimate VaR from the prior lookback window.
2. Compare it with the next realized daily loss.
3. Count breaches.
4. Compare actual breach rate with expected breach rate.
5. Report the Kupiec unconditional coverage LR statistic and p-value.

This validates whether realized breach frequency is consistent with the target VaR tail probability.

## 12. P&L Attribution

P&L attribution decomposes one-day portfolio P&L into PCA factor contributions and residual:

```text
P&L ≈ PC1 contribution + PC2 contribution + PC3 contribution + residual
```

This helps distinguish level, slope, curvature, and non-factor-driven changes.
