# Future Scope

This document lists the most useful extensions for turning RatesFactor from a strong prototype into a more institutionally credible fixed-income risk tool.

## 1. Bootstrapped Zero / Discount Curve

Highest-priority improvement.

Current state:

- Cash flows are discounted using fitted Treasury curve rates.
- This is a public-data-friendly approximation, but not a production pricing method.

Future version:

- Bootstrap discount factors from Treasury bills, notes, and bonds.
- Interpolate log discount factors or zero rates.
- Price each cash flow from the discount curve.
- Compare DV01/KRD differences between par-curve pricing and zero-curve pricing.

This would directly address the most important fixed-income practitioner critique.

## 2. CUSIP-Level Treasury Data

Future ingestion modes could support:

- TLT holdings with CUSIPs and market values.
- TreasuryDirect/FiscalData reference data.
- TRACE or licensed market data for actual transaction/evaluated prices.

This would make portfolio inputs more realistic and allow cleaner curve construction from actual security prices.

## 3. Business-Day and Coupon Calendar Handling

Improve bond schedule handling:

- Business-day adjustment.
- Holiday calendar.
- Issue date support.
- Odd first/last coupons.
- Ex-coupon handling.
- Treasury ACT/ACT convention refinements.

## 4. Carry and Rolldown Attribution

Current backtest focuses on rate-shock P&L. A more desk-like decomposition would include:

- Carry.
- Rolldown.
- Curve move P&L.
- Residual/repricing error.
- Transaction costs.

This would make the hedge backtest more realistic.

## 5. Hedge Constraints

The current hedge solve is ridge-regularized least squares. Future versions could add:

- Notional caps.
- Long/short constraints.
- Liquidity penalties.
- Transaction-cost-aware optimization.
- Weight turnover penalty.
- Maximum leverage rules.

This would make hedge recommendations more realistic and reduce unstable overlays.

## 6. VaR Backtesting Enhancements

Current version includes breach counts, breach rates, and the Kupiec unconditional coverage test.

Future extensions:

- Christoffersen independence test.
- Conditional coverage test.
- Backtest visualizations through time.
- Historical vs parametric VaR reconciliation.
- Multi-window VaR comparison.

## 7. Swaps and Derivatives

Possible extensions:

- Interest-rate swaps.
- Futures hedges.
- Swaptions.
- Caps/floors.

These require additional curves and models:

- OIS discount curve.
- Forward curve.
- Volatility surface.
- Optionality/pricing models.

## 8. Credit and Spread Risk

For corporate bonds or credit portfolios, add:

- Treasury curve + credit spread curve decomposition.
- Spread DV01.
- Rating/sector buckets.
- OAS-style analysis.
- Credit spread scenarios.

## 9. Performance and Deployment

Improve app responsiveness:

- Cache curve pulls and expensive backtest computations.
- Vectorize repeated repricing loops where possible.
- Add progress/status indicators for long runs.
- Prepare a hosted Streamlit deployment with secrets managed outside the repo.

## 10. Documentation and Validation

Add:

- Example screenshots.
- Reproducible sample input files.
- Unit tests for pricing, risk, and VaR helpers.
- Known limitations section in the app itself.
- Methodology diagrams for PCA hedge construction.

## Suggested Roadmap

1. Add zero-curve bootstrap.
2. Add business-day/coupon schedule improvements.
3. Improve performance with caching/vectorization.
4. Add richer VaR backtesting.
5. Add carry/rolldown attribution.
6. Extend to swaps or spread-risk instruments.
