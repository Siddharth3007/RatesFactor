# RatesFactor

RatesFactor is a research-grade fixed-income risk analytics dashboard for Treasury portfolios. It converts a notebook prototype into a modular Python package and Streamlit app for portfolio ingestion, curve analytics, key-rate risk, PCA factor hedging, hedge backtesting, scenarios, VaR/ES, and P&L attribution.

The project is designed as a realistic rates risk prototype, not a production pricing system. The main focus is the risk workflow: where curve exposure sits, how a PCA hedge behaves, and when a hedge universe is unsuitable.

## Highlights

- Treasury curve ingestion from FRED constant-maturity Treasury series across 11 tenors.
- Portfolio ingestion from a toy portfolio, a standard holdings template, or iShares TLT-style holdings files.
- Settlement-date-aware fixed-rate bond valuation with clean value, dirty value, accrued interest, ACT/ACT support, frequency, and maturity dates.
- Portfolio analytics: market value, weighted average maturity, weighted average coupon, DV01, effective duration, convexity, and line-item bond analytics.
- 11-key-rate DV01 ladder and key-rate concentration diagnostics.
- Rolling 3-factor PCA on Treasury curve changes with cosine-similarity sign/alignment handling.
- Ridge-regularized PCA hedge solve with hedge suitability diagnostics.
- Transaction-cost-aware rolling hedge backtest with gross and net P&L.
- Historical simulation VaR/ES and parametric PCA VaR/ES.
- Historical VaR backtesting with breach counts, breach rates, Kupiec LR statistic, and p-value.
- 21 rate shock scenarios, including Basel/IRRBB-style USD shocks.
- PCA P&L attribution across PC1, PC2, PC3, and residual.

## Dashboard

Run the Streamlit app locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The dashboard requires a FRED API key. Use either a local Streamlit secrets file:

```toml
# .streamlit/secrets.toml
FRED_API_KEY = "your_key_here"
```

or an environment variable:

```bash
export FRED_API_KEY="your_key_here"
```

Do not commit `.streamlit/secrets.toml`.

## Project Structure

```text
RatesFactor/
├── streamlit_app.py
├── requirements.txt
├── README.md
├── docs/
│   ├── METHODOLOGY.md
│   ├── ASSUMPTIONS_AND_LIMITATIONS.md
│   ├── FUTURE_SCOPE.md
│   ├── PROOF_OF_WORK.md
│   └── proof-of-work/
│       ├── pca-hedge-derivation.pdf
│       └── prototype-notebooks/
└── ratesfactor/
    ├── attribution.py
    ├── config.py
    ├── curves.py
    ├── data.py
    ├── hedging.py
    ├── pca.py
    ├── plots.py
    ├── portfolio.py
    ├── pricing.py
    ├── risk.py
    ├── scenarios.py
    ├── templates.py
    └── var.py
```

## Portfolio Inputs

The app supports three portfolio modes:

- **Toy portfolio**: built-in 2Y, 5Y, 10Y, and 30Y Treasury example.
- **Standard holdings template**: user-supplied Excel file with description, par value, coupon, maturity, frequency, settlement date, and day count.
- **TLT-style holdings file**: iShares-style holdings CSV cleaned into Treasury bond rows.

Custom hedge instruments can also be uploaded through a hedge-instruments template.

## Modeling Scope

RatesFactor focuses on Treasury curve risk:

- Yield curve shape and PCA factors.
- DV01/key-rate DV01 exposure.
- PCA-based hedge construction.
- Hedge stability diagnostics.
- Scenario shocks and VaR/ES.
- Transaction-cost-aware backtesting.

The pricing engine is intentionally simplified compared with institutional systems. The current version prices from fitted Treasury curve rates rather than a fully bootstrapped zero/discount curve. This limitation is documented because every downstream risk number inherits that approximation.

## Documentation

- [Methodology](docs/METHODOLOGY.md)
- [Assumptions and Limitations](docs/ASSUMPTIONS_AND_LIMITATIONS.md)
- [Future Scope](docs/FUTURE_SCOPE.md)
- [Proof of Work](docs/PROOF_OF_WORK.md)

## Status

This is a research prototype for Treasury risk analytics. It is not a production-grade pricing, risk, or trading system.
