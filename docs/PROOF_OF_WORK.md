# Proof of Work

This folder collects artifacts that show the thinking behind RatesFactor, not just the final dashboard.

## Included Artifacts

### PCA Hedge Derivation

File: [pca-hedge-derivation.pdf](proof-of-work/pca-hedge-derivation.pdf)

Two-page handwritten derivation of the PCA hedge formula used in the project. The notes connect the geometric least-squares view of hedging to the implemented ridge-regularized hedge solve:

```text
min_h || A h + y ||^2 + lambda ||h||^2
```

where:

- `A` is the hedge-instrument factor exposure matrix.
- `y` is the portfolio factor exposure vector.
- `h` is the hedge weight vector.
- `lambda` controls ridge regularization.

### Prototype Notebooks

Folder: [prototype-notebooks](proof-of-work/prototype-notebooks)

These notebooks are archived development artifacts from the exploratory phase of the project. They show how the Treasury curve analytics, scenario analysis, PCA hedging, historical VaR, parametric VaR, TLT/standard holdings ingestion, and spline diagnostics evolved before being refactored into the maintained package under `ratesfactor/` and the Streamlit app.

Included notebooks:

- [treasury-curve-toy.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy.ipynb): initial Treasury curve and portfolio risk prototype.
- [treasury-curve-toy-v2.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy-v2.ipynb): expanded prototype with multi-scenario analysis, historical simulation VaR, parametric PCA VaR, and holdings ingestion experiments.
- [treasury-curve-toy-clean.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy-clean.ipynb): cleaned notebook used as the direct source for modularizing the dashboard.
- [spline-forward-rate-diagnostics.ipynb](proof-of-work/prototype-notebooks/spline-forward-rate-diagnostics.ipynb): diagnostics around spline curve fitting and forward-rate behavior.

Note: these notebooks are not the maintained implementation. They are included to show the project development path and may contain exploratory code, intermediate outputs, and older assumptions. Secrets and local file paths have been replaced with placeholders.

## Additional Proof-of-Work Ideas

The most useful proof-of-work artifacts are the ones that show judgment, debugging, validation, or methodology. Recommended additions:

1. **Methodology walkthrough**
   - A short PDF or Markdown note explaining the full pipeline:
     portfolio inputs -> bond pricing -> KRD ladder -> PCA factors -> hedge solve -> backtest -> VaR.

2. **Known limitations note**
   - A clear note explaining the current fitted-curve pricing limitation and the planned zero-curve bootstrap upgrade.

3. **Validation notebook**
   - A compact notebook that tests:
     - ACT/ACT year fractions.
     - clean vs dirty value behavior.
     - DV01 sign and magnitude.
     - Kupiec VaR backtest outputs.
     - PCA sign-alignment behavior.

4. **Before/after screenshots**
   - Screenshots showing:
     - raw portfolio input.
     - line-item analytics table.
     - KRD concentration.
     - hedge suitability warning.
     - hedge backtest.
     - VaR backtest with Kupiec p-value.

5. **Design decision log**
   - Short notes on why you chose:
     - PCA factor hedging.
     - ridge regularization.
     - cosine-similarity PCA alignment.
     - historical and parametric VaR.
     - IRRBB scenario shocks.

6. **Future-scope roadmap**
   - A prioritized roadmap showing that you know what separates a prototype from a production-grade fixed-income risk engine.

7. **Small test suite**
   - A few unit tests are often more convincing than more screenshots. Useful targets:
     - `year_fraction`
     - `bond_value`
     - `make_delta_ladder`
     - `kupiec_unconditional_coverage`
     - `compute_pca_hedge_weights`

## What Not to Add

Avoid cluttering the repo with:

- Long unedited notebooks.
- Too many screenshots without explanation.
- Raw data files containing downloaded holdings.
- Secret keys or local config.
- Claims that make the pricing engine sound production-grade before the zero-curve bootstrap is added.

The goal is to show clear reasoning and implementation discipline, not just volume.
