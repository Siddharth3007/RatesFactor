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

### PCA Derivation

File: [pca-derivation.pdf](proof-of-work/pca-derivation.pdf)

Four-page handwritten derivation of PCA from the covariance matrix and the variance-maximization objective. The note connects principal components to eigenvectors of the covariance matrix and also relates the covariance-eigenvector view to the SVD view.

### Prototype Notebooks

Folder: [prototype-notebooks](proof-of-work/prototype-notebooks)

These notebooks are archived development artifacts from the exploratory phase of the project. They show how the Treasury curve analytics, scenario analysis, PCA hedging, historical VaR, parametric VaR, TLT/standard holdings ingestion, and spline diagnostics evolved before being refactored into the maintained package under `ratesfactor/` and the Streamlit app.

Included notebooks:

- [treasury-curve-toy.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy.ipynb): initial Treasury curve and portfolio risk prototype.
- [treasury-curve-toy-v2.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy-v2.ipynb): expanded prototype with multi-scenario analysis, historical simulation VaR, parametric PCA VaR, and holdings ingestion experiments.
- [treasury-curve-toy-clean.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy-clean.ipynb): cleaned notebook used as the direct source for modularizing the dashboard.
- [spline-forward-rate-diagnostics.ipynb](proof-of-work/prototype-notebooks/spline-forward-rate-diagnostics.ipynb): diagnostics around spline curve fitting and forward-rate behavior.
- [bootstrap.ipynb](proof-of-work/prototype-notebooks/bootstrap.ipynb): exploratory zero-curve bootstrap prototype using a curve construction universe, dirty prices, coupon cashflows, year fractions, and log discount-factor interpolation.
- [hedge-validation.ipynb](proof-of-work/prototype-notebooks/hedge-validation.ipynb): hedge validation notebook used to test PCA hedge behavior, rolling hedge weights, and backtest diagnostics before dashboard integration.
- [bootstrap-from-fred.ipynb](proof-of-work/prototype-notebooks/bootstrap-from-fred.ipynb): prototype for converting FRED CMT/par-yield history into proxy zero-rate history using deposit-style short tenors, par-bond bootstrapping, year fractions, and log discount-factor interpolation.
- [treasury-curve-toy-clean-bootstrap-stride.ipynb](proof-of-work/prototype-notebooks/treasury-curve-toy-clean-bootstrap-stride.ipynb): modified clean notebook that combines the FRED par-yield bootstrapper with stride-based rolling PCA hedge rebalancing before these ideas were brought into the app codebase.

Note: these notebooks are not the maintained implementation. They are included to show the project development path and may contain exploratory code, intermediate outputs, and older assumptions. Secrets and local file paths have been replaced with placeholders.
