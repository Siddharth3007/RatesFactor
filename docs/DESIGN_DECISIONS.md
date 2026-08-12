# Design Decisions

## Why Key Rate Duration and not just duration

**Context:** Duration assumes a parallel shift in the curve, i.e., the same change in rates across all maturities. However, in a real scenario, sensitivity to only a parallel shift is not enough, since the curve might change by different amounts across different maturities.

**Choice:** We construct a delta ladder to calculate the sensitivity of the portfolio at different maturity points. This accounts for non-parallel shifts in the curve, i.e., when the short end and long end shift by different amounts.

**Reasoning:** Traditional measures like Macaulay duration and modified duration give a single number, but what we require here is a vector of sensitivities, where each number describes the sensitivity of the portfolio to a change at a different tenor.

**Alternatives:** Macaulay duration, modified duration, effective duration, convexity, etc.

## Use PCA-factor hedging

**Context:** Simple duration hedging only protects the portfolio against parallel shifts in the curve. However, it is still exposed to non-parallel shifts in the curve.

**Choice:** We use PCA-factor hedging by decomposing historical yield curve movements into 3 factors that explain most of the movement in the curve. Hedging against these 3 factors can immunize the portfolio better than only duration hedging.

**Reasoning:** Duration hedging would not protect the portfolio against non-parallel movements in the curve, which happen frequently. PCA hedging simplifies the yield curve movement into a few important factors and helps hedge against non-parallel shifts as well.

**Alternatives:** Direct key-rate matching, OLS hedging against benchmark hedge instruments, NSS-factor hedging.

## Using Cosine Similarity PCA alignment

**Context:** In rolling PCA, the signs and order of PCA components can change from one window to another. This creates a problem because PC1, PC2, and PC3 may not represent the same type of curve movement across time.

**Choice:** We use cosine similarity to align the PCA components across rolling windows.

**Reasoning:** Cosine similarity compares the shape of two PCA loading vectors. If two vectors point in opposite directions, their cosine similarity is negative, so we flip the sign. If components change order, we use similarity to match the new components to the previous/reference components. This keeps the PCA factors more consistent through time.

**Tradeoff:** After alignment, the explained variance order might not always be strictly decreasing, because we are prioritizing economic continuity of the factor labels.

## Using ridge regularization

**Context:** A lot of times during rolling PCA hedge, the hedge weights were going to extremes. This was happening due to an ill-conditioned `A.T @ A` matrix.

**Choice:** Ridge regularization stabilizes the weights by reducing the L2 norm of the weight vector. By doing so, it tries to avoid extreme hedge weights.

**Reasoning:** We add a penalty on the L2 norm of the `h` vector in the optimization term, in addition to minimizing the difference between the portfolio factor exposure and the hedge factor exposure. This makes the hedge less exact, but more stable.

**Alternatives:** Unregularized least squares, constrained least squares, notional caps, liquidity penalties, transaction-cost-aware optimization.

## Show hedge suitability warnings

**Context:** Situations like a duration mismatch between the hedging instruments and the hedged portfolio are very likely to give bad or unstable hedges.

**Choice:** We show hedge suitability warnings when the hedge universe does not look appropriate for the portfolio.

**Reasoning:** I think it is important to let the user know when there is a case like this instead of just showing hedge weights and P&L numbers. Otherwise, the hedge may look mathematically valid but practically unreliable.

**Tradeoff:** This is just a warning, not a hard constraint. The model still runs, but the user is told to be careful while interpreting the results.

## Historical and Parametric VaR

**Context:** VaR is important to show tail risk scenarios for the portfolio and hedge, since this is important from a risk management perspective.

**Choice:** We show both historical simulation VaR and parametric PCA VaR.

**Reasoning:** Parametric VaR assumes a distribution for the losses and estimates quantiles from that. It is faster to compute, but it can fail to properly estimate tail risk if the actual losses have fatter tails than the assumed distribution. On the other hand, historical VaR captures real-world skewness and kurtosis better because it uses actual historical shocks, but it assumes that the historical window is relevant for the future. So it is better to give the user a comparison of both.

**Tradeoff:** Neither method is perfect. Historical VaR depends on the lookback window, and parametric VaR depends on the distributional assumption.

## Basel/IRRBB style shocks

**Context:** PCA hedging is based on historical yield curve movements, but in stress scenarios, historical correlations can break down.

**Choice:** We include Basel/IRRBB-style shocks along with custom rate shock scenarios.

**Reasoning:** Regulatory tests like Basel/IRRBB-style shocks expose structural failures, basis risk, and non-linear yield curve movements that the PCA hedging model might ignore. PCA hedging assumes historical market relationships, but in certain cases those relationships can break, and we need to test the portfolio using scenarios. These shocks also reveal extreme tail risks.

**Tradeoff:** These scenarios are not forecasts. They are stress tests.

## Keep pricing caveat visible instead of overclaiming

**Context:** The pricing engine is still simplified and does not yet use a fully bootstrapped zero curve.

**Choice:** We keep the pricing caveat visible in the documentation.

**Reasoning:** This keeps transparency in the model and acknowledges the fact that this is a research-grade model and not a production-grade one.

**Future Improvement:** Add a bootstrapped zero curve and compare how much it changes valuation, DV01, KRD, hedge weights, and VaR.
