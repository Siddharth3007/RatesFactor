from itertools import permutations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def fit_pca(daily_changes_bp, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(daily_changes_bp)
    return pca


def _cosine_similarity_matrix(current_components, reference_components):
    current_norm = current_components / np.linalg.norm(current_components, axis=1, keepdims=True)
    reference_norm = reference_components / np.linalg.norm(reference_components, axis=1, keepdims=True)
    return reference_norm @ current_norm.T


def _best_component_order(cosine_similarities):
    n_components = cosine_similarities.shape[0]
    best_order = None
    best_score = -np.inf

    for order in permutations(range(n_components)):
        score = np.abs(cosine_similarities[np.arange(n_components), order]).sum()
        if score > best_score:
            best_score = score
            best_order = order

    return np.array(best_order)


def align_pca_to_reference(pca, reference_components):
    """Align rolling PCA signs/order to the previous window using cosine similarity."""
    n_components = min(len(pca.components_), len(reference_components))
    cosine_similarities = _cosine_similarity_matrix(
        pca.components_[:n_components],
        reference_components[:n_components],
    )
    order = _best_component_order(cosine_similarities)

    aligned_components = pca.components_[:n_components][order].copy()
    signs = np.sign(cosine_similarities[np.arange(n_components), order])
    signs[signs == 0] = 1
    aligned_components *= signs[:, None]

    pca.components_[:n_components] = aligned_components

    for attr in ("explained_variance_", "explained_variance_ratio_", "singular_values_"):
        if hasattr(pca, attr):
            values = getattr(pca, attr)
            values[:n_components] = values[:n_components][order]

    return pca


def fit_rolling_pca(daily_changes_bp, lookback=252, n_components=3, max_windows=None):
    dates = np.array(daily_changes_bp.index)
    rolling_pca = {}
    reference_components = None
    start_idx = lookback
    if max_windows is not None:
        start_idx = max(lookback, len(daily_changes_bp) + 1 - int(max_windows))

    for idx in range(start_idx, len(daily_changes_bp) + 1):
        window = daily_changes_bp.iloc[idx - lookback : idx]
        pca_window = fit_pca(window, n_components=n_components)
        if reference_components is not None:
            # Rolling PCA has arbitrary signs, and nearby PCs can swap labels across windows.
            # Aligning by cosine similarity keeps hedge weights from jumping because of PCA bookkeeping.
            pca_window = align_pca_to_reference(pca_window, reference_components)

        reference_components = pca_window.components_.copy()
        date = pd.Timestamp(dates[idx - 1])
        rolling_pca[date] = {
            "pca": pca_window,
            "start_date": dates[idx - lookback],
            "end_date": dates[idx - 1],
            "components": pca_window.components_,
            "explained_variance_ratio": pca_window.explained_variance_ratio_,
            "mean": pca_window.mean_,
        }

    return rolling_pca


def decompose_move(pca, rates_data, date, n_components=3):
    change_curve = np.array(rates_data.daily_changes_bp.loc[pd.Timestamp(date)])
    scores = pca.transform(change_curve.reshape(1, -1))
    pc_contribs = {
        f"PC{i + 1}": scores[0, i] * pca.components_[i]
        for i in range(n_components)
    }
    reconstructed_curve = pca.mean_ + sum(pc_contribs.values())
    residual = change_curve - reconstructed_curve
    return change_curve, pc_contribs, reconstructed_curve, residual
