import numpy as np
import pandas as pd

from .pricing import portfolio_value
from .risk import make_delta_ladder


def pca_pnl_attribution(portfolio, rates_data, date_t, date_t1, pca, n_components=3):
    date_t = pd.Timestamp(date_t)
    date_t1 = pd.Timestamp(date_t1)

    curve_t = rates_data.get_curve(date_t, units="decimal")
    curve_t1 = rates_data.get_curve(date_t1, units="decimal")

    move_bp = (
        rates_data.rates_pct.loc[date_t1, rates_data.tenors]
        - rates_data.rates_pct.loc[date_t, rates_data.tenors]
    ) * 100
    move_bp = np.asarray(move_bp, dtype=float)

    dv01_ladder = make_delta_ladder(portfolio, rates_data, date_t)
    mean_move = pca.mean_
    centered_move = move_bp - mean_move
    scores = centered_move @ pca.components_.T

    rows = []
    mean_pnl = -dv01_ladder @ mean_move
    rows.append({"component": "mean", "pnl": mean_pnl})

    explained_move = mean_move.copy()
    for i in range(n_components):
        pc_move = scores[i] * pca.components_[i]
        explained_move = explained_move + pc_move
        rows.append({"component": f"PC{i + 1}", "pnl": -dv01_ladder @ pc_move})

    residual_move = move_bp - explained_move
    rows.append({"component": "residual", "pnl": -dv01_ladder @ residual_move})

    attribution = pd.DataFrame(rows)
    total_linearized_pnl = attribution["pnl"].sum()
    full_reprice_pnl = (
        portfolio_value(portfolio, curve_t1, settlement_date=date_t1)
        - portfolio_value(portfolio, curve_t, settlement_date=date_t)
    )

    attribution = pd.concat(
        [
            attribution,
            pd.DataFrame(
                [
                    {"component": "total_linearized", "pnl": total_linearized_pnl},
                    {"component": "full_reprice", "pnl": full_reprice_pnl},
                    {"component": "linearization_error", "pnl": full_reprice_pnl - total_linearized_pnl},
                ]
            ),
        ],
        ignore_index=True,
    )
    return attribution
