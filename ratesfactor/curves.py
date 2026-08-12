import itertools

import numpy as np
from scipy.interpolate import CubicSpline


def compute_fwd(tenor_grid, yields_grid):
    ty = tenor_grid * yields_grid
    tenor_delta = tenor_grid[1] - tenor_grid[0]
    return np.diff(ty) / tenor_delta


def nss_loadings(tenors, tau1, tau2):
    x1 = tenors / tau1
    x2 = tenors / tau2
    X0 = np.ones(tenors.shape)
    X1 = (1 - np.exp(-x1)) / x1
    X2 = (1 - np.exp(-x1)) / x1 - np.exp(-x1)
    X3 = (1 - np.exp(-x2)) / x2 - np.exp(-x2)
    return np.column_stack([X0, X1, X2, X3])


def fit_nss_grid(tenors, yields, tau_min=0.2, tau_max=30, grid_size=100):
    grid_values = np.linspace(np.log(tau_min), np.log(tau_max), grid_size)
    rmse_min = np.inf
    beta_best = None
    tau_best = None

    for m1, m2 in itertools.product(grid_values, grid_values):
        tau1, tau2 = np.exp(m1), np.exp(m2)
        X = nss_loadings(tenors, tau1, tau2)
        beta = np.linalg.lstsq(X, yields, rcond=None)[0]
        fitted = X @ beta
        rmse = np.sqrt(np.mean((yields - fitted) ** 2))

        if rmse < rmse_min:
            rmse_min = rmse
            beta_best = beta
            tau_best = np.array([tau1, tau2])

    return beta_best, tau_best, rmse_min


def nss_yield(tenors, beta, tau1, tau2):
    return nss_loadings(tenors, tau1, tau2) @ beta


def cubic_spline_yield(tenors, yields, tenor_grid):
    spline = CubicSpline(tenors, yields, bc_type="natural")
    return spline(tenor_grid)


def shock_curve(base_curve, shocks):
    return base_curve + shocks


def count_turning_points(x):
    return np.sum(np.diff(np.sign(np.diff(x))) != 0)
