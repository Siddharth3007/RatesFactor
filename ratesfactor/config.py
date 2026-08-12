import numpy as np


TREASURY_SERIES = [
    (1 / 12, "DGS1MO"),
    (3 / 12, "DGS3MO"),
    (6 / 12, "DGS6MO"),
    (1.0, "DGS1"),
    (2.0, "DGS2"),
    (3.0, "DGS3"),
    (5.0, "DGS5"),
    (7.0, "DGS7"),
    (10.0, "DGS10"),
    (20.0, "DGS20"),
    (30.0, "DGS30"),
]


PRESET_SCENARIOS_BP = {
    "parallel_up_25bp": np.array([25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25]),
    "parallel_down_25bp": np.array([-25, -25, -25, -25, -25, -25, -25, -25, -25, -25, -25]),
    "bear_steepener": np.array([5, 5, 8, 10, 15, 20, 25, 30, 35, 45, 50]),
    "bull_steepener": np.array([-35, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10]),
    "bear_flattener": np.array([50, 45, 40, 35, 30, 25, 18, 12, 8, 5, 3]),
    "bull_flattener": np.array([-5, -8, -12, -18, -25, -30, -35, -40, -45, -50, -55]),
    "belly_selloff": np.array([5, 8, 12, 20, 35, 45, 50, 45, 30, 15, 10]),
    "belly_rally": np.array([-5, -8, -12, -20, -35, -45, -50, -45, -30, -15, -10]),
    "front_end_selloff": np.array([60, 55, 50, 45, 35, 25, 15, 10, 5, 3, 2]),
    "front_end_rally": np.array([-60, -55, -50, -45, -35, -25, -15, -10, -5, -3, -2]),
    "long_end_selloff": np.array([2, 3, 5, 8, 10, 12, 18, 25, 35, 50, 60]),
    "long_end_rally": np.array([-2, -3, -5, -8, -10, -12, -18, -25, -35, -50, -60]),
    "inflation_reacceleration": np.array([35, 40, 45, 50, 55, 60, 65, 65, 60, 55, 50]),
    "growth_scare": np.array([-40, -45, -50, -55, -55, -50, -45, -40, -35, -25, -20]),
    "liquidity_stress_long_end": np.array([10, 12, 15, 20, 25, 30, 40, 50, 65, 85, 100]),
}

