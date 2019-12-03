import numpy as np

import eq_config


def signum_start(config: eq_config.EqConfig):
    start = np.zeros(config.num_points, dtype=np.float64)

    for i in range(config.num_points // 2):
        start[i] = 1

    return start


def linear_start(config: eq_config.EqConfig, max: float = 1., min: float = 0):
    start = np.array([min for i in range(config.num_points)], dtype=np.float64)

    for i in range(config.num_points // 3, 2 * (config.num_points // 3)):
        start[i] = min + (i - config.num_points // 3) / (config.num_points // 3) * (max - min)

    for i in range(2 * (config.num_points // 3), config.num_points):
        start[i] = max

    return start


def random_start(config: eq_config.EqConfig):
    return np.random.rand(config.num_points)


