import numpy as np
from typing import Callable

import eq_config


def implicit_against(prev_ts: np.ndarray, config=eq_config.EqConfig()):
    return _implicit_method(prev_ts, config,
                            lambda prev_t, cur_t, next_t, local_config: cur_t - local_config.s * (cur_t - prev_t) +
                                                                       local_config.r * (prev_t + next_t - 2 * cur_t)
                            )


def implicit_down(prev_ts: np.ndarray, config=eq_config.EqConfig()):
    return _implicit_method(prev_ts, config,
                            lambda prev_t, cur_t, next_t, local_config: cur_t - local_config.s * (next_t - cur_t) +
                                                                       local_config.r * (prev_t + next_t - 2 * cur_t)
                            )


def implicit_leapfrog(prev_ts: np.ndarray, config=eq_config.EqConfig()):
    return _implicit_method(prev_ts, config,
                            lambda prev_t, cur_t, next_t, local_config: cur_t - local_config.s * (next_t - prev_t)
                            )


def _implicit_method(prev_ts: np.ndarray, config, next_elem: Callable[[float, float, float, eq_config.EqConfig], float]):
    next_ts = np.zeros(config.num_points, dtype=np.float64)

    for i in range(config.num_points):
        prev_t = prev_ts[i - 1] if i > 0 else config.left_border(prev_ts)
        cur_t = prev_ts[i]
        next_t = prev_ts[i + 1]

        next_ts[i] = next_elem(prev_t, cur_t, next_t, config)
    return next_ts
