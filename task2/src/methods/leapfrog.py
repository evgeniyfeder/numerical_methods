from typing import Callable

import numpy as np
import pandas as pd

from eq_config import EqConfig
from methods.solve import solve, EqConfig
from methods.explicit import ExplicitDown


def next_leapfrog(prev_ts: np.ndarray, prepre_ts: np.ndarray, config: EqConfig) -> np.array:
    next_ts = np.zeros(config.num_points, dtype=np.float64)

    for i in range(config.num_points):
        prev_t = prev_ts[i - 1] if i > 0 else config.left_border(prev_ts)
        cur_t = prepre_ts[i]
        next_t = prev_ts[i + 1] if i < config.num_points - 1 else config.right_border(prev_ts)

        next_ts[i] = cur_t - config.s * (next_t - prev_t)
    return next_ts


def solve_leapfrog(begin_ts: np.ndarray, config: EqConfig, freq: int = 1) -> pd.DataFrame:
    prepre_ts = begin_ts
    pre_ts = ExplicitDown.next_ts(begin_ts, config)

    results = pd.DataFrame(
        {"time": 0, "x": config.a + config.dx * i, "T": prepre_ts[i]}
        for i in range(config.num_points)
    )

    results.append(pd.DataFrame(
        {"time": config.dt, "x": config.a + config.dx * i, "T": pre_ts[i]}
        for i in range(config.num_points)
    ))

    for i in range(2, config.num_iter + 1):
        t = config.dt * i

        tmp = next_leapfrog(pre_ts, prepre_ts, config)
        prepre_ts = pre_ts
        pre_ts = tmp

        if i % freq == 0:
            df = pd.DataFrame(
                {"time": t, "x": config.a + config.dx * j, "T": tmp[j]}
                for j in range(config.num_points)
            )
            results = results.append(df)

    return results
