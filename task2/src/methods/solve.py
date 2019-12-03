from typing import Callable

import numpy as np
import pandas as pd
from eq_config import EqConfig


def solve(begin_ts: np.array, method: Callable[[np.array, EqConfig], np.array], config: EqConfig) -> pd.DataFrame:
    ts = begin_ts
    results = pd.DataFrame(
        {"time": 0, "x": config.a + config.dx * i, "T": begin_ts[i]}
        for i in range(config.num_points)
    )
    for i in range(1, config.num_iter + 1):
        t = config.dt * i
        ts = method(ts, config)

        df = pd.DataFrame(
            {"time": t, "x": config.a + config.dx * j, "T": ts[j]}
            for j in range(config.num_points)
        )
        results = results.append(df)

    return results
