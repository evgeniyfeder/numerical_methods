import itertools
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from eq_config import EqConfig


def _solve(begin_ts: np.array,
           method: Callable[[np.array, EqConfig], np.array],
           config: EqConfig) -> Iterator[pd.DataFrame]:
    cur_ts = begin_ts
    start_frame = pd.DataFrame(
        {"time": 0, "x": config.a + config.dx * i, "T": begin_ts[i]}
        for i in range(config.num_points))
    yield start_frame
    for i in range(1, config.num_iter + 1):
        t = config.dt * i
        cur_ts = method(cur_ts, config)

        new_frame = pd.DataFrame(
            {"time": t, "x": config.a + config.dx * j, "T": cur_ts[j]}
            for j in range(config.num_points))
        yield new_frame


def solve(begin_ts: np.array,
          method: Callable[[np.array, EqConfig], np.array],
          config: EqConfig,
          freq: int = None) -> pd.DataFrame:
    frames = _solve(begin_ts, method, config)
    if freq is not None:
        frames = itertools.islice(frames, None, None, freq)
    return pd.concat(list(frames))
