import itertools
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from eq_system import SystemConfig
from tqdm import tqdm_notebook


def _solve(begin_ts: np.array,
           begin_xs: np.array,
           method: Callable[[np.array, np.array, SystemConfig], np.array],
           config: SystemConfig,
           freq: int) -> Iterator[pd.DataFrame]:
    cur_ts = begin_ts
    cur_xs = begin_xs

    start_frame = pd.DataFrame(
        {"time": 0, "x": config.dz * i, "T": begin_ts[i], "X": begin_xs[i]}
        for i in range(config.num_points))
    yield start_frame

    for i in tqdm_notebook(range(1, config.num_iter + 1)):
        t = config.dt * i
        cur_xs, cur_ts = method(cur_ts, cur_xs, config)
        if i % freq == 0:
            new_frame = pd.DataFrame(
                {"time": t, "x": config.dz * j, "T": cur_ts[j], "X": cur_xs[j]}
                for j in range(config.num_points))
            yield new_frame


def solve(*, begin_ts: np.array,
          begin_xs: np.array,
          method: Callable[[np.array, np.array, SystemConfig], np.array],
          config: SystemConfig,
          freq: int = None) -> pd.DataFrame:
    frames = _solve(begin_ts, begin_xs, method, config, freq=freq)
    return pd.concat(list(frames))
