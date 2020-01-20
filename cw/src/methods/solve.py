import itertools
from typing import Callable, Iterator, Tuple, Optional

import attr
import numpy as np
import pandas as pd
from eq_system import SystemConfig
from tqdm import tqdm_notebook


@attr.s(auto_attribs=True)
class Frame:
    time: float
    Ts: np.array
    Xs: np.array


def solve_iter(begin_ts: np.array,
               begin_xs: np.array,
               method: Callable[[np.array, np.array, SystemConfig],
                                Iterator[Tuple[np.array, np.array]]],
               config: SystemConfig,
               freq: int = 100,
               num_iter: int = 1000000000) -> Iterator[Frame]:
    # return itertools.islice(
    #     map(lambda p: Frame(time=config.dt * p[0], Ts=p[1][0], Xs=p[1][1]),
    #         enumerate(method(begin_ts, begin_xs, config), start=0))
    #     , 0, None, freq
    # )
    for i, (cur_ts, cur_xs) in tqdm_notebook(enumerate(method(begin_ts, begin_xs, config),
                                                       start=0),
                                             total=num_iter):
        if i % freq == 0:
            yield Frame(time=config.dt * i, Ts=cur_ts, Xs=cur_xs)
        if i > num_iter:
            break

# def solve(*, begin_ts: np.array,
#           begin_xs: np.array,
#           method: Callable[[np.array, np.array, SystemConfig], np.array],
#           config: SystemConfig,
#           freq: int = None) -> pd.DataFrame:
#     frames = _solve(begin_ts, begin_xs, method, config, freq=freq)
#     return pd.concat(list(frames))
