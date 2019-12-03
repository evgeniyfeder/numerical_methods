import numpy as np

from eq_config import EqConfig


class ExplicitMethod:
    @classmethod
    def _next_element(cls, prev_t: float, cur_t: float, next_t: float, config: EqConfig) -> float:
        raise NotImplementedError

    @classmethod
    def next_ts(cls, prev_ts: np.array, config: EqConfig) -> np.array:
        next_ts = np.zeros(config.num_points, dtype=np.float64)

        for i in range(config.num_points):
            prev_t = prev_ts[i - 1] if i > 0 else config.left_border(prev_ts)
            cur_t = prev_ts[i]
            next_t = prev_ts[i + 1] if i < config.num_points - 1 else config.right_border(prev_ts)

            next_ts[i] = cls._next_element(prev_t, cur_t, next_t, config)
        return next_ts


class ExplicitAgainst(ExplicitMethod):
    @classmethod
    def _next_element(cls, prev_t: float, cur_t: float, next_t: float, config: EqConfig) -> float:
        return cur_t - config.s * (cur_t - prev_t) + config.r * (prev_t + next_t - 2 * cur_t)


class ExplicitDown(ExplicitMethod):
    @classmethod
    def _next_element(cls, prev_t: float, cur_t: float, next_t: float, config: EqConfig) -> float:
        return cur_t - config.s * (next_t - cur_t) + config.r * (prev_t + next_t - 2 * cur_t)


class ExplicitLeapfrog(ExplicitMethod):
    @classmethod
    def _next_element(cls, prev_t: float, cur_t: float, next_t: float, config: EqConfig) -> float:
        return cur_t - config.s * (next_t - prev_t)
