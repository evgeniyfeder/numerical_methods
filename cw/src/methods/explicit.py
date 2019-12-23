import numpy as np

from eq_system import SystemConfig


class ExplicitMethod:
    @classmethod
    def _next_t(cls, prev_t: float, cur_t: float, next_t: float,
                prev_x: float, cur_x: float, next_x: float,
                config: SystemConfig) -> float:
        raise NotImplementedError

    @classmethod
    def _next_x(cls, prev_t: float, cur_t: float, next_t: float,
                prev_x: float, cur_x: float, next_x: float,
                config: SystemConfig) -> float:
        raise NotImplementedError

    @classmethod
    def next_xs_ts(cls, prev_ts: np.array, prev_xs: np.array, config: SystemConfig) -> np.array:
        next_ts = np.zeros(config.num_points, dtype=np.float64)
        next_xs = np.zeros(config.num_points, dtype=np.float64)

        for i in range(config.num_points):
            prev_t = prev_ts[i - 1] if i > 0 else config.T_m
            cur_t = prev_ts[i]
            next_t = prev_ts[i + 1] if i < config.num_points - 1 else prev_ts[config.num_points - 1]

            prev_x = prev_xs[i - 1] if i > 0 else 0
            cur_x = prev_xs[i]
            next_x = prev_xs[i + 1] if i < config.num_points - 1 else prev_xs[config.num_points - 1]

            next_ts[i] = cls._next_t(prev_t, cur_t, next_t, prev_x, cur_x, next_x, config)
            next_xs[i] = cls._next_x(prev_t, cur_t, next_t, prev_x, cur_x, next_x, config)

        return next_xs, next_ts


class ExplicitMethodImpl(ExplicitMethod):
    @classmethod
    def _next_t(cls, prev_t: float, cur_t: float, next_t: float,
                prev_x: float, cur_x: float, next_x: float,
                config: SystemConfig) -> float:
        return cur_t + config.kappa * config.dt * (prev_t + next_t - 2 * cur_t) / (config.dz ** 2) \
               - config.Q / config.C * config.W(cur_x, cur_t) * config.dt

    @classmethod
    def _next_x(cls, prev_t: float, cur_t: float, next_t: float,
                prev_x: float, cur_x: float, next_x: float,
                config: SystemConfig) -> float:
        return cur_x + config.D * (prev_x + next_x - 2 * cur_x) / (config.dz ** 2) + config.W(cur_x, cur_t)
