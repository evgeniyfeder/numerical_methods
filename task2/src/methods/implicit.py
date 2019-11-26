import numpy as np

from eq_config import EqConfig


class ImplicitMethod:
    @classmethod
    def _next_element(cls, *, up_left_t: float, up_mid_t: float, down_mid_t: float, config: EqConfig) -> float:
        raise NotImplementedError

    @classmethod
    def next_ts(cls, prev_ts: np.array, config: EqConfig) -> np.array:
        next_ts = np.zeros(config.num_points, dtype=np.float64)

        for i in range(config.num_points):
            up_left_t = next_ts[i - 2] if i > 1 else config.left_border(prev_ts)
            up_mid_t = next_ts[i - 1] if i > 0 else config.left_border(prev_ts)
            down_mid_t = prev_ts[i - 1] if i > 0 else config.left_border(prev_ts)
            # todo: не уверен на счет left_border
            next_ts[i] = cls._next_element(up_left_t=up_left_t, up_mid_t=up_mid_t, down_mid_t=down_mid_t, config=config)
        return next_ts


class ImplicitAgainst(ImplicitMethod):
    @classmethod
    def _next_element(cls, *, up_left_t: float, up_mid_t: float, down_mid_t: float, config: EqConfig) -> float:
        u = config.u
        kappa = config.kappa
        dx = config.dx
        dt = config.dt
        return up_left_t * (config.kappa / (config.u * config.dx - config.kappa)) - \
               up_mid_t * (dx ** 2 - u * dx * dt + 2 * dt * kappa) / (u * dx * dt - kappa * dt) + \
               down_mid_t * (dx ** 2 / (dt * (u * dx - kappa)))


class ImplicitDown(ImplicitMethod):
    @classmethod
    def _next_element(cls, *, up_left_t: float, up_mid_t: float, down_mid_t: float, config: EqConfig) -> float:
        u = config.u
        kappa = config.kappa
        dx = config.dx
        dt = config.dt
        return - up_left_t * (u * dx + kappa) / kappa + \
                 up_mid_t * (dx ** 2 + u * dx * dt + 2 * dt * kappa) / (kappa * dt) - \
                 down_mid_t * dx ** 2 / (kappa * dt)
