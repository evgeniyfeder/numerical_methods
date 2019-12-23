import numpy as np
from eq_system import SystemConfig


class ImplicitMethod:
    @staticmethod
    def _solve_tridiagonal_linear(a: np.array, b: np.array, c: np.array, d: np.array):
        nf = len(d)
        ac, bc, cc, dc = (a, b, c, d)
        for it in range(1, nf):
            mc = ac[it - 1] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]
        xc = bc
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        return xc

    @staticmethod
    def _fill_coeffs_for_linear_x(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.array,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        raise NotImplementedError

    @staticmethod
    def _fill_coeffs_for_linear_t(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.ndarray,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        raise NotImplementedError

    @classmethod
    def count_next(cls, prev_ts: np.ndarray, prev_xs: np.array, f, config: SystemConfig):
        a, c = np.zeros(config.num_points - 1, dtype=np.float64), np.zeros(config.num_points - 1, dtype=np.float64)
        b, d = np.zeros(config.num_points, dtype=np.float64), np.zeros(config.num_points, dtype=np.float64)
        f(a, b, c, d, prev_xs, prev_ts, config)
        return cls._solve_tridiagonal_linear(a, b, c, d)

    @classmethod
    def next_xs_ts(cls, prev_ts: np.ndarray, prev_xs: np.array, config: SystemConfig) -> np.array:
        next_xs = cls.count_next(prev_ts, prev_xs, cls._fill_coeffs_for_linear_x, config)
        #next_xs[0] = 0
        next_xs[-1] = next_xs[-2]

        next_ts = cls.count_next(prev_ts, prev_xs, cls._fill_coeffs_for_linear_t, config)
        #next_ts[0] = config.T_m
        next_ts[-1] = next_ts[-2]
        return next_xs, next_ts


class WImplicitMethod(ImplicitMethod):
    @staticmethod
    def _fill_coeffs_for_linear_x(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.array,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        for i in range(config.num_points - 1):
            a[i] = -config.D / (config.dz ** 2)
            c[i] = -config.D / (config.dz ** 2)

        for i in range(config.num_points):
            b[i] = 2 * config.D / (config.dz ** 2) + 1 / config.dt
            d[i] = config.W(prev_xs[i], prev_ts[i]) + prev_xs[i] / config.dt

    @staticmethod
    def _fill_coeffs_for_linear_t(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.ndarray,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        for i in range(config.num_points - 1):
            a[i] = -config.kappa / (config.dz ** 2)
            c[i] = -config.kappa / (config.dz ** 2)

        for i in range(config.num_points):
            b[i] = 2 * config.kappa / (config.dz ** 2) + 1 / config.dt
            d[i] = -config.Q / config.C * config.W(prev_xs[i], prev_ts[i]) + prev_ts[i] / config.dt


class MagicWImplicitMethod(ImplicitMethod):
    @staticmethod
    def _fill_coeffs_for_linear_x(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.array,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        for i in range(config.num_points - 1):
            a[i] = -config.D / (config.dz ** 2)
            c[i] = -config.D / (config.dz ** 2)

        for i in range(config.num_points):
            b[i] = 2 * config.D / (config.dz ** 2) + 1 / config.dt - config.MagicW(prev_xs[i], prev_ts[i])
            d[i] = prev_xs[i] / config.dt

    @staticmethod
    def _fill_coeffs_for_linear_t(a: np.array, b: np.array, c: np.array, d: np.array, prev_xs: np.ndarray,
                                  prev_ts: np.array,
                                  config: SystemConfig):
        for i in range(config.num_points - 1):
            a[i] = -config.kappa / (config.dz ** 2)
            c[i] = -config.kappa / (config.dz ** 2)

        for i in range(config.num_points):
            b[i] = 2 * config.kappa / (config.dz ** 2) \
                   + 1 / config.dt + config.Q / config.C * config.W(prev_xs[i], prev_ts[i])
            d[i] = prev_ts[i] / config.dt
