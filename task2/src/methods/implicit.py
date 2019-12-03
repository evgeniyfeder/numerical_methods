import numpy as np

from eq_config import EqConfig


class ImplicitMethod:
    @staticmethod
    def _solve_tridiagonal_linear(a: np.array, b: np.array, c: np.array, d: np.array):
        nf = len(d)
        ac, bc, cc, dc = map(np.array, (a, b, c, d))
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
    def _fill_coeffs_for_linear(a: np.array, b: np.array, c: np.array, d: np.array, prev_ts: np.array,
                                config: EqConfig):
        raise NotImplementedError

    @classmethod
    def next_ts(cls, prev_ts: np.array, config: EqConfig) -> np.array:
        a, c = np.zeros(config.num_points - 1, dtype=np.float64), np.zeros(config.num_points - 1, dtype=np.float64)
        b, d = np.zeros(config.num_points, dtype=np.float64), np.zeros(config.num_points, dtype=np.float64)
        cls._fill_coeffs_for_linear(a, b, c, d, prev_ts, config)
        return cls._solve_tridiagonal_linear(a, b, c, d)


class ImplicitAgainst(ImplicitMethod):
    @staticmethod
    def _fill_coeffs_for_linear(a: np.array, b: np.array, c: np.array, d: np.array, prev_ts: np.array,
                                config: EqConfig):
        s = config.s
        r = config.r
        for i in range(config.num_points):
            b[i] = 1 + s + 2 * r
            d[i] = prev_ts[i]
        d[0] += config.left_border(prev_ts) * (r + s)
        d[config.num_points - 1] += config.right_border(prev_ts) * r
        for i in range(config.num_points - 1):
            a[i] = - r - s
            c[i] = -r
        return ImplicitMethod._solve_tridiagonal_linear(a, b, c, d)


class ImplicitDown(ImplicitMethod):
    @staticmethod
    def _fill_coeffs_for_linear(a: np.array, b: np.array, c: np.array, d: np.array, prev_ts: np.array,
                                config: EqConfig):
        s = config.s
        r = config.r
        for i in range(config.num_points):
            b[i] = 1 - s + 2 * r
            d[i] = prev_ts[i]
        d[0] += config.left_border(prev_ts) * r
        d[config.num_points - 1] -= config.right_border(prev_ts) * (s - r)

        for i in range(config.num_points - 1):
            a[i] = -r
            c[i] = s - r

        return ImplicitMethod._solve_tridiagonal_linear(a, b, c, d)
