import numpy as np
from eq_system import SystemConfig


class ImplicitMethod:
    def __init__(self, begin_ts: np.array, begin_xs: np.array, config: SystemConfig):
        self.cur_ts = np.copy(begin_ts)
        self.cur_xs = np.copy(begin_xs)
        # buffers for temporary values
        self.a = np.zeros(config.num_points - 1, dtype=np.float64)
        self.c = np.zeros(config.num_points - 1, dtype=np.float64)
        self.b = np.zeros(config.num_points, dtype=np.float64)
        self.d = np.zeros(config.num_points, dtype=np.float64)
        self.config = config

    def _solve_tridiagonal_linear(self):
        nf = len(self.d)
        ac, bc, cc, dc = (self.a, self.b, self.c, self.d)
        for it in range(1, nf):
            mc = ac[it - 1] / bc[it - 1]
            bc[it] = bc[it] - mc * cc[it - 1]
            dc[it] = dc[it] - mc * dc[it - 1]
        xc = bc
        xc[-1] = dc[-1] / bc[-1]

        for il in range(nf - 2, -1, -1):
            xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

        return xc

    def _fill_abcd_for_linear_x(self):
        raise NotImplementedError

    def _fill_abcd_for_linear_t(self):
        raise NotImplementedError

    def _fill_abcd_zero(self):
        self.a = np.zeros(self.config.num_points - 1, dtype=np.float64)
        self.c = np.zeros(self.config.num_points - 1, dtype=np.float64)
        self.b = np.zeros(self.config.num_points, dtype=np.float64)
        self.d = np.zeros(self.config.num_points, dtype=np.float64)

    def __iter__(self):
        return self

    def __next__(self):
        prev_xs = self.cur_xs
        prev_ts = self.cur_ts

        self._fill_abcd_for_linear_x()
        np.copyto(self.cur_xs, self._solve_tridiagonal_linear())
        # assert not np.array_equal(prev_ts, self.cur_ts)

        self._fill_abcd_for_linear_t()
        np.copyto(self.cur_ts, self._solve_tridiagonal_linear())
        return prev_ts, prev_xs


class WImplicitMethod(ImplicitMethod):
    def _fill_abcd_for_linear_x(self):
        prev_xs = self.cur_xs
        prev_ts = self.cur_ts
        config = self.config

        # for i in range(config.num_points - 1):
        #     self.a[i] = -config.D / (config.dz ** 2)
        #     self.c[i] = -config.D / (config.dz ** 2)
        #
        # # for i in range(config.num_points):
        #     self.b[i] = 2 * config.D / (config.dz ** 2) + 1 / config.dt
        #     self.d[i] = config.W(prev_xs[i], prev_ts[i]) + prev_xs[i] / config.dt
        #
        self.a.fill(-config.D / (config.dz ** 2))
        self.c.fill(-config.D / (config.dz ** 2))
        self.b.fill(2 * config.D / (config.dz ** 2) + 1 / config.dt)
        self.d = np.vectorize(config.W)(prev_xs, prev_ts) + prev_xs / config.dt

        self.b[config.num_points - 1] = config.D / (config.dz ** 2) + 1 / config.dt

    def _fill_abcd_for_linear_t(self):
        prev_xs = self.cur_xs
        prev_ts = self.cur_ts
        config = self.config
        # for i in range(config.num_points - 1):
        #     self.a[i] = -config.kappa / (config.dz ** 2)
        #     self.c[i] = -config.kappa / (config.dz ** 2)
        #
        # for i in range(config.num_points):
        #     self.b[i] = 2 * config.kappa / (config.dz ** 2) + 1 / config.dt
        #     self.d[i] = -config.Q / config.C * config.W(prev_xs[i], prev_ts[i]) + prev_ts[i] / config.dt
        self.a.fill(-config.kappa / (config.dz ** 2))
        self.c.fill(-config.kappa / (config.dz ** 2))
        self.b.fill(2 * config.kappa / (config.dz ** 2) + 1 / config.dt)
        self.d = -config.Q / config.C * np.vectorize(config.W)(prev_xs, prev_ts) + prev_ts / config.dt

        self.d[0] += (config.kappa * config.T_m) / (config.dz ** 2)
        self.b[config.num_points - 1] = config.kappa / (config.dz ** 2) + 1 / config.dt


class MagicWImplicitMethod(ImplicitMethod):
    def _fill_abcd_for_linear_x(self):
        prev_xs = self.cur_xs
        prev_ts = self.cur_ts
        config = self.config
        for i in range(config.num_points - 1):
            self.a[i] = -config.D / (config.dz ** 2)
            self.c[i] = -config.D / (config.dz ** 2)

        for i in range(config.num_points):
            self.b[i] = 2 * config.D / (config.dz ** 2) + 1 / config.dt - config.MagicW(prev_xs[i], prev_ts[i])
            self.d[i] = prev_xs[i] / config.dt

    def _fill_abcd_for_linear_t(self):
        prev_xs = self.cur_xs
        prev_ts = self.cur_ts
        config = self.config
        for i in range(config.num_points - 1):
            self.a[i] = -config.kappa / (config.dz ** 2)
            self.c[i] = -config.kappa / (config.dz ** 2)

        for i in range(config.num_points):
            self.b[i] = 2 * config.kappa / (config.dz ** 2) \
                   + 1 / config.dt + config.Q / config.C * config.W(prev_xs[i], prev_ts[i])
            self.d[i] = prev_ts[i] / config.dt
