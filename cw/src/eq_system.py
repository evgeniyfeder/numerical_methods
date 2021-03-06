import attr
import math
import numpy as np


@attr.s(auto_attribs=True, str=False)
class SystemConfig:
    dt: float
    dz: float
    max_z: float
    alpha: float

    # iter parametres

    @property
    def num_points(self) -> int:
        return int(self.max_z / self.dz)

    @property
    def xs(self) -> np.array:
        return np.linspace(0, self.max_z, self.num_points)

    R: float = 8.314
    K: float = 1.6 * 10 ** 6  # 1 / сек
    E: float = 8 * 10 ** 4  # Дж / моль
    Q: float = 7 * 10 ** 5  # Дж / кг
    T_0: float = 293  # K
    rho: float = 830  # кг / м^3
    C: float = 1990  # Дж / кг * К

    Lambda: float = 0.13  # Вт / м * К
    D: float = 8 * 10 ** (-12)   # м^2 / сек

    @property
    def kappa(self):
        return self.Lambda / (self.rho * self.C)

    @property
    def dT(self):
        return self.Q / self.C

    @property
    def T_m(self):
        return self.T_0 + self.dT
        # return 600

    @property
    def betta(self):
        return self.R * self.T_m / self.E

    @property
    def sigma(self):
        return self.R * (self.T_m ** 2) / (self.E * self.dT)

    def W(self, x, t):
        return -self.K * (x ** self.alpha) * math.exp(-self.E / (self.R * t))

    def MagicW(self, x, t):
        print(t)
        return -self.K * (x ** (self.alpha - 1)) * math.exp(-self.E / (self.R * t))

    @property
    def U(self):
        t = (2 * self.K * self.Lambda) \
            / (self.Q * self.rho * self.dT) \
            * (self.T_0 / self.T_m) \
            * (self.R * self.T_m ** 2 / self.E) ** 2 \
            * math.exp(-self.E / (self.R * self.T_m))
        return math.sqrt(t)

    def __str__(self) -> str:
        res = "System Config:\n"
        for at in attr.fields(type(self)):
            res += f'{at.name}={getattr(self, at.name)}\n'
        for prop in ['num_points', 'kappa', 'dT', 'T_m', 'betta', 'sigma', 'U']:
            res += f'{prop}={getattr(self, prop)}\n'
        return res
