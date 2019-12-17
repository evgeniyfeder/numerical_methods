import attr
import math


@attr.s(auto_attribs=True, str=False)
class SystemConfig:
    dt: float
    dz: float
    max_z: float
    k: int
    alpha: float
    num_iter: int

    #iter parametres

    @property
    def num_points(self) -> int:
        return int(self.max_z / self.dz)

    R: float = 8.314
    K: int = 1.6 * 10 ** 6  # 1 / сек
    E: int = 8 * 10 ** 4  # Дж / моль
    Q: int = 7 * 10 ** 5  # Дж / кг
    E_0: int = 293  # K
    rho: int = 830  # кг / м^3
    C: int = 1990  # Дж / кг * К

    Lambda: float = 0.13  # Вт / м * К
    D: float = 8 * 10 ** (-12)  # м^2 / сек

    T_0: int = 293

    @property
    def kappa(self):
        return self.Lambda / (self.rho * self.C)

    @property
    def dT(self):
        return self.Q / self.C

    @property
    def T_m(self):
        return self.T_0 + self.dT

    @property
    def betta(self):
        return self.R * self.T_m / self.E

    @property
    def sigma(self):
        return self.R * (self.T_m ** 2) / (self.E * self.dT)

    def W(self, x, t):
        return -self.k * (x ** self.alpha) * math.exp(-self.E / (self.R * t))

    @property
    def U(self):
        return (2 * self.k * self.Lambda) / (self.Q * self.rho * self.dT) \
               * (self.T_0 / self.T_m) \
               * (self.R * self.T_m ** 2 / self.E) ** 2 \
               * math.exp(-self.E / (self.R * self.T_m))
    
    def __str__(self) -> str:
        res = "System Config:\n"
        for at in attr.fields(type(self)):
            res += f'{at.name}={getattr(self, at.name)}\n'
        for prop in ['num_points', 'kappa', 'dT', 'T_m', 'betta', 'sigma', 'U']:
            res += f'{prop}={getattr(self, prop)}\n'
        return res
        
