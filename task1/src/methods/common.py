import attr


@attr.s(auto_attribs=True)
class Point:
    x: float
    y: float
    z: float
    t: float


@attr.s(auto_attribs=True)
class SystemConfig:
    sigma: float = 10
    b: float = 8 / 3
    r: float = 0.5

    dt: float = 0.001
    max_t: float = 20

    def dx_dt(self, p):
        return self.sigma * (p.y - p.x)

    def dy_dt(self, p):
        return p.x * (self.r - p.z) - p.y

    def dz_dt(self, p):
        return p.x * p.y - self.b * p.z
