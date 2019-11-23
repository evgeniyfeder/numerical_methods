import attr


@attr.s(auto_attribs=True)
class EqConfig:
    v: float
    kappa: float
    dt: float
    dx: float

    @property
    def r(self):
        return self.v * self.dt / self.dx

    @property
    def s(self):
        return self.kappa * self.dt / (self.dx ** 2)

    a: float
    b: float

    num_iter: int

    @property
    def num_points(self):
        return int((self.b - self.a) / self.dt)

    def left_border(self, Ts):
        return 0

    def right_border(self, Ts):
        return 0