import attr

@attr.s(auto_attribs=True)
class SystemConfig:
    sigma:float = 10
    b: float = 8/3

    dx_dt = lambda t, x, y, z, sigma, b, r: sigma * (y - x)
    dy_dt = lambda t, x, y, z, sigma, b, r: x * (r - z) - y
    dz_dt = lambda t, x, y, z, sigma, b, r: x * y - b * z
