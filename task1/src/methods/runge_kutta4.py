from methods.common import SystemConfig, Point


def _one_comp_step(df, p, dt, changer):
    k1 = df(p)
    k2 = df(changer(p, 0.5 * dt, 0.5 * dt * k1))
    k3 = df(changer(p, 0.5 * dt, 0.5 * dt * k2))
    k4 = df(changer(p, dt, dt * k3))
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * dt


def x_changer(p, dt, dx):
    return Point(x=p.x + dx, y=p.y     , z=p.z     , t=p.t + dt)


def y_changer(p, dt, dy):
    return Point(x=p.x     , y=p.y + dy, z=p.z     , t=p.t + dt)


def z_changer(p, dt, dz):
    return Point(x=p.x      , y=p.y    , z=p.z + dz, t=p.t + dt)


def solve_rk4(x0, y0, z0, t0, config=SystemConfig()):
    p = Point(x0, y0, z0, t0)
    pts = [p]

    while p.t < config.max_t:
        p = Point(x=p.x + _one_comp_step(config.dx_dt, p, config.dt, x_changer),
                  y=p.y + _one_comp_step(config.dy_dt, p, config.dt, y_changer),
                  z=p.z + _one_comp_step(config.dz_dt, p, config.dt, z_changer),
                  t=p.t + config.dt)
        pts.append(p)
    return pts
