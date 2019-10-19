from methods.common import SystemConfig, Point


def _one_step(p_n: Point, dp_n: Point, config: SystemConfig):
    return Point(x=p_n.x + config.dt * config.dx_dt(dp_n),
                 y=p_n.y + config.dt * config.dy_dt(dp_n),
                 z=p_n.z + config.dt * config.dz_dt(dp_n),
                 t=p_n.t + config.dt)


def solve_euler_explicit(x0, y0, z0, t0, config=SystemConfig()):
    p = Point(x0, y0, z0, t0)

    pts = [p]
    while p.t < config.max_t:
        p = _one_step(p, p, config)
        pts.append(p)

    return pts


def solve_euler_implicit(x0, y0, z0, t0, config=SystemConfig()):
    p = Point(x0, y0, z0, t0)

    pts = [p]
    while p.t < config.max_t:
        p_next = _one_step(p, p, config)

        p = _one_step(p, p_next, config)
        p.t = p_next.t

        pts.append(p)
    return pts
