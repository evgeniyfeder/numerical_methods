from methods.common import SystemConfig, Point


def one_step(p: Point, config: SystemConfig):
    return Point(x=p.x + config.dt * config.dx_dt(p),
                 y=p.y + config.dt * config.dy_dt(p),
                 z=p.z + config.dt * config.dz_dt(p),
                 t=p.t + config.dt)


def solve_euler(x0, y0, z0, t0, config=SystemConfig()):
    p = Point(x0, y0, z0, t0)

    pts = [p]
    while p.t < config.max_t:
        p = one_step(p, config)
        pts.append(p)

    return pts
