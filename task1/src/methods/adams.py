from methods.common import SystemConfig, Point
from methods.runge_kutta4 import solve_rk4


def one_comp_step(df, pts):
    return 1 / 24 * (55 * df(pts[-1]) - 59 * df(pts[-2]) + 37 * df(pts[-3]) - 9 * df(pts[-4]))


def solve_adams_4_explicit(x0, y0, z0, t0, config=SystemConfig()):
    elem_count = 4
    pts = solve_rk4(x0, y0, z0, t0, config)[:elem_count]

    p = pts[:-1]
    while p.t < config.max_t:
        p = Point(x=p.x + one_comp_step(config.dx_dt, pts),
                       y=p.y + one_comp_step(config.dy_dt, pts),
                       z=p.z + one_comp_step(config.dz_dt, pts),
                       t=p.t + config.dt)
        pts.append(p)
    return pts

