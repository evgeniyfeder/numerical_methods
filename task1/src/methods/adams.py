from methods.common import SystemConfig, Point
from methods.runge_kutta4 import solve_rk4


def one_comp_step_explicit(df, pts):
    return 1 / 24 * (55 * df(pts[-1]) - 59 * df(pts[-2]) + 37 * df(pts[-3]) - 9 * df(pts[-4]))


def solve_adams_4_explicit(x0, y0, z0, t0, config=SystemConfig()):
    pts = solve_rk4(x0, y0, z0, t0, config)[:4]

    p = pts[:-1]
    while p.t < config.max_t:
        p = Point(x=p.x + one_comp_step_explicit(config.dx_dt, pts),
                  y=p.y + one_comp_step_explicit(config.dy_dt, pts),
                  z=p.z + one_comp_step_explicit(config.dz_dt, pts),
                  t=p.t + config.dt)
        pts.append(p)
    return pts


def one_comp_step_implicit(df, pts, f_next):
    return 1 / 24 * (9 * df(f_next) + 19 * df(pts[-1]) - 5 * df(pts[-2]) + df(pts[-3]))


def solve_adams_4_implicit(x0, y0, z0, t0, config=SystemConfig()):
    pts = solve_rk4(x0, y0, z0, t0, config)[:4]

    p = pts[:-1]
    while p.t < config.max_t:
        f_n = Point(x=p.x + one_comp_step_explicit(config.dx_dt, pts),
                    y=p.y + one_comp_step_explicit(config.dy_dt, pts),
                    z=p.z + one_comp_step_explicit(config.dz_dt, pts),
                    t=p.t + config.dt)

        p = Point(x=p.x + one_comp_step_implicit(config.dx_dt, pts, f_n),
                  y=p.y + one_comp_step_implicit(config.dy_dt, pts, f_n),
                  z=p.z + one_comp_step_implicit(config.dz_dt, pts, f_n),
                  t=p.t + config.dt)
        pts.append(p)
    return pts
