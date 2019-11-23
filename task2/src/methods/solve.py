import pandas as pd
import eq_config


def solve(begin_ts, method, config=eq_config.EqConfig()):
    dfs = []

    df = pd.DataFrame(columns=["time", "x", "T"])
    ts = begin_ts
    for i in range(config.num_points):
        df.append({"time": 0, "x": config.a + config.dx * i, "T": begin_ts[i]})
    dfs.append(df)

    for i in range(1, config.num_iter + 1):
        t = config.dt * i
        ts = method(ts, config)

        df = pd.DataFrame(columns=["time", "x", "T"])
        for j in range(config.num_points):
            df.append({"time": t, "x": config.a + config.dx * i, "T": begin_ts[i]})
        dfs.append(df)

    return dfs
