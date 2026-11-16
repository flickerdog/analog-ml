import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.ota5t import evaluate

# (min, max), sampled uniformly in log space
RANGES = {
    "w1": (7, 50),            # um
    "l1": (0.15, 2.0),        # um
    "w3": (7, 50),            # um
    "l3": (0.15, 2.0),        # um
    "itail": (5e-6, 100e-6),  # A
}


def sample(m=11, seed=0):
    """2**m quasi-random points covering the 5-D design space evenly."""
    sampler = qmc.Sobol(d=len(RANGES), scramble=True, seed=seed)
    u = sampler.random_base2(m=m)                  # points in [0, 1)^5
    lo = np.log10([r[0] for r in RANGES.values()])
    hi = np.log10([r[1] for r in RANGES.values()])
    return pd.DataFrame(10 ** (lo + u * (hi - lo)), columns=list(RANGES))


def run_point(row):
    try:
        return evaluate(**row)
    except Exception as e:                         # a failed sim must not kill the sweep
        return {"error": str(e)[:200]}


if __name__ == "__main__":
    df = sample(m=11)                              # 2048 points
    rows = df.to_dict("records")
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        results = list(pool.map(run_point, rows, chunksize=16))
    out = pd.concat([df, pd.DataFrame(results)], axis=1)
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/ota5t_sobol.csv", index=False)
    print(out.describe().T)
    print(f"{len(out)} points in {time.perf_counter() - t0:.0f} s")