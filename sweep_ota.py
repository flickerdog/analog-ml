import os
import time

import pandas as pd
from scipy.stats import qmc

from src.ota5t import evaluate
from src.specs import LOAD_CAP, LOAD_CAP_PF

# (min, max), sampled uniformly in log space
RANGES = {
    "w1": (7, 50),               # um
    "l1": (0.15, 2.0),           # um
    "w3": (7, 50),               # um
    "l3": (0.15, 2.0),           # um
    "itail": (100e-9, 100e-6),    # A
}


def sample(m=12, seed=0):
    """2**m quasi-random points covering the design space evenly (in log space)."""
    import numpy as np
    sampler = qmc.Sobol(d=len(RANGES), scramble=True, seed=seed)
    u = sampler.random_base2(m=m)                  # points in [0, 1)^5
    lo = np.log10([r[0] for r in RANGES.values()])
    hi = np.log10([r[1] for r in RANGES.values()])
    return pd.DataFrame(10 ** (lo + u * (hi - lo)), columns=list(RANGES))


def run_point(row):
    err = ""
    for _ in range(2):                             # one retry before giving up
        try:
            return evaluate(**row, cl=LOAD_CAP)
        except Exception as e:
            err = str(e)[:200]
    return {"error": err}


if __name__ == "__main__":
    df = sample(m=12)                              # 4096 points
    df["cl_pf"] = LOAD_CAP_PF                      # the data records its own test condition
    rows = df[list(RANGES)].to_dict("records")
    t0 = time.perf_counter()
    results = []
    for i, row in enumerate(rows):
        results.append(run_point(row))
        if i % 200 == 0:
            print(i, f"{time.perf_counter() - t0:.0f} s", flush=True)
    out = pd.concat([df, pd.DataFrame(results)], axis=1)
    os.makedirs("data", exist_ok=True)
    out.to_csv("data/ota5t_sobol.csv", index=False)
    n_err = out["error"].notna().sum() if "error" in out else 0
    print(f"{len(out)} points, {n_err} failed, {time.perf_counter() - t0:.0f} s")