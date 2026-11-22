import argparse
import os
import time

import pandas as pd
from scipy.stats import qmc

from src.topology import load_topology, topology_names
from src.specs import LOAD_CAP, LOAD_CAP_PF

# (min, max), sampled uniformly in log space
DEFAULT_TOPOLOGY = load_topology("ota5t")
RANGES = DEFAULT_TOPOLOGY.RANGES


def sample(m=12, seed=0, ranges=RANGES):
    """2**m quasi-random points covering the design space evenly (in log space)."""
    import numpy as np
    sampler = qmc.Sobol(d=len(ranges), scramble=True, seed=seed)
    u = sampler.random_base2(m=m)                  # points in [0, 1)^5
    lo = np.log10([r[0] for r in ranges.values()])
    hi = np.log10([r[1] for r in ranges.values()])
    return pd.DataFrame(10 ** (lo + u * (hi - lo)), columns=list(ranges))


def run_point(row, topology=DEFAULT_TOPOLOGY):
    err = ""
    for _ in range(2):                             # one retry before giving up
        try:
            return topology.evaluate(**row, cl=LOAD_CAP)
        except Exception as e:
            err = str(e)[:200]
    return {"error": err}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", choices=topology_names(), default="ota5t")
    parser.add_argument("--m", type=int, default=12, help="sample 2**m designs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.m < 1:
        parser.error("--m must be at least 1")
    topology = load_topology(args.topology)
    ranges = topology.RANGES
    output = args.output or str(topology.DATA_FILE)

    df = sample(m=args.m, seed=args.seed, ranges=ranges)
    df["cl_pf"] = LOAD_CAP_PF                      # the data records its own test condition
    rows = df[list(ranges)].to_dict("records")
    t0 = time.perf_counter()
    results = []
    for i, row in enumerate(rows):
        results.append(run_point(row, topology))
        if i % 200 == 0:
            print(i, f"{time.perf_counter() - t0:.0f} s", flush=True)
    out = pd.concat([df, pd.DataFrame(results)], axis=1)
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    out.to_csv(output, index=False)
    n_err = out["error"].notna().sum() if "error" in out else 0
    print(f"{topology.__name__}: {len(out)} points, {n_err} failed, "
          f"{time.perf_counter() - t0:.0f} s -> {output}")