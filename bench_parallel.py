import time
from concurrent.futures import ProcessPoolExecutor

from sweep_ota import run_point, sample

if __name__ == "__main__":
    rows = sample(m=6).to_dict("records")        # 64 points

    t0 = time.perf_counter()
    for r in rows:
        run_point(r)
    print(f"serial      {time.perf_counter() - t0:6.1f} s")

    for n in (2, 4, 8):
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n) as pool:
            list(pool.map(run_point, rows, chunksize=4))
        print(f"{n} workers   {time.perf_counter() - t0:6.1f} s")