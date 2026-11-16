import numpy as np
from src.ngspice_backend import evaluate


def smoothness(label, xs, make):
    res = [make(x) for x in xs]
    for key in ("id", "gm_over_id"):
        y = np.array([r[key] for r in res])
        d = np.abs(np.diff(np.log(y)))        # relative step between neighbours
        k = d.argmax()
        print(f"{label} {key:10s} largest {100*d[k]:.2f} %  "
              f"median {100*np.median(d):.2f} %  ratio {d[k]/np.median(d):.1f}  "
              f"near {xs[k]:.3f} um")


ls = np.geomspace(0.15, 2.0, 300)
ws = np.geomspace(1.0, 50.0, 300)
smoothness("L scan (W=10)  ", ls, lambda l: evaluate(w=10, l=l))
smoothness("W scan (L=0.6) ", ws, lambda w: evaluate(w=w, l=0.6))