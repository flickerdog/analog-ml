import time
from src.ngspice_backend import evaluate

t0 = time.perf_counter()
print(evaluate(w=1, l=0.15))
print(f"{time.perf_counter() - t0:.2f} s")