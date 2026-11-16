import time
from src.ota5t import evaluate

t0 = time.perf_counter()
print(evaluate(w1=10, l1=0.5, w3=10, l3=0.5))
print(f"{time.perf_counter() - t0:.2f} s")