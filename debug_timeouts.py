import time

import pandas as pd

import src.ngspice_backend as nb
from src.ota5t import evaluate

nb.TIMEOUT = 120                      # let the slow ones finish this time

df = pd.read_csv("data/ota5t_sobol.csv")
cols = ["w1", "l1", "w3", "l3", "itail"]
bad = df[df.error.notna()]
good = df[df.error.isna()]

print("failed points:");  print(bad[cols].describe().T[["min", "50%", "max"]])
print("good points:");    print(good[cols].describe().T[["min", "50%", "max"]])

for _, row in bad.head(5).iterrows():
    t0 = time.perf_counter()
    try:
        evaluate(**row[cols].to_dict())
        status = "ok"
    except Exception as e:
        status = str(e)[:80]
    print(f"{time.perf_counter() - t0:6.1f} s  {status}")