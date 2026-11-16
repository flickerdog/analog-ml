import pandas as pd

from src.ota5t import evaluate

df = pd.read_csv("data/ota5t_sobol.csv")
cols = ["w1", "l1", "w3", "l3", "itail"]
todo = df.index[df["error"].notna()]

for i in todo:
    try:
        res = evaluate(**df.loc[i, cols].to_dict())
    except Exception as e:
        df.loc[i, "error"] = str(e)[:200]
        continue
    for k, v in res.items():
        df.loc[i, k] = v
    df.loc[i, "error"] = None

df.to_csv("data/ota5t_sobol.csv", index=False)
print(len(todo), "rows retried;", df["error"].notna().sum(), "still failing")