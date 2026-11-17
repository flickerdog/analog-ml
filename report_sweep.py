import sys
from pathlib import Path

import pandas as pd

from src.specs import classify, spec_text, validity_text
from src.tables import readable, save_report, stats

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ota5t_sobol.csv")
df = pd.read_csv(CSV)
df["class"] = classify(df)
valid = df[df["class"] != "invalid"]
passing = df[df["class"] == "passes"]
cols = ["w1", "l1", "w3", "l3", "itail", "gain_db", "gbw_hz", "pm_deg",
        "power_w", "sat_margin", "v_tail"]

summary = pd.DataFrame(
    {"value": [len(df), int(df["gain_db"].notna().sum()), len(valid), len(passing),
               f"{len(passing) / len(df):.1%}"]},
    index=["points", "simulated without error", "valid designs",
           "pass all specs", "fraction passing"],
)

save_report(
    f"{CSV.name}: {spec_text()}  |  {validity_text()}",
    [
        ("Summary", summary),
        ("Statistics, valid designs", stats(valid[cols])),
        ("Statistics, passing designs", stats(passing[cols])),
        ("10 lowest-power passing designs",
         readable(passing.nsmallest(10, "power_w")[cols])),
        ("10 highest-gain passing designs",
         readable(passing.nlargest(10, "gain_db")[cols])),
    ],
    Path("reports") / f"{CSV.stem}.html",
)