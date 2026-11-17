import sys
from pathlib import Path

import pandas as pd

from src.tables import readable, save_report, stats

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "data/ota5t_sobol.csv")
df = pd.read_csv(CSV)
ok = df[df["valid"] == True]
cols = ["w1", "l1", "w3", "l3", "itail", "gain_db", "gbw_hz", "pm_deg",
        "power_w", "sat_margin", "v_tail"]

summary = pd.DataFrame(
    {"value": [len(df), int(df["gain_db"].notna().sum()), len(ok),
               f"{len(ok) / len(df):.1%}"]},
    index=["points", "simulated without error", "valid designs", "valid fraction"],
)

save_report(
    CSV.name,
    [
        ("Summary", summary),
        ("Statistics, all simulated points", stats(df[cols])),
        ("Statistics, valid designs only", stats(ok[cols])),
        ("10 lowest-gain valid designs", readable(ok.nsmallest(10, "gain_db")[cols])),
        ("10 highest-GBW valid designs", readable(ok.nlargest(10, "gbw_hz")[cols])),
    ],
    Path("reports") / f"{CSV.stem}.html",
)