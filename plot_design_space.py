from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.specs import SPECS, classify

df = pd.read_csv("data/ota5t_sobol.csv")
df["class"] = classify(df)
df["itail_uA"] = df.itail * 1e6
df["gbw_MHz"] = df.gbw_hz / 1e6
df["power_uW"] = df.power_w * 1e6

# Drawn back to front, so passing designs end up on top
STYLE = {
    "invalid":    dict(color="0.75",       zorder=1, label="invalid (device out of saturation)"),
    "fails spec": dict(color="tab:orange", zorder=2, label="valid, misses a spec"),
    "passes":     dict(color="tab:blue",   zorder=3, label="passes all specs"),
}

print(df["class"].value_counts().to_string())
ok = df[df["class"] != "invalid"]
for col, minimum in SPECS.items():
    print(f"valid designs failing {col} >= {minimum:g}: {(ok[col] < minimum).sum()}")

spec_text = (f"specs: gain ≥ {SPECS['gain_db']:g} dB, PM ≥ {SPECS['pm_deg']:g}°, "
             f"GBW ≥ {SPECS['gbw_hz'] / 1e6:g} MHz")
legend = [Line2D([], [], marker="o", ls="", color=s["color"], label=s["label"])
          for s in STYLE.values()]
Path("reports").mkdir(exist_ok=True)

# ---- Figure 1: the five design variables, every pair --------------------
VARS = [("w1", "W1 (µm)"), ("l1", "L1 (µm)"), ("w3", "W3 (µm)"),
        ("l3", "L3 (µm)"), ("itail_uA", "Itail (µA)")]
lims = {c: (df[c].min() * 0.9, df[c].max() * 1.1) for c, _ in VARS}
n = len(VARS)

fig, axes = plt.subplots(n, n, figsize=(13, 13))
for i, (yc, ylab) in enumerate(VARS):
    for j, (xc, xlab) in enumerate(VARS):
        ax = axes[i, j]
        if j > i:                                   # upper triangle left empty
            ax.axis("off")
            continue
        if i == j:                                  # diagonal: stacked histogram
            bins = np.geomspace(*lims[xc], 16)
            ax.hist([df.loc[df["class"] == k, xc] for k in STYLE], bins=bins,
                    stacked=True, color=[s["color"] for s in STYLE.values()])
        else:                                       # lower triangle: scatter
            for k, s in STYLE.items():
                d = df[df["class"] == k]
                ax.scatter(d[xc], d[yc], s=8, color=s["color"], zorder=s["zorder"],
                           alpha=0.6, linewidths=0)
            ax.set_yscale("log")
            ax.set_ylim(*lims[yc])
        ax.set_xscale("log")
        ax.set_xlim(*lims[xc])
        ax.tick_params(labelbottom=(i == n - 1), labelleft=(j == 0 and i > 0))
        if i == n - 1:
            ax.set_xlabel(xlab)
        if j == 0 and i > 0:
            ax.set_ylabel(ylab)

fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.97, 0.95))
fig.suptitle(f"Design space, {len(df)} designs, {spec_text}", y=0.99)
fig.savefig("reports/design_space.png", dpi=130, bbox_inches="tight")

# ---- Figure 2: performance space with the spec limits -------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
for k, s in STYLE.items():
    d = df[df["class"] == k]
    kw = dict(s=10, color=s["color"], zorder=s["zorder"], alpha=0.7, linewidths=0)
    ax[0].scatter(d.gbw_MHz, d.gain_db, **kw)
    ax[1].scatter(d.gbw_MHz, d.power_uW, **kw)
gbw_min = SPECS["gbw_hz"] / 1e6
ax[0].axhline(SPECS["gain_db"], color="k", ls="--", lw=1)
ax[0].axvline(gbw_min, color="k", ls="--", lw=1)
ax[1].axvline(gbw_min, color="k", ls="--", lw=1)
ax[0].set_xlabel("GBW (MHz)");  ax[0].set_ylabel("DC gain (dB)")
ax[1].set_xlabel("GBW (MHz)");  ax[1].set_ylabel("Power (µW)")
for a in ax:
    a.set_xscale("log")
ax[1].set_yscale("log")
ax[0].legend(handles=legend, loc="lower right", fontsize=8)
fig.suptitle(spec_text)
fig.tight_layout()
fig.savefig("reports/performance_space.png", dpi=130)
print("saved reports/design_space.png and reports/performance_space.png")