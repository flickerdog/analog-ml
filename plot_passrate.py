from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.specs import classify

df = pd.read_csv("data/ota5t_sobol.csv")
df["class"] = classify(df)
df["itail_uA"] = df.itail * 1e6
# Current density Id/(W/L) of each pair: sets how strongly it is inverted
df["jd_in_uA"] = df.itail_uA / 2 * df.l1 / df.w1
df["jd_load_uA"] = df.itail_uA / 2 * df.l3 / df.w3
Path("reports").mkdir(exist_ok=True)

CLASSES = ["passes", "fails spec", "invalid"]
COLORS = ["tab:blue", "tab:orange", "0.75"]

# ---- Figure 1: what fraction of designs pass, along each variable -------
VARS = [("w1", "W1 (µm)"), ("l1", "L1 (µm)"), ("w3", "W3 (µm)"),
        ("l3", "L3 (µm)"), ("itail_uA", "Itail (µA)")]
NB = 12
fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
for ax, (c, label) in zip(axes, VARS):
    edges = np.geomspace(df[c].min(), df[c].max() * 1.0001, NB + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.digitize(df[c], edges) - 1, 0, NB - 1)   # bin of each design
    frac = np.array([[((df["class"] == k) & (idx == b)).sum() for b in range(NB)]
                     for k in CLASSES], dtype=float)
    frac /= frac.sum(axis=0)                                  # fractions add up to 1
    ax.stackplot(centers, frac, colors=COLORS, labels=CLASSES)
    ax.set_xscale("log")
    ax.set_xlabel(label)
axes[0].set_ylabel("fraction of designs")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("reports/passrate_vs_variable.png", dpi=130)

# ---- Figure 2: pass rate over pairs of variables ------------------------
PAIRS = [
    ("l1", "L1 (µm)", "l3", "L3 (µm)"),
    ("l3", "L3 (µm)", "jd_in_uA", "Input pair Id/(W/L) (µA)"),
    ("jd_in_uA", "Input pair Id/(W/L) (µA)", "jd_load_uA", "Load Id/(W/L) (µA)"),
    ("w1", "W1 (µm)", "itail_uA", "Itail (µA)"),
]
passes = df["class"] == "passes"
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (xc, xl, yc, yl) in zip(axes, PAIRS):
    xe = np.geomspace(df[xc].min(), df[xc].max() * 1.0001, 9)
    ye = np.geomspace(df[yc].min(), df[yc].max() * 1.0001, 9)
    total, _, _ = np.histogram2d(df[xc], df[yc], bins=[xe, ye])
    n_pass, _, _ = np.histogram2d(df.loc[passes, xc], df.loc[passes, yc], bins=[xe, ye])
    frac = np.where(total > 0, n_pass / np.maximum(total, 1), np.nan)
    m = ax.pcolormesh(xe, ye, frac.T, vmin=0, vmax=1, cmap="Blues")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    fig.colorbar(m, ax=ax, label="fraction passing")
fig.tight_layout()
fig.savefig("reports/passrate_pairs.png", dpi=130)
print("saved reports/passrate_vs_variable.png and reports/passrate_pairs.png")