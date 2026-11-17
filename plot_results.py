from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.plotting import log_axis, setup_style
from src.specs import SPECS, classify, spec_text, validity_text
from src.specs import LOAD_CAP_PF

setup_style()
Path("reports").mkdir(exist_ok=True)

df = pd.read_csv("data/ota5t_sobol.csv")
df["class"] = classify(df)
if "cl_pf" in df and abs(df["cl_pf"] - LOAD_CAP_PF).max() > 1e-9:
    print("WARNING: data simulated with a different load cap than specs.yaml")
df["itail_uA"] = df.itail * 1e6
df["gbw_MHz"] = df.gbw_hz / 1e6
df["power_uW"] = df.power_w * 1e6
df["jd_in_uA"] = df.itail_uA / 2 * df.l1 / df.w1       # Id/(W/L), input pair
df["jd_load_uA"] = df.itail_uA / 2 * df.l3 / df.w3     # Id/(W/L), load

LABEL = {"w1": "W1 (µm)", "l1": "L1 (µm)", "w3": "W3 (µm)", "l3": "L3 (µm)",
         "itail_uA": "Itail (µA)", "gbw_MHz": "GBW (MHz)", "power_uW": "Power (µW)",
         "jd_in_uA": "Input pair Id/(W/L) (µA)", "jd_load_uA": "Load Id/(W/L) (µA)"}
TICKS = {"w1": [7, 10, 20, 30, 50], "w3": [7, 10, 20, 30, 50],
         "l1": [0.15, 0.2, 0.3, 0.5, 1, 2], "l3": [0.15, 0.2, 0.3, 0.5, 1, 2],
         "itail_uA": [0.05, 0.1, 0.5, 1, 5, 10, 50, 100], "gbw_MHz": [0.1, 1, 10, 100],
         "power_uW": [0.1, 1, 10, 100],
         "jd_in_uA": [1e-4, 1e-3, 0.01, 0.1, 1, 10], "jd_load_uA": [1e-4, 1e-3, 0.01, 0.1, 1, 10]}

COLOR = {"invalid": "0.7", "fails spec": "tab:orange", "passes": "tab:blue"}
TEXT = {"invalid": "invalid", "fails spec": "valid, misses a spec",
        "passes": "passes all specs"}
dots = [Line2D([], [], marker="o", ls="", color=COLOR[k], label=TEXT[k], markersize=8)
        for k in ("fails spec", "passes")]
patches = [Patch(color=COLOR[k], label=TEXT[k]) for k in ("passes", "fails spec", "invalid")]

n_pass = int((df["class"] == "passes").sum())
n_invalid = int((df["class"] == "invalid").sum())
title = (f"{n_pass} of {len(df)} designs pass, {n_invalid} invalid\n"
         f"specs: {spec_text()}   |   {validity_text()}")

# ---- Image 1: performance space (valid designs only) --------------------
val = df[df["class"] != "invalid"]
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
Y = [("gain_db", "DC gain (dB)"), ("pm_deg", "Phase margin (°)"), ("power_uW", "Power (µW)")]
for a, (yc, ylab) in zip([axs[0, 0], axs[0, 1], axs[1, 0]], Y):
    for k in ("fails spec", "passes"):                 # passing designs drawn last, on top
        d = val[val["class"] == k]
        a.scatter(d.gbw_MHz, d[yc], s=9, color=COLOR[k], alpha=0.6, linewidths=0)
    log_axis(a, "x", TICKS["gbw_MHz"])
    if "gbw_hz" in SPECS:
        a.axvline(SPECS["gbw_hz"] / 1e6, color="k", ls="--", lw=1)
    a.set_xlabel(LABEL["gbw_MHz"])
    a.set_ylabel(ylab)
if "gain_db" in SPECS:
    axs[0, 0].axhline(SPECS["gain_db"], color="k", ls="--", lw=1)
if "pm_deg" in SPECS:
    axs[0, 1].axhline(SPECS["pm_deg"], color="k", ls="--", lw=1)
log_axis(axs[1, 0], "y", TICKS["power_uW"])

# Bottom right: which spec costs the most designs (among valid ones)
NAMES = {"gain_db": "gain", "pm_deg": "phase margin", "gbw_hz": "GBW"}
fails = {NAMES[c]: int((val[c] < m).sum()) for c, m in SPECS.items()}
b = axs[1, 1]
b.barh(list(fails), list(fails.values()), color="tab:orange")
b.invert_yaxis()
b.grid(False)
b.set_xlabel("valid designs failing this spec")
for y, v in enumerate(fails.values()):
    b.text(v, y, f" {v}", va="center")
fig.legend(handles=dots, loc="lower center", ncol=2)
fig.suptitle(title + "\n(invalid designs not shown)", fontsize=12)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig("reports/performance_space.png")

# ---- Image 2: fraction of designs passing, along each variable ----------
NB = 10
fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
for a, c in zip(axs.flat, ["w1", "l1", "itail_uA", "w3", "l3"]):
    edges = np.geomspace(df[c].min(), df[c].max() * 1.0001, NB + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.digitize(df[c], edges) - 1, 0, NB - 1)    # bin of each design
    counts = np.array([[((df["class"] == k) & (idx == i)).sum() for i in range(NB)]
                       for k in ("passes", "fails spec", "invalid")], dtype=float)
    a.stackplot(centers, counts / counts.sum(axis=0),
                colors=[COLOR[k] for k in ("passes", "fails spec", "invalid")])
    log_axis(a, "x", TICKS[c])
    a.set_xlim(centers[0], centers[-1])
    a.set_ylim(0, 1)
    a.set_xlabel(LABEL[c])
for a in axs[:, 0]:
    a.set_ylabel("fraction of designs")
axs[1, 2].axis("off")
axs[1, 2].legend(handles=patches, loc="center")
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("reports/passrate_vs_variable.png")

# ---- Image 3: pass rate over pairs of variables --------------------------
PAIRS = [("l1", "l3"), ("l3", "jd_in_uA"), ("jd_in_uA", "jd_load_uA"), ("w1", "itail_uA")]
passes = df["class"] == "passes"
fig, axs = plt.subplots(2, 2, figsize=(13, 10))
for a, (xc, yc) in zip(axs.flat, PAIRS):
    xe = np.geomspace(df[xc].min(), df[xc].max() * 1.0001, 9)
    ye = np.geomspace(df[yc].min(), df[yc].max() * 1.0001, 9)
    total, _, _ = np.histogram2d(df[xc], df[yc], bins=[xe, ye])
    won, _, _ = np.histogram2d(df.loc[passes, xc], df.loc[passes, yc], bins=[xe, ye])
    frac = np.where(total > 0, won / np.maximum(total, 1), np.nan)
    m = a.pcolormesh(xe, ye, frac.T, vmin=0, vmax=1, cmap="Blues")
    log_axis(a, "x", TICKS[xc])
    log_axis(a, "y", TICKS[yc])
    a.grid(False)
    a.set_xlabel(LABEL[xc])
    a.set_ylabel(LABEL[yc])
    fig.colorbar(m, ax=a, label="fraction passing")
fig.suptitle(title, fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig("reports/passrate_pairs.png")

# ---- Image 4: DC gain versus minimum channel length ---------------------
min_length = df[["l1", "l3"]].min(axis=1)
input_current_density = df.itail / 2 * df.l1 / df.w1
fig, ax = plt.subplots(figsize=(8.5, 5.5))
points = ax.scatter(
    min_length, df.gain_db, c=np.log10(input_current_density),
    cmap="viridis", s=12, alpha=0.8, linewidths=0,
)
log_axis(ax, "x", [0.2, 0.3, 0.5, 1, 1.5, 2])
ax.set_xlabel("Minimum channel length, min(L1, L3) (µm)")
ax.set_ylabel("DC gain (dB)")
ax.set_title("DC gain versus minimum channel length")
if "gain_db" in SPECS:
    ax.axhline(SPECS["gain_db"], color="black", ls="--", lw=1, label="gain requirement")
    ax.legend()
fig.colorbar(
    points, ax=ax, label="log10 input-pair current density, Id/(W/L) (A)"
)
fig.tight_layout()
fig.savefig("reports/gain_vs_l.png")
print("saved 4 images in reports/")