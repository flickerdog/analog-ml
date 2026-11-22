"""Sweep complementary folded-cascode bias currents and voltages.

The generated circuits use zero-current output fixtures, with no explicit
0.9 V output-voltage source.
"""

import base64
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.complementary_ota import evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
Wn = 48.0
Ln = 0.19
Wp = 90.0
Lp = 0.19
CL = 1e-12


def run_point(name, itail=20e-6, vb1=1.30, vb2=1.05, vb3=0.66,
              fold_w=20.0, fold_l=0.15):
    result = evaluate(
        Wn, Ln, Wp, Lp, itail, CL,
        fold_w=fold_w, fold_l=fold_l,
        vb1=vb1, vb2=vb2, vb3=vb3,
    )
    return {
        "sweep": name,
        "itail": itail,
        "vb1": vb1,
        "vb2": vb2,
        "vb3": vb3,
        "fold_w": fold_w,
        "fold_l": fold_l,
        **result,
    }


def fig_data_uri(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def plot_oat(rows):
    df = pd.DataFrame(rows)
    figures = []
    for column, label in (
        ("vb1", "VB1 (V)"),
        ("vb2", "VB2 (V)"),
        ("vb3", "VB3 (V)"),
    ):
        subset = df[df["sweep"] == column]
        if subset.empty:
            continue
        x = subset[column]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, metric, ylabel in (
            (axes[0], "gain_db", "Gain (dB)"),
            (axes[1], "sat_margin", "Saturation margin (V)"),
            (axes[2], "mirror_ratio_error", "Mirror ratio error"),
        ):
            ax.plot(x, subset[metric], marker="o")
            ax.set_xlabel(label)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        figures.append(fig_data_uri(fig))
    return figures


def plot_heatmap(rows):
    df = pd.DataFrame(rows)
    subset = df[df["sweep"] == "vb1_vb3"]
    pivot = subset.pivot(index="vb3", columns="vb1", values="gain_db")
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.min(), pivot.index.max(),
        ],
        cmap="viridis",
    )
    ax.set_xlabel("VB1 (V)")
    ax.set_ylabel("VB3 (V)")
    ax.set_title("Gain sweep: VB1 vs VB3")
    fig.colorbar(image, ax=ax, label="Gain (dB)")
    return fig_data_uri(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()

    for value in np.arange(0.90, 1.701, 0.10):
        rows.append(run_point("vb1", vb1=float(value)))
    for value in np.arange(0.85, 1.151, 0.05):
        rows.append(run_point("vb2", vb2=float(value)))
    for value in np.arange(0.40, 0.801, 0.05):
        rows.append(run_point("vb3", vb3=float(value)))
    for vb1 in np.arange(1.00, 1.501, 0.10):
        for vb3 in np.arange(0.50, 0.701, 0.05):
            rows.append(run_point(
                "vb1_vb3",
                vb1=float(vb1),
                vb3=float(vb3),
            ))

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "complementary_bias_sweep.csv"
    df.to_csv(csv_path, index=False)

    oat_plots = plot_oat(rows)
    heatmap = plot_heatmap(rows)
    valid = df[df["valid"]]
    best = valid.sort_values("gain_db", ascending=False).iloc[0].to_dict() if not valid.empty else None
    center = df[
        (df["sweep"] == "vb1")
        & np.isclose(df["vb1"], 1.30)
        & np.isclose(df["vb2"], 1.05)
        & np.isclose(df["vb3"], 0.66)
    ].iloc[0].to_dict()

    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "points": len(df),
        "valid_points": len(valid),
        "center": center,
        "best_valid": best,
        "fixed_geometry": {"wn": Wn, "ln": Ln, "wp": Wp, "lp": Lp, "cl": CL},
        "fixed_itail_a": 20e-6,
        "fixed_itail_uA": 20.0,
        "output_fixture": "Ioutbias = 0 A; no output DC voltage source",
        "vb1_sweep_range_v": [0.90, 1.70],
        "center_vb1_v": 1.30,
        "vb1_definition": "VB1 biases PMOS cascode devices M7/M8.",
        "mirror_ratio": "|I(M5)| / |I(M3)|",
        "mirror_ratio_target": 1.5,
        "mirror_ratio_error": "abs(measured mirror ratio - 1.5)",
        "csv": str(csv_path),
    }
    json_path = DATA_DIR / "complementary_bias_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    display = df.copy()
    display["itail_uA"] = display["itail"] * 1e6
    display["gbw_MHz"] = display["gbw_hz"] * 1e-6
    display["power_uW"] = display["power_w"] * 1e6
    display["sat_margin_mV"] = display["sat_margin"] * 1e3
    display["v_tail_mV"] = display["v_tail"] * 1e3
    display = display[
        ["sweep", "itail_uA", "vb1", "vb2", "vb3", "gain_db", "gbw_MHz",
         "pm_deg", "power_uW", "sat_margin_mV", "v_tail_mV", "vout",
         "mirror_ratio_error", "valid"]
    ]
    rows_html = display.to_html(index=False, border=0, classes="table")
    detailed_html = df.to_html(index=False, border=0, classes="table")

    plots_html = "".join(
        f'<img style="max-width:100%;margin:12px 0" src="{uri}" alt="Bias sweep plot">'
        for uri in oat_plots
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Complementary OTA bias sweeps</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1200px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:5px 8px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
</style></head><body>
<h1>Complementary OTA bias sweeps</h1>
<p>Generated netlists use <code>Ioutbias = 0 A</code>. There is no 0.9 V output-voltage clamp/source.</p>
<p><b>Tail current fixed at 20 µA</b> for all points. The detailed tables include source currents, every node voltage, and VGS/VDS/VDSAT/VTH/saturation margin for M1–M12.</p>
<p><b>Mirror ratio definition:</b> <code>|I(M5)| / |I(M3)|</code>. M5 is the PMOS folding-mirror current on the left summing node; M3 is the left NMOS input-device current. The target ratio is 1.5, and mirror-ratio error is <code>|measured ratio - 1.5|</code>.</p>
<p><b>Points:</b> {len(df)} &nbsp; <b>Valid:</b> {len(valid)} &nbsp; <b>Elapsed:</b> {summary['elapsed_seconds']:.1f} s</p>
<h2>One-at-a-time sweeps</h2>{plots_html}
<h2>VB1 vs VB3 gain</h2>
<img style="max-width:800px" src="{heatmap}" alt="VB1 versus VB3 gain heatmap">
<h2>Center point</h2>{pd.DataFrame([center]).to_html(index=False)}
<h2>Best valid point</h2>{("None" if best is None else pd.DataFrame([best]).to_html(index=False))}
<h2>All points</h2>{rows_html}
<h2>Complete operating-point data</h2>{detailed_html}
</body></html>"""
    html_path = OUTPUT_DIR / "complementary_bias_sweep.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
