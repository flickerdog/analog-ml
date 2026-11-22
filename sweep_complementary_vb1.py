"""Isolated VB1 sweep for the complementary folded-cascode OTA.

VB2, VB3, tail current, geometry, load, and common mode stay fixed.
"""

import base64
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.complementary_ota import evaluate


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"

WN = 48.0
LN = 0.19
WP = 90.0
LP = 0.19
ITAIL = 20e-6
CL = 1e-12
VB2 = 1.05
VB3 = 0.66
FOLD_W = 20.0
FOLD_L = 0.15


def fig_uri(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = []

    for vb1 in np.arange(0.50, 1.701, 0.025):
        result = evaluate(
            WN, LN, WP, LP, ITAIL, CL,
            fold_w=FOLD_W,
            fold_l=FOLD_L,
            vb1=float(vb1),
            vb2=VB2,
            vb3=VB3,
        )
        rows.append({
            "vb1": float(vb1),
            "vb2": VB2,
            "vb3": VB3,
            "itail": ITAIL,
            **result,
        })

    df = pd.DataFrame(rows)
    df["m5_minus_m3_plus_m7_uA"] = (
        df["m5_id_a"] - df["m3_id_a"] - df["m7_id_a"]
    ) * 1e6
    df["m6_minus_m4_plus_m8_uA"] = (
        df["m6_id_a"] - df["m4_id_a"] - df["m8_id_a"]
    ) * 1e6
    csv_path = DATA_DIR / "complementary_vb1_sweep.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for device in ("m3", "m4", "m5", "m6", "m7", "m8"):
        ax.plot(
            df["vb1"],
            df[f"{device}_id_a"] * 1e6,
            marker="o",
            label=device.upper(),
        )
    ax.set_xlabel("VB1 (V)")
    ax.set_ylabel("Drain current (uA)")
    ax.set_title("M3-M8 drain currents versus VB1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    current_plot = fig_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for node in ("pmid1", "pmid2", "upper1", "lower1", "lower2", "out"):
        ax.plot(
            df["vb1"],
            df[f"v_{node}"],
            marker="o",
            label=node,
        )
    ax.set_xlabel("VB1 (V)")
    ax.set_ylabel("Node voltage (V)")
    ax.set_title("Important nodes versus VB1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    node_plot = fig_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        df["vb1"],
        df["m5_minus_m3_plus_m7_uA"],
        marker="o",
        label="M5 - (M3 + M7)",
    )
    ax.plot(
        df["vb1"],
        df["m6_minus_m4_plus_m8_uA"],
        marker="o",
        label="M6 - (M4 + M8)",
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("VB1 (V)")
    ax.set_ylabel("Current residual (uA)")
    ax.set_title("Shared-gate PMOS summation KCL residual")
    ax.grid(True, alpha=0.3)
    ax.legend()
    kcl_plot = fig_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for device in ("m3", "m4", "m5", "m6", "m7", "m8"):
        ax.plot(
            df["vb1"],
            df[f"{device}_sat_margin_v"],
            marker="o",
            label=device.upper(),
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("VB1 (V)")
    ax.set_ylabel("Saturation margin (V)")
    ax.set_title("Key-device saturation margins versus VB1")
    ax.grid(True, alpha=0.3)
    ax.legend()
    sat_plot = fig_uri(fig)

    def table_for(columns):
        return df[columns].to_html(index=False, border=0, classes="table", float_format=lambda value: f"{value:.6g}")

    summary_columns = [
        "vb1", "v_nsrc", "v_psrc", "v_pmid1", "v_pmid2",
        "v_upper1", "v_lower1", "v_lower2", "v_out",
        "m3_id_a", "m4_id_a", "m5_id_a", "m6_id_a", "m7_id_a", "m8_id_a",
        "gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail",
        "mirror_current_ratio", "mirror_ratio_error", "valid",
    ]
    saturation_columns = ["vb1"] + [
        f"{device}_{field}"
        for device in ("m1", "m2", "m3", "m4", "m5", "m6",
                       "m7", "m8", "m9", "m10", "m11", "m12")
        for field in ("vgs_v", "vds_v", "vdsat_v", "vth_v", "sat_margin_v")
    ]
    current_columns = ["vb1"] + [
        f"{device}_id_a"
        for device in ("m1", "m2", "m3", "m4", "m5", "m6",
                       "m7", "m8", "m9", "m10", "m11", "m12")
    ]
    node_columns = [
        "vb1", "v_nsrc", "v_psrc", "v_pmid1", "v_pmid2",
        "v_upper1", "v_lower1", "v_lower2", "v_out",
        "i_supply", "i_vb1", "i_vb2", "i_vb3", "i_output",
        "i_tail_p_measured", "i_tail_n_measured",
    ]

    center = df.iloc[(df["vb1"] - 1.30).abs().argmin()].to_dict()
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "fixed": {
            "itail_a": ITAIL,
            "vb2_v": VB2,
            "vb3_v": VB3,
            "wn_um": WN,
            "ln_um": LN,
            "wp_um": WP,
            "lp_um": LP,
            "fold_w_um": FOLD_W,
            "fold_l_um": FOLD_L,
            "cl_f": CL,
        },
        "vb1_range_v": [0.50, 1.70],
        "points": len(df),
        "center_nearest": center,
        "csv": str(csv_path),
        "m5_m6_note": (
            "M5 and M6 are PMOS devices sharing upper10 as gate and VDD "
            "as source; M5 drains to pmid10 and M6 drains to pmid20."
        ),
    }
    json_path = DATA_DIR / "complementary_vb1_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Isolated complementary OTA VB1 sweep</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:4px 7px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
img{{max-width:100%;margin:12px 0}}
</style></head><body>
<h1>Isolated VB1 sweep</h1>
<p>Fixed: tail current=20 uA, VB2={VB2:.3f} V, VB3={VB3:.3f} V, output fixture=<code>Ioutbias=0 A</code>, no 0.9 V output source.</p>
<p><b>VB1 sweep:</b> 0.50 to 1.70 V in 0.025 V steps. M5 and M6 share <code>upper10</code> as gate and VDD as source. M5 drains to <code>pmid10</code>; M6 drains to <code>pmid20</code>.</p>
<h2>Drain currents</h2><img src="{current_plot}" alt="M3-M8 currents">
{table_for(current_columns)}
<h2>Node voltages</h2><img src="{node_plot}" alt="Node voltages">
{table_for(node_columns)}
<h2>Shared-gate M5/M6 current summation</h2>
<p>M5 and M6 share <code>upper10</code> as gate, but they are not conventional current mirrors: upper10 is an internal summing/cascode node set by the branches. At these operating points, the branch currents satisfy <code>I(M5) ~= I(M3)+I(M7)</code> and <code>I(M6) ~= I(M4)+I(M8)</code>. The residual plot and table expose deviations from exact KCL.</p>
<img src="{kcl_plot}" alt="M5/M6 KCL residual">
{table_for(["vb1", "m3_id_a", "m7_id_a", "m5_id_a", "m5_minus_m3_plus_m7_uA", "m4_id_a", "m8_id_a", "m6_id_a", "m6_minus_m4_plus_m8_uA"])}
<h2>Key saturation margins</h2><img src="{sat_plot}" alt="Saturation margins">
<h2>All M1-M12 VGS/VDS/VDSAT/VTH/saturation data</h2>
{table_for(saturation_columns)}
<h2>Requested summary</h2>
{table_for(summary_columns)}
<h2>Nearest VB1=1.30 V point</h2>
{pd.DataFrame([center]).to_html(index=False, border=0)}
<h2>Complete operating-point CSV table</h2>
{df.to_html(index=False, border=0, classes="table", float_format=lambda value: f"{value:.6g}")}
</body></html>"""
    html_path = OUTPUT_DIR / "complementary_vb1_sweep.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
