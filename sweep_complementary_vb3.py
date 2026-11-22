"""Isolated VB3 sweep for the complementary folded-cascode OTA.

VB1, VB2, tail current, geometry, load, and common mode stay fixed.
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

WN = 48.0
LN = 0.19
WP = 90.0
LP = 0.19
ITAIL = 20e-6
CL = 1e-12
VB1 = 0.80
VB2 = 1.05
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

    for vb3 in np.arange(0.10, 1.301, 0.025):
        result = evaluate(
            WN, LN, WP, LP, ITAIL, CL,
            fold_w=FOLD_W,
            fold_l=FOLD_L,
            vb1=VB1,
            vb2=VB2,
            vb3=float(vb3),
        )
        rows.append({
            "vb1": VB1,
            "vb2": VB2,
            "vb3": float(vb3),
            "itail": ITAIL,
            **result,
        })

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "complementary_vb3_sweep.csv"
    df.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for device in ("m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11", "m12"):
        ax.plot(df["vb3"], df[f"{device}_id_a"] * 1e6, marker="o", label=device.upper())
    ax.set_xlabel("VB3 (V)")
    ax.set_ylabel("Drain current (uA)")
    ax.set_title("M3-M12 drain currents versus VB3")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    current_plot = fig_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for node in ("pmid1", "pmid2", "upper1", "lower1", "lower2", "out"):
        ax.plot(df["vb3"], df[f"v_{node}"], marker="o", label=node)
    ax.set_xlabel("VB3 (V)")
    ax.set_ylabel("Node voltage (V)")
    ax.set_title("Important nodes versus VB3")
    ax.grid(True, alpha=0.3)
    ax.legend()
    node_plot = fig_uri(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for device in ("m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11", "m12"):
        ax.plot(df["vb3"], df[f"{device}_sat_margin_v"], marker="o", label=device.upper())
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("VB3 (V)")
    ax.set_ylabel("Saturation margin (V)")
    ax.set_title("Saturation margins versus VB3")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    sat_plot = fig_uri(fig)

    def table(columns):
        return df[columns].to_html(
            index=False, border=0, classes="table",
            float_format=lambda value: f"{value:.6g}",
        )

    node_columns = [
        "vb3", "v_nsrc", "v_psrc", "v_pmid1", "v_pmid2",
        "v_upper1", "v_lower1", "v_lower2", "v_out",
        "i_supply", "i_vb1", "i_vb2", "i_vb3", "i_output",
        "i_tail_p_measured", "i_tail_n_measured",
    ]
    current_columns = ["vb3"] + [
        f"{device}_id_a"
        for device in ("m1", "m2", "m3", "m4", "m5", "m6",
                       "m7", "m8", "m9", "m10", "m11", "m12")
    ]
    saturation_columns = ["vb3"] + [
        f"{device}_{field}"
        for device in ("m1", "m2", "m3", "m4", "m5", "m6",
                       "m7", "m8", "m9", "m10", "m11", "m12")
        for field in ("vgs_v", "vds_v", "vdsat_v", "vth_v", "sat_margin_v")
    ]
    summary_columns = [
        "vb3", "v_pmid1", "v_pmid2", "v_upper1", "v_lower1",
        "v_lower2", "v_out",
        "m3_id_a", "m4_id_a", "m5_id_a", "m6_id_a",
        "m7_id_a", "m8_id_a", "m9_id_a", "m10_id_a",
        "m11_id_a", "m12_id_a",
        "gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail",
        "mirror_current_ratio", "mirror_ratio_error", "valid",
    ]

    nearest = df.iloc[(df["vb3"] - 0.66).abs().argmin()].to_dict()
    valid = df[df["valid"]]
    best = valid.sort_values("gain_db", ascending=False).iloc[0].to_dict() if not valid.empty else None
    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "fixed": {
            "itail_a": ITAIL,
            "vb1_v": VB1,
            "vb2_v": VB2,
            "wn_um": WN,
            "ln_um": LN,
            "wp_um": WP,
            "lp_um": LP,
            "fold_w_um": FOLD_W,
            "fold_l_um": FOLD_L,
            "cl_f": CL,
        },
        "vb3_range_v": [0.10, 1.30],
        "points": len(df),
        "valid_points": len(valid),
        "nearest_vb3_0p66": nearest,
        "best_valid": best,
        "csv": str(csv_path),
    }
    json_path = DATA_DIR / "complementary_vb3_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Isolated complementary OTA VB3 sweep</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:4px 7px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
img{{max-width:100%;margin:12px 0}}
</style></head><body>
<h1>Isolated VB3 sweep</h1>
<p>Fixed: tail current=20 uA, VB1={VB1:.2f} V, VB2={VB2:.2f} V, output fixture=<code>Ioutbias=0 A</code>, no 0.9 V output source.</p>
<p><b>VB3 sweep:</b> 0.10 to 1.30 V in 0.025 V steps.</p>
<p><b>Points:</b> {len(df)} &nbsp; <b>Valid:</b> {len(valid)} &nbsp; <b>Elapsed:</b> {summary['elapsed_seconds']:.1f} s</p>
<h2>Drain currents</h2><img src="{current_plot}" alt="M3-M12 currents">{table(current_columns)}
<h2>Node voltages</h2><img src="{node_plot}" alt="Node voltages">{table(node_columns)}
<h2>Saturation margins</h2><img src="{sat_plot}" alt="Saturation margins">
<h2>All M1-M12 VGS/VDS/VDSAT/VTH/saturation data</h2>{table(saturation_columns)}
<h2>VB3 summary</h2>{table(summary_columns)}
<h2>Nearest VB3=0.66 V point</h2>{pd.DataFrame([nearest]).to_html(index=False, border=0)}
<h2>Complete operating-point data</h2>{df.to_html(index=False, border=0, classes="table", float_format=lambda value: f"{value:.6g}")}
</body></html>"""
    html_path = OUTPUT_DIR / "complementary_vb3_sweep.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
