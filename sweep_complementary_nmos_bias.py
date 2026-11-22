"""2D VB2/VB3 sweep for the complementary folded-cascode OTA."""

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

    for vb2 in np.arange(0.70, 1.301, 0.05):
        for vb3 in np.arange(0.40, 1.001, 0.05):
            result = evaluate(
                WN, LN, WP, LP, ITAIL, CL,
                fold_w=FOLD_W,
                fold_l=FOLD_L,
                vb1=VB1,
                vb2=float(vb2),
                vb3=float(vb3),
            )
            rows.append({
                "vb1": VB1,
                "vb2": float(vb2),
                "vb3": float(vb3),
                "itail": ITAIL,
                **result,
            })

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "complementary_nmos_bias_sweep.csv"
    df.to_csv(csv_path, index=False)

    def heatmap(metric, title, label):
        pivot = df.pivot(index="vb3", columns="vb2", values=metric)
        fig, ax = plt.subplots(figsize=(9, 6))
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
        ax.set_xlabel("VB2 (V)")
        ax.set_ylabel("VB3 (V)")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=label)
        return fig_uri(fig)

    heatmaps = [
        heatmap("m11_sat_margin_v", "M11 saturation margin: VB2 vs VB3", "Margin (V)"),
        heatmap("m12_sat_margin_v", "M12 saturation margin: VB2 vs VB3", "Margin (V)"),
        heatmap("m9_sat_margin_v", "M9 saturation margin: VB2 vs VB3", "Margin (V)"),
        heatmap("m10_sat_margin_v", "M10 saturation margin: VB2 vs VB3", "Margin (V)"),
        heatmap("m5_sat_margin_v", "M5 saturation margin: VB2 vs VB3", "Margin (V)"),
        heatmap("mirror_current_ratio", "M5/M3 mirror ratio: VB2 vs VB3", "I(M5)/I(M3)"),
        heatmap("gain_db", "Gain: VB2 vs VB3", "Gain (dB)"),
    ]

    valid = df[df["valid"]]
    best = valid.sort_values("gain_db", ascending=False).iloc[0].to_dict() if not valid.empty else None

    summary_columns = [
        "vb2", "vb3",
        "v_pmid1", "v_pmid2", "v_upper1", "v_lower1", "v_lower2", "v_out",
        "m3_id_a", "m4_id_a", "m5_id_a", "m6_id_a",
        "m7_id_a", "m8_id_a", "m9_id_a", "m10_id_a",
        "m11_id_a", "m12_id_a",
        "m9_sat_margin_v", "m10_sat_margin_v",
        "m11_sat_margin_v", "m12_sat_margin_v",
        "m5_sat_margin_v", "m6_sat_margin_v",
        "gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail",
        "mirror_current_ratio", "mirror_ratio_error", "valid",
    ]
    saturation_columns = ["vb2", "vb3"] + [
        f"{device}_{field}"
        for device in ("m1", "m2", "m3", "m4", "m5", "m6",
                       "m7", "m8", "m9", "m10", "m11", "m12")
        for field in ("vgs_v", "vds_v", "vdsat_v", "vth_v", "sat_margin_v")
    ]

    def table(columns):
        return df[columns].to_html(
            index=False, border=0, classes="table",
            float_format=lambda value: f"{value:.6g}",
        )

    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "fixed": {
            "itail_a": ITAIL,
            "vb1_v": VB1,
            "wn_um": WN,
            "ln_um": LN,
            "wp_um": WP,
            "lp_um": LP,
            "fold_w_um": FOLD_W,
            "fold_l_um": FOLD_L,
            "cl_f": CL,
        },
        "vb2_range_v": [0.70, 1.30],
        "vb3_range_v": [0.40, 1.00],
        "points": len(df),
        "valid_points": len(valid),
        "mirror_ratio_constraint": "I(M5)/I(M3) > 1.2 and I(M6)/I(M4) > 1.2",
        "best_valid": best,
        "csv": str(csv_path),
    }
    json_path = DATA_DIR / "complementary_nmos_bias_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Complementary OTA VB2/VB3 sweep</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:4px 7px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
img{{max-width:100%;margin:12px 0}}
</style></head><body>
<h1>2D VB2/VB3 sweep</h1>
<p>Fixed: tail current=20 uA, VB1={VB1:.2f} V, output fixture=<code>Ioutbias=0 A</code>, no 0.9 V output source.</p>
<p><b>VB2:</b> 0.70–1.30 V; <b>VB3:</b> 0.40–1.00 V. This sweep varies VB2 and VB3 together so NMOS bottom-device VDSAT and cascode-device VDS can be inspected.</p>
<p><b>Validity rule:</b> M5/M6 current must exceed M3/M4 current by a factor greater than 1.2, in addition to saturation and tail headroom.</p>
<p><b>Points:</b> {len(df)} &nbsp; <b>Valid:</b> {len(valid)} &nbsp; <b>Elapsed:</b> {summary['elapsed_seconds']:.1f} s</p>
<h2>Heatmaps</h2>
{''.join(f'<img src="{uri}" alt="Bias heatmap">' for uri in heatmaps)}
<h2>NMOS bias summary</h2>{table(summary_columns)}
<h2>All M1-M12 VGS/VDS/VDSAT/VTH/saturation data</h2>{table(saturation_columns)}
<h2>Complete operating-point data</h2>{df.to_html(index=False, border=0, classes="table", float_format=lambda value: f"{value:.6g}")}
</body></html>"""
    html_path = OUTPUT_DIR / "complementary_nmos_bias_sweep.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
