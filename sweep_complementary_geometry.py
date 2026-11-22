"""One-factor-at-a-time geometry sweep around the nominal complementary OTA point."""

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

NOMINAL = {
    "wn": 15.192377,
    "ln": 1.464142,
    "wp": 35.440671,
    "lp": 0.151400,
    "itail": 18e-6,
    "vb1": 0.80,
    "vb2": 1.00,
    "vb3": 0.65,
    "fold_w": 20.0,
    "fold_l": 0.15,
    "cl": 1e-12,
}


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

    def run(sweep, **changes):
        params = dict(NOMINAL)
        params.update(changes)
        result = evaluate(
            params["wn"], params["ln"], params["wp"], params["lp"],
            params["itail"], params["cl"],
            fold_w=params["fold_w"], fold_l=params["fold_l"],
            vb1=params["vb1"], vb2=params["vb2"], vb3=params["vb3"],
        )
        rows.append({"sweep": sweep, **params, **result})

    for value in np.arange(7.0, 31.001, 3.0):
        run("wn", wn=float(value))
    for value in np.arange(0.15, 2.001, 0.25):
        run("ln", ln=float(value))
    for value in np.arange(7.0, 55.001, 6.0):
        run("wp", wp=float(value))
    for value in np.arange(0.15, 2.001, 0.25):
        run("lp", lp=float(value))
    for value in np.arange(10.0, 31.001, 2.5):
        run("fold_w", fold_w=float(value))
    for value in np.arange(0.15, 1.001, 0.10):
        run("fold_l", fold_l=float(value))

    df = pd.DataFrame(rows)
    csv_path = DATA_DIR / "complementary_geometry_sweep.csv"
    df.to_csv(csv_path, index=False)

    plots = []
    for sweep in ("wn", "ln", "wp", "lp", "fold_w", "fold_l"):
        subset = df[df["sweep"] == sweep]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric, ylabel in (
            (axes[0], "gain_db", "Gain (dB)"),
            (axes[1], "gbw_hz", "GBW (Hz)"),
            (axes[2], "sat_margin", "Saturation margin (V)"),
        ):
            ax.plot(subset[sweep], subset[metric], marker="o")
            ax.set_xlabel(sweep)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        plots.append(fig_uri(fig))

    valid = df[df["valid"].astype(bool)]
    best = valid.sort_values("gain_db", ascending=False).iloc[0].to_dict() if not valid.empty else None

    summary = {
        "elapsed_seconds": time.perf_counter() - started,
        "points": len(df),
        "valid_points": len(valid),
        "nominal": NOMINAL,
        "minimum_mirror_ratio": 1.20,
        "mirror_rule": "I(M5)/I(M3) > 1.2 and I(M6)/I(M4) > 1.2",
        "best_valid": best,
        "csv": str(csv_path),
    }
    json_path = DATA_DIR / "complementary_geometry_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def table(columns):
        return df[columns].to_html(
            index=False, border=0, classes="table",
            float_format=lambda value: f"{value:.6g}",
        )

    summary_columns = [
        "sweep", "wn", "ln", "wp", "lp", "fold_w", "fold_l",
        "gain_db", "gbw_hz", "pm_deg", "power_w",
        "sat_margin", "v_tail", "mirror_current_ratio",
        "m5_sat_margin_v", "m6_sat_margin_v", "valid",
    ]
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Complementary OTA geometry sweep</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:4px 7px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
img{{max-width:100%;margin:12px 0}}
</style></head><body>
<h1>Complementary OTA one-factor geometry sweep</h1>
<p><b>Nominal bias:</b> VB1={NOMINAL['vb1']} V, VB2={NOMINAL['vb2']} V, VB3={NOMINAL['vb3']} V, Itail={NOMINAL['itail']*1e6:g} uA.</p>
<p>Nominal geometry: wn={NOMINAL['wn']:.6g} um, ln={NOMINAL['ln']:.6g} um, wp={NOMINAL['wp']:.6g} um, lp={NOMINAL['lp']:.6g} um, folding width={NOMINAL['fold_w']:.6g} um, folding length={NOMINAL['fold_l']:.6g} um.</p>
<p><b>Validity:</b> saturation margin &gt; 50 mV, tail headroom &gt; 150 mV, and both M5/M3 and M6/M4 current ratios &gt; 1.2.</p>
<p><b>Points:</b> {len(df)} &nbsp; <b>Valid:</b> {len(valid)} &nbsp; <b>Elapsed:</b> {summary['elapsed_seconds']:.1f} s</p>
<h2>Gain, GBW, and saturation plots</h2>
{''.join(f'<img src="{uri}" alt="Geometry sweep plot">' for uri in plots)}
<h2>Best valid point</h2>{("None" if best is None else pd.DataFrame([best]).to_html(index=False, border=0))}
<h2>All geometry points</h2>{table(summary_columns)}
<h2>Complete operating-point data</h2>{df.to_html(index=False, border=0, classes="table", float_format=lambda value: f"{value:.6g}")}
</body></html>"""
    html_path = OUTPUT_DIR / "complementary_geometry_sweep.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
