"""Compare complementary folded-cascode and 5T OTA nominal sweeps."""

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
COMP_PATH = ROOT / "data/complementary_search256.csv"
OTA5T_PATH = ROOT / "data/ota5t_sobol.csv"
REPORT_PATH = ROOT / "reports/complementary_vs_5t.html"


def fig_uri(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def passes_specs(df):
    return (
        df["gain_db"].ge(35)
        & df["pm_deg"].ge(60)
        & df["gbw_hz"].ge(1e6)
        & df["valid"].astype(bool)
    )


def stats(df, name):
    valid = df[df["valid"].astype(bool)]
    spec_pass = df[passes_specs(df)]
    metrics = ["gain_db", "gbw_hz", "pm_deg", "power_w", "sat_margin", "v_tail"]
    rows = []
    for metric in metrics:
        values = valid[metric].dropna()
        rows.append({
            "Topology": name,
            "Metric": metric,
            "Valid count": len(values),
            "Median": values.median(),
            "Minimum": values.min(),
            "Maximum": values.max(),
        })
    rows.append({
        "Topology": name,
        "Metric": "spec_pass_count",
        "Valid count": len(valid),
        "Median": len(spec_pass),
        "Minimum": len(spec_pass),
        "Maximum": len(spec_pass),
    })
    return pd.DataFrame(rows)


def main():
    complementary = pd.read_csv(COMP_PATH)
    ota5t = pd.read_csv(OTA5T_PATH)
    ota5t_valid = ota5t[ota5t["valid"].astype(bool)]
    complementary_valid = complementary[complementary["valid"].astype(bool)]
    complementary_pass = complementary[passes_specs(complementary)]
    ota5t_pass = ota5t[passes_specs(ota5t)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    metrics = [
        ("gain_db", "Gain (dB)"),
        ("gbw_hz", "GBW (Hz)"),
        ("pm_deg", "Phase margin (deg)"),
        ("power_w", "Power (W)"),
        ("sat_margin", "Sat margin (V)"),
        ("v_tail", "Tail headroom (V)"),
    ]
    for ax, (metric, label) in zip(axes.flat, metrics):
        ax.hist(ota5t_valid[metric].dropna(), bins=30, alpha=0.55, label="5T")
        ax.hist(complementary_valid[metric].dropna(), bins=30, alpha=0.55, label="Complementary")
        ax.set_xlabel(label)
        ax.set_ylabel("Valid designs")
        ax.legend()
        ax.grid(True, alpha=0.25)
    distributions = fig_uri(fig)

    table = pd.concat([
        stats(complementary, "Complementary"),
        stats(ota5t, "5T"),
    ], ignore_index=True)

    best_comp = complementary_valid.sort_values("gain_db", ascending=False).head(10)
    best_5t = ota5t_valid.sort_values("gain_db", ascending=False).head(10)

    summary = pd.DataFrame([
        {
            "Topology": "Complementary",
            "Total designs": len(complementary),
            "Valid operating points": len(complementary_valid),
            "Valid (%)": 100 * len(complementary_valid) / len(complementary),
            "Gain/PM/GBW spec pass": len(complementary_pass),
            "Best gain (dB)": complementary_valid["gain_db"].max(),
            "Median gain (dB)": complementary_valid["gain_db"].median(),
            "Median GBW (Hz)": complementary_valid["gbw_hz"].median(),
            "Median PM (deg)": complementary_valid["pm_deg"].median(),
            "Median power (W)": complementary_valid["power_w"].median(),
        },
        {
            "Topology": "5T",
            "Total designs": len(ota5t),
            "Valid operating points": len(ota5t_valid),
            "Valid (%)": 100 * len(ota5t_valid) / len(ota5t),
            "Gain/PM/GBW spec pass": len(ota5t_pass),
            "Best gain (dB)": ota5t_valid["gain_db"].max(),
            "Median gain (dB)": ota5t_valid["gain_db"].median(),
            "Median GBW (Hz)": ota5t_valid["gbw_hz"].median(),
            "Median PM (deg)": ota5t_valid["pm_deg"].median(),
            "Median power (W)": ota5t_valid["power_w"].median(),
        },
    ])

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Complementary vs 5T OTA sweep</title>
<style>
body{{font-family:sans-serif;margin:2rem;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:1rem 0}}
th,td{{border:1px solid #ccc;padding:4px 7px;text-align:right}}
th{{background:#eee}}
code{{background:#f4f4f4;padding:2px 4px}}
img{{max-width:100%;margin:12px 0}}
</style></head><body>
<h1>Complementary folded-cascode vs 5T OTA</h1>
<p>Complementary validity uses positive saturation margins, tail headroom, and <code>I(M5)/I(M3) > 1.2</code> plus <code>I(M6)/I(M4) > 1.2</code>. Both sweeps use the 1 pF load condition.</p>
<h2>Summary</h2>{summary.to_html(index=False, border=0)}
<h2>Valid-design distributions</h2><img src="{distributions}" alt="Metric distributions">
<h2>Detailed valid statistics</h2>{table.to_html(index=False, border=0)}
<h2>Best complementary valid designs</h2>{best_comp.to_html(index=False, border=0)}
<h2>Best 5T valid designs</h2>{best_5t.to_html(index=False, border=0)}
</body></html>"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print("wrote", REPORT_PATH)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
