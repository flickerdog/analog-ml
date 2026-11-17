import webbrowser
from pathlib import Path

import pandas as pd

# raw column -> (label with unit, multiplier applied to the raw value)
UNITS = {
    "w1": ("W1 (µm)", 1), "l1": ("L1 (µm)", 1),
    "w3": ("W3 (µm)", 1), "l3": ("L3 (µm)", 1),
    "itail": ("Itail (µA)", 1e6),
    "gain_db": ("Gain (dB)", 1),
    "gbw_hz": ("GBW (MHz)", 1e-6),
    "pm_deg": ("PM (°)", 1),
    "power_w": ("Power (µW)", 1e6),
    "sat_margin": ("Sat. margin (mV)", 1e3),
    "v_tail": ("V tail (mV)", 1e3),
}

CSS = """<style>
body{font-family:sans-serif;margin:2rem}
table{border-collapse:collapse;margin-bottom:2rem}
td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}
th{background:#f0f0f0}
</style>"""


def readable(df):
    """Rename columns with units and rescale values (A -> µA, Hz -> MHz, ...)."""
    out = {}
    for c in df.columns:
        label, k = UNITS.get(c, (c, None))
        out[label] = df[c] if k is None else df[c] * k
    return pd.DataFrame(out)


def stats(df):
    """describe() with readable units; one row per quantity."""
    s = readable(df).describe().T.rename(columns={"50%": "median"})
    s["count"] = s["count"].astype(int)
    return s


def table_html(df):
    numeric = df.select_dtypes("number").columns
    fmt = {c: ("{:.0f}" if c == "count" else "{:.3g}") for c in numeric}
    return df.style.format(fmt, na_rep="–").to_html()


def save_report(title, sections, path, open_browser=True):
    """sections: list of (heading, DataFrame). Writes one HTML page."""
    body = "".join(f"<h2>{h}</h2>{table_html(t)}" for h, t in sections)
    html = (f'<html><head><meta charset="utf-8">{CSS}</head>'
            f"<body><h1>{title}</h1>{body}</body></html>")
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print("wrote", path)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())