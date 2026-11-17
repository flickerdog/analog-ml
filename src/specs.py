from pathlib import Path

import pandas as pd
import yaml

_FILE = Path(__file__).resolve().parent.parent / "specs.yaml"

# YAML key -> (data column, factor from YAML unit to data unit, text template)
_TABLE = {
    "gain_db": ("gain_db", 1, "gain ≥ {:g} dB"),
    "pm_deg": ("pm_deg", 1, "PM ≥ {:g}°"),
    "gbw_mhz": ("gbw_hz", 1e6, "GBW ≥ {:g} MHz"),
}

with open(_FILE) as f:
    CFG = yaml.safe_load(f)               # nested dict read from specs.yaml

unknown = set(CFG["specs"]) - set(_TABLE)
if unknown:
    raise KeyError(f"unknown spec(s) in specs.yaml: {sorted(unknown)}")

# Minimum values in the units of the data columns (Hz, dB, degrees)
SPECS = {_TABLE[k][0]: v * _TABLE[k][1] for k, v in CFG["specs"].items()}
SAT_MARGIN = CFG["validity"]["sat_margin_min_mv"] / 1e3   # V
V_TAIL_MIN = CFG["validity"]["v_tail_min_mv"] / 1e3       # V
LOAD_CAP_PF = CFG["conditions"]["load_cap_pf"]
LOAD_CAP = LOAD_CAP_PF * 1e-12            # F


def spec_text():
    specs = ", ".join(_TABLE[k][2].format(v) for k, v in CFG["specs"].items())
    return f"{specs} (CL = {LOAD_CAP_PF:g} pF)"


def validity_text():
    return (f"valid: OTA devices saturated (margin > {SAT_MARGIN * 1e3:g} mV), "
            f"tail node > {V_TAIL_MIN * 1e3:g} mV")


def classify(df):
    """Label each design: 'invalid', 'fails spec' or 'passes'."""
    valid = (df["sat_margin"] > SAT_MARGIN) & (df["v_tail"] > V_TAIL_MIN)
    meets = pd.Series(True, index=df.index)
    for col, minimum in SPECS.items():
        meets &= df[col] >= minimum
    label = pd.Series("fails spec", index=df.index)
    label[meets] = "passes"
    label[~valid] = "invalid"
    return label