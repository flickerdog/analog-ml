"""Benchmark the saved surrogate against fresh ngspice simulations.

    python benchmark_surrogate.py
    python benchmark_surrogate.py --n 100 --seed 123

The benchmark samples new designs inside the sweep ranges, checks that none
match a row in the training sweep, then compares per-design ngspice results
with predictions from the saved model.
"""
import argparse
from html import escape
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.specs import LOAD_CAP, SAT_MARGIN, V_TAIL_MIN
from src.topology import load_topology, topology_names

ROOT = Path(__file__).resolve().parent
METRICS = ("gain_db", "gbw_hz", "pm_deg")
PREDICTION_TARGETS = (*METRICS, "sat_margin", "v_tail")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--topology", choices=topology_names(), default="ota5t")
parser.add_argument("--model", type=Path)
parser.add_argument("--training-data", type=Path)
parser.add_argument("--n", type=int, default=100, help="number of fresh configurations")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="per-design benchmark results CSV",
)
parser.add_argument(
    "--summary",
    type=Path,
    default=None,
    help="aggregate results JSON",
)
parser.add_argument(
    "--html",
    type=Path,
    default=None,
    help="HTML benchmark report",
)
args = parser.parse_args()
topology = load_topology(args.topology)
args.model = args.model or topology.MODEL_FILE
args.training_data = args.training_data or topology.DATA_FILE
suffix = "" if args.topology == "ota5t" else f"_{args.topology}"
args.output = args.output or ROOT / f"reports/surrogate_benchmark{suffix}.csv"
args.summary = args.summary or ROOT / f"reports/surrogate_benchmark_summary{suffix}.json"
args.html = args.html or ROOT / f"reports/surrogate_benchmark{suffix}.html"
evaluate = topology.evaluate
RANGES = topology.RANGES
if getattr(topology, "TRAIN_POWER", False):
    PREDICTION_TARGETS = (*PREDICTION_TARGETS, "power_w")

if args.n < 1:
    parser.error("--n must be at least 1")
if not args.model.is_file():
    parser.error(f"model file not found: {args.model}")
if not args.training_data.is_file():
    parser.error(f"training sweep not found: {args.training_data}")

training_df = pd.read_csv(args.training_data)
features = list(RANGES)
missing = sorted(set(features) - set(training_df.columns))
if missing:
    parser.error(f"training sweep is missing columns: {', '.join(missing)}")
training_x = training_df[features].dropna().to_numpy(dtype=float)

model_load_started = time.perf_counter()
saved = joblib.load(args.model)
model_load_s = time.perf_counter() - model_load_started
if saved.get("topology", "ota5t") != args.topology:
    parser.error(f"model was trained for topology {saved.get('topology', 'ota5t')}")
if saved.get("features") != features:
    parser.error(
        f"model features {saved.get('features')} do not match sweep features {features}"
    )
if saved.get("input_transform") != "log10":
    parser.error("benchmark expects a model trained with log10 input transformation")
if any(name not in saved.get("models", {}) for name in PREDICTION_TARGETS):
    parser.error(
        "model must contain regressors for: "
        + ", ".join(PREDICTION_TARGETS)
    )


def sample_fresh_designs(n, seed):
    """Draw log-uniform Sobol points not present in the model's training sweep."""
    lo = np.log10([RANGES[name][0] for name in features])
    hi = np.log10([RANGES[name][1] for name in features])
    selected = []
    current_seed = seed
    block_size = max(2, 2 ** math.ceil(math.log2(n)))

    while len(selected) < n:
        sampler = qmc.Sobol(d=len(features), scramble=True, seed=current_seed)
        unit = sampler.random_base2(m=int(math.log2(block_size)))
        block = 10 ** (lo + unit * (hi - lo))
        current_seed += 1

        for point in block:
            if len(training_x) and np.isclose(training_x, point, rtol=1e-12, atol=0).all(axis=1).any():
                continue
            if any(np.allclose(previous, point, rtol=1e-12, atol=0) for previous in selected):
                continue
            selected.append(point)
            if len(selected) == n:
                break

        if current_seed - seed > 100:
            raise RuntimeError("could not generate enough designs absent from the training sweep")

    return pd.DataFrame(selected, columns=features)


designs = sample_fresh_designs(args.n, args.seed)
saved_targets = saved.get("targets", {})


def predict_one(values):
    x = np.log10(np.asarray(values, dtype=float).reshape(1, -1))
    result = {}
    for metric in PREDICTION_TARGETS:
        encoded = float(saved["models"][metric].predict(x)[0])
        transform = saved_targets.get(metric, (None, None, "linear"))[2]
        result[metric] = 10**encoded if transform == "log10" else encoded
    return result


def predict_batch(values):
    x = np.log10(np.asarray(values, dtype=float))
    result = {}
    for metric in PREDICTION_TARGETS:
        encoded = saved["models"][metric].predict(x)
        transform = saved_targets.get(metric, (None, None, "linear"))[2]
        result[metric] = 10**encoded if transform == "log10" else encoded
    return result


records = []
for index, (_, design) in enumerate(designs.iterrows(), start=1):
    params = design.to_dict()

    spice_started = time.perf_counter()
    try:
        actual = evaluate(**params, cl=LOAD_CAP)
        spice_time_s = time.perf_counter() - spice_started
        spice_error = ""
    except Exception as error:
        spice_time_s = time.perf_counter() - spice_started
        actual = {}
        spice_error = f"{type(error).__name__}: {error}"

    predict_started = time.perf_counter()
    predicted = predict_one(design[features].to_numpy())
    surrogate_time_s = time.perf_counter() - predict_started

    record = {
        "design_id": index,
        **params,
        "ngspice_valid": actual.get("valid", np.nan),
        "surrogate_valid": bool(
            predicted["sat_margin"] > SAT_MARGIN
            and predicted["v_tail"] > V_TAIL_MIN
        ),
        "sat_margin_spice": actual.get("sat_margin", np.nan),
        "sat_margin_surrogate": predicted["sat_margin"],
        "v_tail_spice": actual.get("v_tail", np.nan),
        "v_tail_surrogate": predicted["v_tail"],
        "power_w_spice": actual.get("power_w", np.nan),
        "power_w_surrogate": predicted.get("power_w", np.nan),
        "spice_time_s": spice_time_s,
        "surrogate_time_s": surrogate_time_s,
        "spice_error": spice_error,
    }
    for metric in METRICS:
        actual_value = actual.get(metric, np.nan)
        predicted_value = predicted[metric]
        error_pct = (
            abs(actual_value - predicted_value) / abs(actual_value) * 100
            if actual.get("valid", False)
            and np.isfinite(actual_value)
            and actual_value != 0
            else np.nan
        )
        record[f"{metric}_spice"] = actual_value
        record[f"{metric}_surrogate"] = predicted_value
        record[f"{metric}_error_pct"] = error_pct
    records.append(record)
    if index % 10 == 0 or index == args.n:
        print(f"Completed {index}/{args.n}", flush=True)

results = pd.DataFrame(records)
batch_started = time.perf_counter()
batch_predictions = predict_batch(designs[features].to_numpy())
batch_time_s = time.perf_counter() - batch_started
for metric in PREDICTION_TARGETS:
    if not np.allclose(
        results[f"{metric}_surrogate"],
        batch_predictions[metric],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise RuntimeError(f"batched predictions differ from single-design predictions for {metric}")
results["surrogate_batch_time_s_per_design"] = batch_time_s / args.n
args.output.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(args.output, index=False)

successful = results[results["spice_error"] == ""]
valid_designs = successful[successful["ngspice_valid"]]
invalid_designs = successful[~successful["ngspice_valid"]]
spice_total_s = float(valid_designs["spice_time_s"].sum())
surrogate_total_s = float(valid_designs["surrogate_time_s"].sum())
valid_batch_started = time.perf_counter()
if len(valid_designs):
    valid_batch_predictions = predict_batch(valid_designs[features].to_numpy())
    valid_batch_time_s = time.perf_counter() - valid_batch_started
    for metric in PREDICTION_TARGETS:
        if not np.allclose(
            valid_designs[f"{metric}_surrogate"],
            valid_batch_predictions[metric],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"batched predictions differ from single-design predictions "
                f"for valid designs and {metric}"
            )
else:
    valid_batch_predictions = {metric: np.array([]) for metric in PREDICTION_TARGETS}
    valid_batch_time_s = 0.0
predicted_valid = successful["surrogate_valid"]
actual_valid = successful["ngspice_valid"]
true_valid = int((actual_valid & predicted_valid).sum())
false_invalid = int((actual_valid & ~predicted_valid).sum())
false_valid = int((~actual_valid & predicted_valid).sum())
true_invalid = int((~actual_valid & ~predicted_valid).sum())
summary = {
    "topology": args.topology,
    "requested_designs": args.n,
    "successful_ngspice_runs": len(successful),
    "failed_ngspice_runs": len(results) - len(successful),
    "valid_designs": len(valid_designs),
    "invalid_designs": len(invalid_designs),
    "seed": args.seed,
    "load_capacitance_f": LOAD_CAP,
    "model": str(args.model),
    "model_load_time_s": model_load_s,
    "ngspice_total_time_s": spice_total_s,
    "surrogate_single_prediction_total_time_s": surrogate_total_s,
    "surrogate_batch_prediction_total_time_s": valid_batch_time_s,
    "ngspice_mean_time_s": spice_total_s / len(valid_designs) if len(valid_designs) else None,
    "surrogate_single_mean_time_s": surrogate_total_s / len(valid_designs) if len(valid_designs) else None,
    "surrogate_batch_mean_time_s_per_design": (
        valid_batch_time_s / len(valid_designs) if len(valid_designs) else None
    ),
    "single_design_speedup_excluding_model_load": (
        spice_total_s / surrogate_total_s if surrogate_total_s > 0 else None
    ),
    "batch_throughput_speedup_excluding_model_load": (
        spice_total_s / valid_batch_time_s if valid_batch_time_s > 0 else None
    ),
    "validity_classification": {
        "true_valid": true_valid,
        "valid_flagged_invalid": false_invalid,
        "invalid_flagged_valid": false_valid,
        "true_invalid": true_invalid,
        "valid_precision": (
            true_valid / (true_valid + false_valid) if true_valid + false_valid else None
        ),
        "valid_recall": (
            true_valid / (true_valid + false_invalid) if true_valid + false_invalid else None
        ),
        "invalid_recall": (
            true_invalid / (true_invalid + false_valid) if true_invalid + false_valid else None
        ),
    },
    "metrics": {},
}
if hasattr(topology, "COMMON_MODE_RANGE_V"):
    summary["common_mode_range_v"] = topology.COMMON_MODE_RANGE_V
    summary["common_mode_points"] = topology.COMMON_MODE_POINTS
for metric in METRICS:
    errors = valid_designs[f"{metric}_error_pct"].dropna()
    actual = valid_designs[f"{metric}_spice"]
    predicted = valid_designs[f"{metric}_surrogate"]
    finite = np.isfinite(actual) & np.isfinite(predicted)
    summary["metrics"][metric] = {
        "valid_comparisons": int(errors.size),
        "mean_absolute_percentage_error": float(errors.mean()) if errors.size else None,
        "median_absolute_percentage_error": float(errors.median()) if errors.size else None,
        "max_absolute_percentage_error": float(errors.max()) if errors.size else None,
        "mae": float((actual[finite] - predicted[finite]).abs().mean()) if finite.any() else None,
    }

args.summary.parent.mkdir(parents=True, exist_ok=True)
args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
metric_summary = pd.DataFrame([
    {
        "metric": metric,
        "valid comparisons": values["valid_comparisons"],
        "MAE": values["mae"],
        "mean absolute percentage error (%)": values["mean_absolute_percentage_error"],
        "median absolute percentage error (%)": values["median_absolute_percentage_error"],
        "maximum absolute percentage error (%)": values["max_absolute_percentage_error"],
    }
    for metric, values in summary["metrics"].items()
])
overview = pd.DataFrame([
    {"measure": "Requested designs", "value": summary["requested_designs"]},
    {"measure": "Successful ngspice runs", "value": summary["successful_ngspice_runs"]},
    {"measure": "Failed ngspice runs", "value": summary["failed_ngspice_runs"]},
    {"measure": "Valid designs (used for accuracy)", "value": summary["valid_designs"]},
    {"measure": "Invalid designs (excluded from accuracy)", "value": summary["invalid_designs"]},
    {"measure": "Seed", "value": summary["seed"]},
    {"measure": "Model load time (s, excluded from speedup)", "value": model_load_s},
    {"measure": "Valid-design ngspice total time (s)", "value": spice_total_s},
    {"measure": "Valid-design surrogate single-prediction total (s)", "value": surrogate_total_s},
    {"measure": "Valid-design surrogate batch-prediction total (s)", "value": valid_batch_time_s},
    {"measure": "Single-design speedup", "value": summary["single_design_speedup_excluding_model_load"]},
    {"measure": "Batch throughput speedup", "value": summary["batch_throughput_speedup_excluding_model_load"]},
])
validity_summary = pd.DataFrame([
    {"measure": "Actual valid; predicted valid", "count": true_valid},
    {"measure": "Actual valid; predicted invalid", "count": false_invalid},
    {"measure": "Actual invalid; predicted valid", "count": false_valid},
    {"measure": "Actual invalid; predicted invalid", "count": true_invalid},
])
detail_columns = [
    "design_id", *features,
    "ngspice_valid", "surrogate_valid",
    "sat_margin_spice", "sat_margin_surrogate",
    "v_tail_spice", "v_tail_surrogate",
    *[
        column
        for metric in METRICS
        for column in (
            f"{metric}_spice",
            f"{metric}_surrogate",
            f"{metric}_error_pct",
        )
    ],
    "spice_time_s", "surrogate_time_s", "spice_error",
]
def html_table(frame):
    return frame.to_html(
        index=False,
        escape=True,
        border=0,
        float_format=lambda value: f"{value:.6g}",
        na_rep="N/A",
    )


args.html.parent.mkdir(parents=True, exist_ok=True)
html_report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surrogate benchmark</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#24292f}}
h1,h2{{margin-top:1.6rem}}table{{border-collapse:collapse;display:block;overflow-x:auto;margin:1rem 0 2rem}}
th,td{{border:1px solid #d0d7de;padding:.4rem .6rem;text-align:right;white-space:nowrap}}
th{{background:#f6f8fa;position:sticky;top:0}}td:first-child,th:first-child{{text-align:left}}
a{{color:#0969da}}
</style>
</head>
<body>
<h1>Surrogate benchmark: {escape(args.topology)}</h1>
<p>Fresh designs compared against ngspice. Model: <code>{escape(str(args.model))}</code>;
training data excluded during sampling: <code>{escape(str(args.training_data))}</code>.</p>
{f"<p>Complementary-input performance and validity use the worst case over {topology.COMMON_MODE_POINTS} input common-mode points from {topology.COMMON_MODE_RANGE_V[0]:g} to {topology.COMMON_MODE_RANGE_V[1]:g} V.</p>" if hasattr(topology, "COMMON_MODE_RANGE_V") else ""}
<p>Download the <a href="{escape(args.output.name)}">per-design CSV</a> or
<a href="{escape(args.summary.name)}">summary JSON</a>.</p>
<h2>Run summary</h2>
{html_table(overview)}
<h2>Prediction accuracy</h2>
<p>Accuracy statistics below use only designs ngspice marked valid
(saturation margin &gt; {SAT_MARGIN * 1e3:g} mV and tail voltage &gt; {V_TAIL_MIN * 1e3:g} mV).
Invalid designs remain listed below with validity flags, but their metric errors are excluded.</p>
{html_table(metric_summary)}
<h2>Validity flag performance</h2>
<p>The surrogate estimates saturation margin and tail voltage, then applies
the same validity thresholds as ngspice.</p>
{html_table(validity_summary)}
<h2>Per-design results</h2>
<p>Error percentage is absolute error divided by the ngspice result and is
shown only for valid designs. Invalid designs are flagged; failed simulations
have no ngspice validity result.</p>
{html_table(results[detail_columns])}
</body>
</html>
"""
args.html.write_text(html_report, encoding="utf-8")
print(f"Per-design results: {args.output}")
print(f"Summary:            {args.summary}")
print(f"HTML report:        {args.html}")
print(
    f"Accuracy/timing sample: {len(valid_designs)} valid, "
    f"{len(invalid_designs)} invalid, {len(results) - len(successful)} failed"
)
print(f"Model load time:    {model_load_s:.6g} s (excluded from speedup)")
single_speedup = summary["single_design_speedup_excluding_model_load"]
batch_speedup = summary["batch_throughput_speedup_excluding_model_load"]
print(f"Single-design speedup: {single_speedup:.3g}x" if single_speedup is not None
      else "Single-design speedup: unavailable (no valid designs)")
print(
    f"Batch throughput speedup: {batch_speedup:.3g}x "
    f"({valid_batch_time_s:.6g} s for {len(valid_designs)} valid predictions)"
    if batch_speedup is not None
    else "Batch throughput speedup: unavailable (no valid designs)"
)
for metric, values in summary["metrics"].items():
    mape = values["mean_absolute_percentage_error"]
    print(f"{metric} mean absolute percentage error: {mape:.4g}%" if mape is not None
          else f"{metric} mean absolute percentage error: unavailable")
