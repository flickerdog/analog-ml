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

from src.ota5t import evaluate
from src.specs import LOAD_CAP
from sweep_ota import RANGES

ROOT = Path(__file__).resolve().parent
METRICS = ("gain_db", "gbw_hz", "pm_deg")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", type=Path, default=ROOT / "data/ota5t_surrogate.joblib")
parser.add_argument("--training-data", type=Path, default=ROOT / "data/ota5t_sobol.csv")
parser.add_argument("--n", type=int, default=100, help="number of fresh configurations")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "reports/surrogate_benchmark.csv",
    help="per-design benchmark results CSV",
)
parser.add_argument(
    "--summary",
    type=Path,
    default=ROOT / "reports/surrogate_benchmark_summary.json",
    help="aggregate results JSON",
)
parser.add_argument(
    "--html",
    type=Path,
    default=ROOT / "reports/surrogate_benchmark.html",
    help="HTML benchmark report",
)
args = parser.parse_args()

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
if saved.get("features") != features:
    parser.error(
        f"model features {saved.get('features')} do not match sweep features {features}"
    )
if saved.get("input_transform") != "log10":
    parser.error("benchmark expects a model trained with log10 input transformation")
if any(name not in saved.get("models", {}) for name in METRICS):
    parser.error(f"model must contain regressors for: {', '.join(METRICS)}")


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
    for metric in METRICS:
        encoded = float(saved["models"][metric].predict(x)[0])
        transform = saved_targets.get(metric, (None, None, "linear"))[2]
        result[metric] = 10**encoded if transform == "log10" else encoded
    return result


def predict_batch(values):
    x = np.log10(np.asarray(values, dtype=float))
    result = {}
    for metric in METRICS:
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
        "spice_time_s": spice_time_s,
        "surrogate_time_s": surrogate_time_s,
        "spice_error": spice_error,
    }
    for metric in METRICS:
        actual_value = actual.get(metric, np.nan)
        predicted_value = predicted[metric]
        error_pct = (
            abs(actual_value - predicted_value) / abs(actual_value) * 100
            if np.isfinite(actual_value) and actual_value != 0
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
for metric in METRICS:
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
spice_total_s = float(successful["spice_time_s"].sum())
surrogate_total_s = float(results["surrogate_time_s"].sum())
summary = {
    "requested_designs": args.n,
    "successful_ngspice_runs": len(successful),
    "failed_ngspice_runs": len(results) - len(successful),
    "seed": args.seed,
    "load_capacitance_f": LOAD_CAP,
    "model": str(args.model),
    "model_load_time_s": model_load_s,
    "ngspice_total_time_s": spice_total_s,
    "surrogate_single_prediction_total_time_s": surrogate_total_s,
    "surrogate_batch_prediction_total_time_s": batch_time_s,
    "ngspice_mean_time_s": spice_total_s / len(successful) if len(successful) else None,
    "surrogate_single_mean_time_s": surrogate_total_s / len(results) if len(results) else None,
    "surrogate_batch_mean_time_s_per_design": batch_time_s / args.n,
    "single_design_speedup_excluding_model_load": (
        spice_total_s / surrogate_total_s if surrogate_total_s > 0 else None
    ),
    "batch_throughput_speedup_excluding_model_load": (
        spice_total_s / batch_time_s if batch_time_s > 0 else None
    ),
    "metrics": {},
}
for metric in METRICS:
    errors = successful[f"{metric}_error_pct"].dropna()
    actual = successful[f"{metric}_spice"]
    predicted = successful[f"{metric}_surrogate"]
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
    {"measure": "Seed", "value": summary["seed"]},
    {"measure": "Model load time (s, excluded from speedup)", "value": model_load_s},
    {"measure": "ngspice total time (s)", "value": spice_total_s},
    {"measure": "Surrogate single-prediction total (s)", "value": surrogate_total_s},
    {"measure": "Surrogate batch-prediction total (s)", "value": batch_time_s},
    {"measure": "Single-design speedup", "value": summary["single_design_speedup_excluding_model_load"]},
    {"measure": "Batch throughput speedup", "value": summary["batch_throughput_speedup_excluding_model_load"]},
])
detail_columns = [
    "design_id", *features,
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
<h1>Surrogate benchmark</h1>
<p>Fresh designs compared against ngspice. Model: <code>{escape(str(args.model))}</code>;
training data excluded during sampling: <code>{escape(str(args.training_data))}</code>.</p>
<p>Download the <a href="{escape(args.output.name)}">per-design CSV</a> or
<a href="{escape(args.summary.name)}">summary JSON</a>.</p>
<h2>Run summary</h2>
{html_table(overview)}
<h2>Prediction accuracy</h2>
{html_table(metric_summary)}
<h2>Per-design results</h2>
<p>Error percentage is absolute error divided by the ngspice result. A blank
ngspice value indicates a failed simulation; its error is shown in the last column.</p>
{html_table(results[detail_columns])}
</body>
</html>
"""
args.html.write_text(html_report, encoding="utf-8")
print(f"Per-design results: {args.output}")
print(f"Summary:            {args.summary}")
print(f"HTML report:        {args.html}")
print(f"Model load time:    {model_load_s:.6g} s (excluded from speedup)")
print(
    "Single-design speedup: "
    f"{summary['single_design_speedup_excluding_model_load']:.3g}x"
)
print(
    "Batch throughput speedup: "
    f"{summary['batch_throughput_speedup_excluding_model_load']:.3g}x "
    f"({batch_time_s:.6g} s for {args.n} predictions)"
)
for metric, values in summary["metrics"].items():
    mape = values["mean_absolute_percentage_error"]
    print(f"{metric} mean absolute percentage error: {mape:.4g}%" if mape is not None
          else f"{metric} mean absolute percentage error: unavailable")
