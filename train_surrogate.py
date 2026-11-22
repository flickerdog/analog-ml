"""Train and evaluate a first surrogate for nominal OTA performance.

    python train_surrogate.py
    python run_all.py --surrogate

The reported metrics use a held-out test split. The saved model is then retrained
on all available rows, so it can be used for later design-space exploration.
"""
import argparse
from html import escape
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.specs import classify
from src.topology import load_topology, topology_names

ROOT = Path(__file__).resolve().parent
TARGETS = {
    "gain_db": ("DC gain", "dB", "linear"),
    "gbw_hz": ("GBW", "MHz", "log10"),
    "pm_deg": ("Phase margin", "degrees", "linear"),
    "sat_margin": ("Saturation margin", "mV", "linear"),
    "v_tail": ("Tail voltage", "mV", "linear"),
}
DISPLAY_SCALES = {
    "gbw_hz": 1e-6,
    "sat_margin": 1e3,
    "v_tail": 1e3,
    "power_w": 1e6,
    "power_w": 1e6,
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--topology", choices=topology_names(), default="ota5t")
parser.add_argument("--data", type=Path)
parser.add_argument("--metrics", type=Path)
parser.add_argument("--model", type=Path)
parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trees", type=int, default=300)
args = parser.parse_args()
topology = load_topology(args.topology)
FEATURES = list(topology.RANGES)
if getattr(topology, "TRAIN_POWER", False):
    TARGETS["power_w"] = ("Power", "uW", "log10")
args.data = args.data or topology.DATA_FILE
args.model = args.model or topology.MODEL_FILE
suffix = "" if args.topology == "ota5t" else f"_{args.topology}"
args.metrics = args.metrics or ROOT / f"reports/surrogate_metrics{suffix}.csv"
prediction_name = f"surrogate_test_predictions{suffix}.csv"
actual_plot_name = f"surrogate_actual_vs_predicted{suffix}.png"
error_plot_name = f"surrogate_prediction_errors{suffix}.png"
report_name = f"surrogate_report{suffix}.html"

if not 0 < args.test_size < 1:
    parser.error("--test-size must be between 0 and 1")
if args.trees < 1:
    parser.error("--trees must be at least 1")
if not args.data.is_file():
    parser.error(f"data file not found: {args.data}")

df = pd.read_csv(args.data)
required = FEATURES + list(TARGETS)
missing = sorted(set(required) - set(df.columns))
if missing:
    parser.error(f"data file is missing required columns: {', '.join(missing)}")

data = df[required].replace([np.inf, -np.inf], np.nan).dropna()
if (data[FEATURES] <= 0).any().any():
    parser.error("all design input values must be positive for log-space training")
for column, (_name, _unit, transform) in TARGETS.items():
    if transform == "log10":
        data = data[data[column] > 0]
if len(data) < 10:
    parser.error(f"at least 10 complete simulation rows are required; found {len(data)}")

# The sweep samples dimensions and current logarithmically, so model in that space.
x = np.log10(data[FEATURES].to_numpy())
indices = np.arange(len(data))
train_idx, test_idx = train_test_split(
    indices, test_size=args.test_size, random_state=args.seed
)
results = []
models = {}
test_predictions = {}
test_actuals = {}

for column, (name, unit, transform) in TARGETS.items():
    values = data[column].to_numpy()
    if transform == "log10":
        if (values <= 0).any():
            parser.error(f"{column} must be positive for log-space training")
        encoded = np.log10(values)
    else:
        encoded = values

    model = RandomForestRegressor(
        n_estimators=args.trees,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x[train_idx], encoded[train_idx])
    prediction = model.predict(x[test_idx])
    baseline_encoded = np.full(len(test_idx), encoded[train_idx].mean())

    if transform == "log10":
        prediction = 10**prediction
        baseline = 10**baseline_encoded
        actual = values[test_idx]
        scale = DISPLAY_SCALES.get(column, 1)
    else:
        baseline = baseline_encoded
        actual = values[test_idx]
        scale = 1e3 if column in {"sat_margin", "v_tail"} else 1

    test_predictions[column] = prediction.copy()
    test_actuals[column] = values[test_idx].copy()
    prediction *= scale
    baseline *= scale
    actual = actual * scale
    results.append({
        "metric": name,
        "unit": unit,
        "mae": mean_absolute_error(actual, prediction),
        "rmse": np.sqrt(mean_squared_error(actual, prediction)),
        "r2": r2_score(actual, prediction),
        "mean_baseline_mae": mean_absolute_error(actual, baseline),
    })

    # Refit on every row only after measuring performance on the untouched split.
    models[column] = RandomForestRegressor(
        n_estimators=args.trees,
        min_samples_leaf=2,
        random_state=args.seed,
        n_jobs=-1,
    ).fit(x, encoded)

actual_class = classify(data.iloc[test_idx])
predicted_class = classify(pd.DataFrame(test_predictions, index=data.iloc[test_idx].index))
actual_passes = actual_class == "passes"
predicted_passes = predicted_class == "passes"
true_positives = int((actual_passes & predicted_passes).sum())
false_positives = int((~actual_passes & predicted_passes).sum())
false_negatives = int((actual_passes & ~predicted_passes).sum())
true_negatives = int((~actual_passes & ~predicted_passes).sum())
precision = true_positives / max(int(predicted_passes.sum()), 1)
recall = true_positives / max(int(actual_passes.sum()), 1)

args.report_dir.mkdir(parents=True, exist_ok=True)
args.metrics.parent.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(results)
metrics.to_csv(args.metrics, index=False)

predictions = data.iloc[test_idx][FEATURES].copy().reset_index(drop=True)
for column, (name, unit, transform) in TARGETS.items():
    scale = DISPLAY_SCALES.get(column, 1)
    predictions[f"{column}_actual_{unit}"] = test_actuals[column] * scale
    predictions[f"{column}_predicted_{unit}"] = test_predictions[column] * scale
predictions["class_actual"] = actual_class.to_numpy()
predictions["class_predicted"] = predicted_class.to_numpy()
predictions["predicted_pass_but_failed"] = (
    (predicted_class == "passes") & (actual_class != "passes")
).to_numpy()
predictions_path = args.report_dir / prediction_name
predictions.to_csv(predictions_path, index=False)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (column, (name, unit, transform)) in zip(axes.flat, TARGETS.items()):
    scale = DISPLAY_SCALES.get(column, 1)
    actual = test_actuals[column] * scale
    predicted = test_predictions[column] * scale
    ax.scatter(actual, predicted, s=12, alpha=0.55, linewidths=0)
    low = min(actual.min(), predicted.min())
    high = max(actual.max(), predicted.max())
    ax.plot([low, high], [low, high], "k--", lw=1, label="perfect prediction")
    if transform == "log10":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_title(name)
    ax.set_xlabel(f"ngspice result ({unit})")
    ax.set_ylabel(f"surrogate prediction ({unit})")
    ax.legend(fontsize=8)
axes.flat[-1].axis("off")
fig.suptitle("Held-out test set: predicted versus simulated values")
fig.tight_layout()
actual_plot = args.report_dir / actual_plot_name
fig.savefig(actual_plot, dpi=160)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for ax, (column, (name, unit, transform)) in zip(axes.flat, TARGETS.items()):
    scale = 1e-6 if transform == "log10" else (
        1e3 if column in {"sat_margin", "v_tail"} else
        1e6 if column == "power_w" else 1
    )
    actual = test_actuals[column] * scale
    residual = (test_actuals[column] - test_predictions[column]) * scale
    ax.scatter(actual, residual, s=12, alpha=0.55, linewidths=0)
    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_title(name)
    ax.set_xlabel(f"ngspice result ({unit})")
    ax.set_ylabel(f"ngspice − prediction ({unit})")
axes.flat[-1].axis("off")
fig.suptitle("Held-out test set: prediction errors (zero is perfect)")
fig.tight_layout()
error_plot = args.report_dir / error_plot_name
fig.savefig(error_plot, dpi=160)
plt.close(fig)

confusion = pd.DataFrame(
    [[true_positives, false_negatives], [false_positives, true_negatives]],
    index=["Actually passes", "Actually does not pass"],
    columns=["Predicted passes", "Predicted does not pass"],
)
false_passes = predictions[predictions["predicted_pass_but_failed"]].copy()
display_columns = list(FEATURES)
for column, (_name, unit, _transform) in TARGETS.items():
    display_columns.extend([
        f"{column}_actual_{unit}",
        f"{column}_predicted_{unit}",
    ])
display_columns = [
    column for column in display_columns
    if column in false_passes.columns
]
false_pass_html = (
    false_passes[display_columns].to_html(index=False, float_format=lambda x: f"{x:.4g}")
    if len(false_passes)
    else "<p>None in this held-out test split.</p>"
)
metrics_html = metrics.to_html(index=False, float_format=lambda x: f"{x:.4g}")
confusion_html = confusion.to_html(float_format=lambda x: f"{x:.4g}")
report = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>{escape(args.topology)} OTA surrogate evaluation</title>
<style>
body{{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;line-height:1.5}}
table{{border-collapse:collapse;margin:1rem 0 2rem}}th,td{{border:1px solid #ccc;padding:.4rem .65rem}}
th{{background:#f2f2f2}}img{{max-width:100%;height:auto}}.note{{background:#f6f8fa;padding:1rem}}
</style></head>
<body>
<h1>{escape(args.topology)} OTA surrogate: held-out evaluation</h1>
<p>Data: <code>{escape(str(args.data))}</code>. Complete rows: {len(data)}.
Training rows: {len(train_idx)}. Test rows not used for fitting: {len(test_idx)}.
Seed: {args.seed}.</p>
<h2>Training method</h2>
<p>This script uses scikit-learn's <code>RandomForestRegressor</code>: {len(TARGETS)}
separate regressors predict {", ".join(name for name, _unit, _transform in TARGETS.values())}. Each forest averages {args.trees} decision trees. Tree leaves
predict from training examples in that region of input space. The {len(FEATURES)} design
inputs are transformed with <code>log10</code>; GBW targets are also trained
in <code>log10</code> space. The held-out split evaluates models trained on
{len(train_idx)} rows. After evaluation, those models are independently
refit on all {len(data)} complete rows and saved for future use.</p>
<div class="note"><strong>How to read this:</strong> these plots compare model
predictions with ngspice results for designs withheld during training. Points
near the dashed diagonal are accurate. This tests nominal circuit metrics only;
it does not test Monte Carlo mismatch or input offset. The final saved model is
refit on all rows after this evaluation.</div>
<h2>Error summary</h2>
<p>MAE is the average absolute prediction error in the listed unit. RMSE gives
more weight to large errors. R² compares the model with predicting the mean;
higher is better. The last column is the MAE of that simple mean-only baseline.</p>
{metrics_html}
<h2>Predicted versus simulated</h2>
<p>Each dot is one held-out design. GBW axes are logarithmic because its values
span a wide range.</p>
<img src="{actual_plot.name}" alt="Predicted versus simulated plots">
<h2>Prediction errors</h2>
<p>Error is defined as ngspice result minus model prediction; zero is perfect.</p>
<img src="{error_plot.name}" alt="Prediction error plots">
<h2>Pass/fail classification</h2>
<p>Classification applies the current requirements and validity thresholds
from <code>specs.yaml</code> to the held-out actual and predicted metrics.
Pass precision: {precision:.1%}; pass recall: {recall:.1%};
predicted passes that failed in ngspice: {false_positives}.</p>
{confusion_html}
<h3>False passes (important to inspect)</h3>
<p>These designs were predicted to pass all configured requirements but did
not pass when simulated in ngspice. Verify proposed designs with ngspice.</p>
{false_pass_html}
<p>All held-out predictions and design inputs are in
<code>{predictions_path.name}</code>.</p>
</body></html>"""
report_path = args.report_dir / report_name
report_path.write_text(report, encoding="utf-8")

args.model.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "features": FEATURES,
    "topology": args.topology,
    "targets": TARGETS,
    "models": models,
    "input_transform": "log10",
    "training_rows": len(data),
}, args.model)

print(f"Training rows: {len(train_idx)}; held-out test rows: {len(test_idx)}")
print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print(
    "Held-out spec classification: "
    f"{int((actual_class == predicted_class).sum())}/{len(test_idx)} correct; "
    f"pass precision={precision:.1%}, pass recall={recall:.1%}, "
    f"false passes={false_positives}"
)
print(f"HTML report: {report_path}")
print(f"Plots:       {actual_plot}, {error_plot}")
print(f"Predictions: {predictions_path}")
print(f"Metrics: {args.metrics}")
print(f"Model:   {args.model}")
