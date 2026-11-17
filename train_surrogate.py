"""Train and evaluate a first surrogate for nominal OTA performance.

    python train_surrogate.py
    python run_all.py --surrogate

The reported metrics use a held-out test split. The saved model is then retrained
on all available rows, so it can be used for later design-space exploration.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.specs import classify

ROOT = Path(__file__).resolve().parent
FEATURES = ["w1", "l1", "w3", "l3", "itail"]
TARGETS = {
    "gain_db": ("DC gain", "dB", "linear"),
    "gbw_hz": ("GBW", "MHz", "log10"),
    "pm_deg": ("Phase margin", "degrees", "linear"),
    "sat_margin": ("Saturation margin", "mV", "linear"),
    "v_tail": ("Tail voltage", "mV", "linear"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=ROOT / "data/ota5t_sobol.csv")
parser.add_argument("--metrics", type=Path, default=ROOT / "reports/surrogate_metrics.csv")
parser.add_argument("--model", type=Path, default=ROOT / "data/ota5t_surrogate.joblib")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--trees", type=int, default=300)
args = parser.parse_args()

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
        scale = 1e-6  # Report GBW in MHz.
    else:
        baseline = baseline_encoded
        actual = values[test_idx]
        scale = 1e3 if column in {"sat_margin", "v_tail"} else 1

    test_predictions[column] = prediction.copy()
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
precision = true_positives / max(int(predicted_passes.sum()), 1)
recall = true_positives / max(int(actual_passes.sum()), 1)

args.metrics.parent.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(results)
metrics.to_csv(args.metrics, index=False)

args.model.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "features": FEATURES,
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
print(f"Metrics: {args.metrics}")
print(f"Model:   {args.model}")
