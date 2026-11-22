"""Compare saved-surrogate-guided optimization with GP Bayesian optimization.

    python3 optimize_ota.py --scenario minimize_power
    python3 optimize_ota.py --scenario minimize_power --iterations 50 --seed 123

Both methods start from the existing sweep and evaluate each new candidate
with ngspice. The saved random-forest model is not retrained or overwritten.
"""
import argparse
import html
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import yaml
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from scipy.spatial import cKDTree
from scipy.stats import qmc

from src.specs import LOAD_CAP
from src.topology import load_topology, topology_names

ROOT = Path(__file__).resolve().parent
TARGETS = ("power_w", "gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail")

parser = argparse.ArgumentParser(
    description="Compare saved-surrogate-guided optimization with GP Bayesian optimization."
)
parser.add_argument("--topology", choices=topology_names(), default="ota5t")
parser.add_argument("--config", type=Path)
parser.add_argument("--scenario", default="minimize_power")
parser.add_argument("--model", type=Path)
parser.add_argument("--iterations", type=int, default=200, help="ngspice evaluations per search method")
parser.add_argument("--initial-size", type=int, default=128, help="existing sweep rows used to initialize each method")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
args = parser.parse_args()
topology = load_topology(args.topology)
RANGES = topology.RANGES
VDD = topology.VDD
evaluate = topology.evaluate
SURROGATE_TARGETS = topology.SURROGATE_TARGETS
TRAINING_DATA = topology.DATA_FILE
args.model = args.model or topology.MODEL_FILE
args.config = args.config or topology.SCENARIO_FILE

if args.iterations < 1:
    parser.error("--iterations must be at least 1")
if args.initial_size < 10:
    parser.error("--initial-size must be at least 10")
if not args.config.is_file():
    parser.error(f"scenario config not found: {args.config}")
if not TRAINING_DATA.is_file():
    parser.error(f"existing sweep data not found: {TRAINING_DATA}")
if not args.model.is_file():
    parser.error(f"saved surrogate model not found: {args.model}")
with args.config.open(encoding="utf-8") as file:
    config = yaml.safe_load(file)
scenarios = config.get("scenarios", {})
if args.scenario not in scenarios:
    parser.error(f"unknown scenario {args.scenario!r}; choose from: {', '.join(scenarios)}")
scenario = scenarios[args.scenario]
objective_name = scenario.get("objective")
direction = scenario.get("direction")
constraints_config = scenario.get("constraints", {})
if objective_name not in TARGETS:
    parser.error(f"objective must be one of: {', '.join(TARGETS)}")
if direction not in {"minimize", "maximize"}:
    parser.error("scenario direction must be 'minimize' or 'maximize'")
if not constraints_config:
    parser.error("scenario must define at least one constraint")
for name, limits in constraints_config.items():
    if name not in TARGETS or name == objective_name:
        parser.error(f"invalid constraint target: {name}")
    if not isinstance(limits, dict) or set(limits) not in ({"min"}, {"max"}):
        parser.error(f"constraint {name} must specify exactly one of min or max")

features = list(RANGES)
lower_log = np.log10([RANGES[name][0] for name in features])
upper_log = np.log10([RANGES[name][1] for name in features])
torch.set_num_threads(1)
torch.set_default_dtype(torch.double)

training = pd.read_csv(TRAINING_DATA)
saved_model = joblib.load(args.model)
required = set(features) | set(TARGETS)
missing = sorted(required - set(training.columns))
if missing:
    parser.error(f"training sweep is missing columns: {', '.join(missing)}")
training = training.replace([np.inf, -np.inf], np.nan).dropna(subset=list(required)).reset_index(drop=True)
if len(training) < args.initial_size:
    parser.error(f"need {args.initial_size} complete sweep rows; found {len(training)}")
if saved_model.get("features") != features:
    parser.error("saved model features do not match topology parameter ranges")
if saved_model.get("topology", "ota5t") != args.topology:
    parser.error(f"model was trained for topology {saved_model.get('topology', 'ota5t')}")
if saved_model.get("input_transform") != "log10":
    parser.error("saved surrogate must use log10 input transformation")
if any(name not in saved_model.get("models", {}) for name in SURROGATE_TARGETS):
    parser.error(
        "saved surrogate must contain regressors for: "
        + ", ".join(SURROGATE_TARGETS)
    )

training_log_x = np.log10(training[features].to_numpy(dtype=float))
training_x = (training_log_x - lower_log) / (upper_log - lower_log)
training_y = training[list(TARGETS)].to_numpy(dtype=float)
rng = np.random.default_rng(args.seed)


def target_satisfies(values):
    for name, limits in constraints_config.items():
        value = values[name]
        if "min" in limits and (
            value <= limits["min"] if name in {"sat_margin", "v_tail"}
            else value < limits["min"]
        ):
            return False
        if "max" in limits and value > limits["max"]:
            return False
    return True


def best_feasible(rows):
    feasible = [row for row in rows if target_satisfies(row)]
    if not feasible:
        return None
    choose_min = direction == "minimize"
    return min(feasible, key=lambda row: row[objective_name]) if choose_min else max(
        feasible, key=lambda row: row[objective_name]
    )


all_training_records = training[list(TARGETS) + features].to_dict("records")
known_incumbent = best_feasible(all_training_records)
incumbent_index = (
    next(
        index for index, row in enumerate(all_training_records)
        if row == known_incumbent
    )
    if known_incumbent is not None
    else None
)
if incumbent_index is None:
    initial_indices = rng.choice(len(training), size=args.initial_size, replace=False)
else:
    remaining = np.delete(np.arange(len(training)), incumbent_index)
    random_initial = rng.choice(remaining, size=args.initial_size - 1, replace=False)
    initial_indices = np.concatenate(([incumbent_index], random_initial))
initial_x = training_x[initial_indices]
initial_y = training_y[initial_indices]
initial_rows = training.iloc[initial_indices].copy()


def make_model(x, y):
    tx = torch.as_tensor(x, dtype=torch.double)
    ty = torch.as_tensor(y, dtype=torch.double)
    models = [
        SingleTaskGP(
            tx,
            ty[:, index:index + 1],
            outcome_transform=Standardize(m=1),
        )
        for index in range(len(TARGETS))
    ]
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def make_acquisition(model, x_baseline):
    objective_index = TARGETS.index(objective_name)
    sign = -1.0 if direction == "minimize" else 1.0
    objective = GenericMCObjective(
        lambda samples, X=None: sign * samples[..., objective_index]
    )
    constraint_functions = []
    for name, limits in constraints_config.items():
        index = TARGETS.index(name)
        if "min" in limits:
            threshold = float(limits["min"])
            constraint_functions.append(
                lambda samples, idx=index, bound=threshold: bound - samples[..., idx]
            )
        else:
            threshold = float(limits["max"])
            constraint_functions.append(
                lambda samples, idx=index, bound=threshold: samples[..., idx] - bound
            )
    return qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=torch.as_tensor(x_baseline, dtype=torch.double),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128]), seed=args.seed),
        objective=objective,
        constraints=constraint_functions,
        prune_baseline=True,
    )


def normalized_candidate_pool(sampler, seen_tree, training_tree, size=1024):
    while True:
        unit = sampler.random(size)
        physical_log = lower_log + unit * (upper_log - lower_log)
        candidates = (physical_log - lower_log) / (upper_log - lower_log)
        distances, _ = training_tree.query(candidates, k=1)
        if seen_tree is not None:
            seen_distances, _ = seen_tree.query(candidates, k=1)
            distances = np.minimum(distances, seen_distances)
        candidates = candidates[distances > 1e-5]
        if len(candidates):
            return candidates


def predict_surrogate(unit_x):
    log_x = lower_log + np.asarray(unit_x) * (upper_log - lower_log)
    model_x = log_x.reshape(-1, len(features))
    physical_x = 10**log_x
    result = {"power_w": topology.power_estimate(physical_x)}
    transforms = saved_model.get("targets", {})
    for name in SURROGATE_TARGETS:
        if name == "power_w" and name not in saved_model["models"]:
            continue
        encoded = saved_model["models"][name].predict(model_x)
        transform = transforms.get(name, (None, None, "linear"))[2]
        result[name] = 10**encoded if transform == "log10" else encoded
    return result


def constraint_violations(predictions):
    total = np.zeros(len(next(iter(predictions.values()))))
    for name, limits in constraints_config.items():
        values = predictions[name]
        if "min" in limits:
            threshold = float(limits["min"])
            scale = max(abs(threshold), float(np.std(training[name])), 1e-12)
            total += np.maximum(threshold - values, 0) / scale
        else:
            threshold = float(limits["max"])
            scale = max(abs(threshold), float(np.std(training[name])), 1e-12)
            total += np.maximum(values - threshold, 0) / scale
    return total


def candidate_order(unit_x):
    predictions = predict_surrogate(unit_x)
    feasible = np.ones(len(unit_x), dtype=bool)
    for name, limits in constraints_config.items():
        if "min" in limits:
            feasible &= predictions[name] >= limits["min"]
            if name in {"sat_margin", "v_tail"}:
                feasible &= predictions[name] > limits["min"]
        else:
            feasible &= predictions[name] <= limits["max"]
    violations = constraint_violations(predictions)
    objective = predictions[objective_name]
    objective_key = objective if direction == "minimize" else -objective
    return np.lexsort((objective_key, violations, ~feasible)), predictions, feasible


def evaluate_design(unit_x):
    log_values = lower_log + np.asarray(unit_x) * (upper_log - lower_log)
    values = 10**log_values
    params = dict(zip(features, values))
    started = time.perf_counter()
    actual = evaluate(**params, cl=LOAD_CAP)
    elapsed = time.perf_counter() - started
    return {**params, **actual, "spice_time_s": elapsed}


initial_records = initial_rows[list(TARGETS) + features].to_dict("records")
gp_x = initial_x.copy()
gp_y = initial_y.copy()
gp_records = list(initial_records)
rf_records = list(initial_records)
gp_seen_x = training_x.copy()
training_tree = cKDTree(training_x)
gp_sampler = qmc.Sobol(d=len(features), scramble=True, seed=args.seed + 1)
rf_compute_started = time.perf_counter()
rf_sampler = qmc.Sobol(d=len(features), scramble=True, seed=args.seed + 2)
rf_pool_unit = rf_sampler.random_base2(m=16)
rf_distances, _ = training_tree.query(rf_pool_unit, k=1)
rf_pool_unit = rf_pool_unit[rf_distances > 1e-5]
rf_ranked_indices, rf_predictions, rf_predicted_feasible = candidate_order(rf_pool_unit)
rf_ranked_candidates = rf_pool_unit[rf_ranked_indices]
rf_setup_compute_s = time.perf_counter() - rf_compute_started
rf_candidate_cursor = 0
records = []
fit_times = []
acquisition_times = []
rf_selection_times = []

for iteration in range(1, args.iterations + 1):
    fit_started = time.perf_counter()
    model = make_model(gp_x, gp_y)
    fit_times.append(time.perf_counter() - fit_started)

    acquisition_started = time.perf_counter()
    acquisition = make_acquisition(model, gp_x)
    seen_tree = cKDTree(gp_seen_x)
    candidate_pool = normalized_candidate_pool(gp_sampler, seen_tree, training_tree)
    scores = []
    for offset in range(0, len(candidate_pool), 256):
        batch = torch.as_tensor(candidate_pool[offset:offset + 256], dtype=torch.double)
        with torch.no_grad():
            score = acquisition(batch.unsqueeze(-2))
        scores.extend(score.detach().cpu().numpy().reshape(-1))
    candidate_index = int(np.argmax(scores))
    gp_candidate = candidate_pool[candidate_index]
    acquisition_times.append(time.perf_counter() - acquisition_started)

    gp_spice_started = time.perf_counter()
    try:
        gp_actual = evaluate_design(gp_candidate)
        gp_actual["method"] = "gp_bayesian_optimization"
        gp_actual["iteration"] = iteration
        gp_actual["acquisition_value"] = float(scores[candidate_index])
        gp_actual["surrogate_feasible"] = target_satisfies(
            {name: gp_actual[name] for name in TARGETS}
        )
        records.append(gp_actual)
        gp_x = np.vstack((gp_x, gp_candidate))
        gp_y = np.vstack((gp_y, [gp_actual[name] for name in TARGETS]))
        gp_records.append(gp_actual)
    except Exception as error:
        records.append({
            "method": "gp_bayesian_optimization",
            "iteration": iteration,
            **dict(zip(features, 10 ** (lower_log + gp_candidate * (upper_log - lower_log)))),
            "spice_error": f"{type(error).__name__}: {error}",
            "spice_time_s": time.perf_counter() - gp_spice_started,
        })
    gp_seen_x = np.vstack((gp_seen_x, gp_candidate))

    rf_selection_started = time.perf_counter()
    while rf_candidate_cursor < len(rf_ranked_candidates):
        rf_candidate = rf_ranked_candidates[rf_candidate_cursor]
        rf_candidate_cursor += 1
        distance = training_tree.query(rf_candidate, k=1)[0]
        if distance > 1e-5:
            break
    else:
        raise RuntimeError("saved-surrogate candidate pool was exhausted")
    rf_selection_times.append(time.perf_counter() - rf_selection_started)
    rf_spice_started = time.perf_counter()
    try:
        rf_actual = evaluate_design(rf_candidate)
        rf_actual["method"] = "saved_surrogate_search"
        rf_actual["iteration"] = iteration
        rf_idx = rf_ranked_indices[rf_candidate_cursor - 1]
        rf_actual["surrogate_feasible"] = bool(rf_predicted_feasible[rf_idx])
        rf_actual["predicted_objective"] = float(rf_predictions[objective_name][rf_idx])
        rf_actual["acquisition_value"] = np.nan
        records.append(rf_actual)
        rf_records.append(rf_actual)
    except Exception as error:
        records.append({
            "method": "saved_surrogate_search",
            "iteration": iteration,
            **dict(zip(features, 10 ** (lower_log + rf_candidate * (upper_log - lower_log)))),
            "spice_error": f"{type(error).__name__}: {error}",
            "spice_time_s": time.perf_counter() - rf_spice_started,
        })

    if iteration % 5 == 0 or iteration == args.iterations:
        print(f"Completed GP and saved-surrogate evaluation {iteration}/{args.iterations}", flush=True)

gp_best = best_feasible(gp_records)
rf_best = best_feasible(rf_records)
initial_best = best_feasible(initial_records)
scenario_stem = args.scenario.replace("/", "_")
if args.topology != "ota5t":
    scenario_stem = f"{args.topology}_{scenario_stem}"
run_tag = f"{args.iterations}_seed{args.seed}"
args.output_dir.mkdir(parents=True, exist_ok=True)
trials_path = args.output_dir / f"optimization_{scenario_stem}_{run_tag}_trials.csv"
summary_path = args.output_dir / f"optimization_{scenario_stem}_{run_tag}_summary.json"
html_path = args.output_dir / f"optimization_{scenario_stem}_{run_tag}.html"
plot_path = args.output_dir / f"optimization_{scenario_stem}_{run_tag}_design_space.png"
results = pd.DataFrame(records)
results.to_csv(trials_path, index=False)


def report_row(best):
    if best is None:
        return None
    return {
        "objective": objective_name,
        "objective_value": float(best[objective_name]),
        **{name: float(best[name]) for name in features},
        **{name: float(best[name]) for name in TARGETS if name != objective_name},
    }


summary = {
    "scenario": args.scenario,
    "topology": args.topology,
    "description": scenario.get("description", ""),
    "objective": objective_name,
    "direction": direction,
    "constraints": constraints_config,
    "seed": args.seed,
    "initial_sweep_rows": args.initial_size,
    "ngspice_budget_per_method": args.iterations,
    "ngspice_evaluations": {
        "gp_bayesian_optimization": int((results.method == "gp_bayesian_optimization").sum()),
        "saved_surrogate_search": int((results.method == "saved_surrogate_search").sum()),
    },
    "initial_best_feasible": report_row(initial_best),
    "gp_bayesian_optimization_best_feasible": report_row(gp_best),
    "saved_surrogate_search_best_feasible": report_row(rf_best),
    "gp_fit_mean_s": float(np.mean(fit_times)),
    "gp_acquisition_mean_s": float(np.mean(acquisition_times)),
    "ngspice_failures": int(results.get("spice_error", pd.Series(dtype=str)).notna().sum()),
    "model": str(args.model),
    "training_data": str(TRAINING_DATA),
}
if hasattr(topology, "COMMON_MODE_RANGE_V"):
    summary["common_mode_range_v"] = topology.COMMON_MODE_RANGE_V
    summary["common_mode_points"] = topology.COMMON_MODE_POINTS
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
comparison = pd.DataFrame([
    {"method": "Existing shared initial sample", **(report_row(initial_best) or {})},
    {"method": "Saved surrogate guided (ngspice verified)", **(report_row(rf_best) or {})},
    {"method": "GP Bayesian optimization (ngspice verified)", **(report_row(gp_best) or {})},
])
if "objective_value" not in comparison:
    comparison["objective_value"] = np.nan
comparison_display = comparison.copy()
display_units = {
    "power_w": (1e6, "power (µW)"),
    "gbw_hz": (1e-6, "GBW (MHz)"),
    "sat_margin": (1e3, "sat_margin (mV)"),
    "v_tail": (1e3, "v_tail (mV)"),
    "itail": (1e6, "itail (µA)"),
}
objective_scale, objective_unit = display_units.get(objective_name, (1.0, objective_name))
objective_header_unit = objective_unit.rsplit("(", 1)[-1].rstrip(")")
comparison_display["objective"] = objective_unit
comparison_display["objective_value"] *= objective_scale
comparison_display = comparison_display.rename(
    columns={"objective_value": f"objective value ({objective_header_unit})"}
)
for column, (scale, label) in display_units.items():
    if column in comparison_display:
        comparison_display[column] *= scale
        comparison_display = comparison_display.rename(columns={column: label})

gp_compute_s = float(sum(fit_times) + sum(acquisition_times))
rf_compute_s = float(rf_setup_compute_s + sum(rf_selection_times))
gp_spice_s = float(results.loc[results.method == "gp_bayesian_optimization", "spice_time_s"].sum())
rf_spice_s = float(results.loc[results.method == "saved_surrogate_search", "spice_time_s"].sum())
timing = {
    "gp_bayesian_optimization": {
        "optimization_computation_s": gp_compute_s,
        "ngspice_evaluation_s": gp_spice_s,
        "total_method_time_s": gp_compute_s + gp_spice_s,
        "mean_computation_per_evaluation_s": gp_compute_s / args.iterations,
    },
    "saved_surrogate_search": {
        "optimization_computation_s": rf_compute_s,
        "ngspice_evaluation_s": rf_spice_s,
        "total_method_time_s": rf_compute_s + rf_spice_s,
        "mean_computation_per_evaluation_s": rf_compute_s / args.iterations,
    },
}
gp_total_s = timing["gp_bayesian_optimization"]["total_method_time_s"]
rf_total_s = timing["saved_surrogate_search"]["total_method_time_s"]
time_savings_factor = gp_total_s / rf_total_s if rf_total_s > 0 else float("inf")
summary["timing_seconds"] = timing
summary["saved_surrogate_time_savings_factor"] = float(time_savings_factor)
summary["saved_surrogate_wall_time_reduction_percent"] = float(
    100 * (gp_total_s - rf_total_s) / gp_total_s
)
summary["topology"] = args.topology
summary["gp_fit_mean_s"] = float(np.mean(fit_times))
summary["gp_acquisition_mean_s"] = float(np.mean(acquisition_times))
summary["gp_wall_time_s"] = timing["gp_bayesian_optimization"]["total_method_time_s"]
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
timing_display = pd.DataFrame([
    {
        "method": "Saved-surrogate search",
        "optimization computation (s)": rf_compute_s,
        "ngspice evaluations (s)": rf_spice_s,
        "total method time (s)": timing["saved_surrogate_search"]["total_method_time_s"],
        "computation per evaluation (s)": timing["saved_surrogate_search"]["mean_computation_per_evaluation_s"],
    },
    {
        "method": "GP Bayesian optimization",
        "optimization computation (s)": gp_compute_s,
        "ngspice evaluations (s)": gp_spice_s,
        "total method time (s)": timing["gp_bayesian_optimization"]["total_method_time_s"],
        "computation per evaluation (s)": timing["gp_bayesian_optimization"]["mean_computation_per_evaluation_s"],
    },
])
timing_display["time savings factor (x)"] = [
    time_savings_factor,
    1.0,
]

fig, axes = plt.subplots(5, 2, figsize=(13, 22), squeeze=False)
axes = axes.flat
pair_index = 0
training_feasible = np.ones(len(training), dtype=bool)
for name, limits in constraints_config.items():
    if "min" in limits:
        training_feasible &= training[name] >= limits["min"]
        if name in {"sat_margin", "v_tail"}:
            training_feasible &= training[name] > limits["min"]
    else:
        training_feasible &= training[name] <= limits["max"]
plot_sources = []
for method, marker, size in (
    ("saved_surrogate_search", "o", 24),
    ("gp_bayesian_optimization", "^", 28),
):
    method_rows = results[results.method == method].dropna(subset=features + [objective_name])
    method_feasible = np.array([
        target_satisfies(row)
        for row in method_rows[list(TARGETS)].to_dict("records")
    ], dtype=bool)
    plot_sources.append((method, method_rows[features], method_feasible, marker, size))

for i, x_name in enumerate(features):
    for y_name in features[i + 1:]:
        ax = next(axes)
        ax.scatter(
            training[x_name],
            training[y_name],
            c="#8c959f",
            marker=".",
            s=8,
            alpha=0.16,
            linewidths=0,
            rasterized=True,
            label="Existing sweep",
        )
        for label, frame, feasible_mask, marker, size in plot_sources:
            for pass_state, color in ((True, "#238636"), (False, "#cf222e")):
                mask = feasible_mask == pass_state
                if mask.any():
                    ax.scatter(
                        frame.loc[mask, x_name],
                        frame.loc[mask, y_name],
                        c=color,
                        marker=marker,
                        s=size,
                        alpha=0.95,
                        edgecolors="white",
                        linewidths=0.35,
                        label=f"{label}: {'feasible' if pass_state else 'infeasible'}",
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.grid(True, which="major", alpha=0.25, linewidth=0.7)
        ax.tick_params(labelsize=8)
        pair_index += 1
for ax in list(axes)[pair_index:]:
    ax.axis("off")
legend_handles = [
    Line2D([], [], marker=".", color="#8c959f", linestyle="None", markersize=8, label="Existing sweep"),
    Line2D([], [], marker="o", color="#57606a", linestyle="None", markersize=7, label="Saved-surrogate search"),
    Line2D([], [], marker="^", color="#57606a", linestyle="None", markersize=7, label="GP Bayesian optimization"),
    Line2D([], [], marker="o", color="#238636", linestyle="None", markersize=7, label="Ngspice feasible"),
    Line2D([], [], marker="o", color="#cf222e", linestyle="None", markersize=7, label="Ngspice infeasible"),
]
fig.legend(legend_handles, [handle.get_label() for handle in legend_handles],
           loc="lower center", ncol=3, fontsize=9, frameon=False)
fig.suptitle(
    f"{args.scenario}: ngspice-verified design feasibility "
    f"(seed {args.seed}, {args.iterations} evaluations/method)",
    y=0.997,
)
fig.tight_layout(rect=(0, 0.035, 1, 0.985), h_pad=1.6, w_pad=1.0)
fig.savefig(plot_path, dpi=160)
plt.close(fig)

convergence_path = args.output_dir / f"optimization_{scenario_stem}_{run_tag}_convergence.png"
fig, ax = plt.subplots(figsize=(9, 5.5))
initial_objective = (
    float(initial_best[objective_name]) if initial_best is not None else None
)
for method, label, color, marker in (
    ("saved_surrogate_search", "Saved-surrogate search", "#8250df", "o"),
    ("gp_bayesian_optimization", "GP Bayesian optimization", "#0969da", "^"),
):
    method_rows = (
        results[results.method == method]
        .dropna(subset=list(TARGETS))
        .sort_values("iteration")
    )
    best_so_far = initial_objective
    x_values, y_values = [], []
    for row in method_rows.to_dict("records"):
        if target_satisfies(row):
            candidate_value = float(row[objective_name])
            if best_so_far is None:
                best_so_far = candidate_value
            elif direction == "minimize":
                best_so_far = min(best_so_far, candidate_value)
            else:
                best_so_far = max(best_so_far, candidate_value)
        if best_so_far is not None:
            x_values.append(int(row["iteration"]))
            y_values.append(best_so_far * objective_scale)
    if x_values:
        ax.step(x_values, y_values, where="post", color=color, linewidth=2, label=label)
        ax.scatter(
            x_values[::max(1, len(x_values) // 12)],
            y_values[::max(1, len(y_values) // 12)],
            color=color,
            marker=marker,
            s=30,
            zorder=3,
        )
if initial_objective is not None:
    ax.axhline(
        initial_objective * objective_scale,
        color="#57606a",
        linestyle="--",
        linewidth=1.2,
        label="Best shared initial design",
    )
ax.set_xlabel("ngspice evaluation number")
ax.set_ylabel(f"Best feasible objective ({objective_unit})")
ax.set_title("Best ngspice-verified feasible objective as evaluations progress")
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(convergence_path, dpi=180)
plt.close(fig)

summary["plots"] = {
    "design_space": plot_path.name,
    "convergence": convergence_path.name,
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

html_report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Optimization benchmark: {html.escape(args.scenario)}</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;display:block;overflow-x:auto;margin:1rem 0 2rem}}
th,td{{border:1px solid #d0d7de;padding:.4rem .6rem;white-space:nowrap}}th{{background:#f6f8fa}}
img{{max-width:100%;height:auto}}</style></head>
<body><h1>Optimization benchmark: {html.escape(args.scenario)}</h1>
<p>{html.escape(str(summary["description"]))}</p>
<p>The saved random-forest model and GP Bayesian optimization each receive
{args.iterations} new ngspice evaluations and share the same {args.initial_size}
initial sweep observations. Every proposed candidate is checked with ngspice;
the saved model is not retrained. The GP updates its own posterior after each
ngspice result.</p>
{f"<p>Each candidate is checked at {topology.COMMON_MODE_POINTS} input common-mode points from {topology.COMMON_MODE_RANGE_V[0]:g} to {topology.COMMON_MODE_RANGE_V[1]:g} V; reported performance and margins are worst-case across that range.</p>" if hasattr(topology, "COMMON_MODE_RANGE_V") else ""}
<h2>Best feasible designs</h2>{comparison_display.to_html(index=False, border=0, escape=True)}
<h2>Computational time by methodology</h2>
<p>Optimization computation excludes ngspice simulation time; total method
time adds that evaluation time. It excludes one-time model/data loading and
report generation.</p>
{timing_display.to_html(index=False, border=0, float_format=lambda value: f"{value:.3f}")}
<p><strong>Saved-surrogate wall-time savings:</strong> {time_savings_factor:.2f}x faster
({summary["saved_surrogate_wall_time_reduction_percent"]:.1f}% reduction versus GP)
for this ngspice-verified search.</p>
<h2>Passing and failing design spaces</h2>
<p>Green points meet every scenario constraint in ngspice; red points fail at
least one. Faint gray points show the existing sweep; colored symbols show
the new evaluations. Circle and triangle markers distinguish the saved
surrogate and GP searches.</p>
<img src="{html.escape(plot_path.name)}" alt="Pairwise transistor design-space projections colored by scenario feasibility">
<h2>Optimization progress</h2>
<p>Lines show the best feasible objective seen so far, based on ngspice.
Lower is better for minimization and higher is better for maximization.</p>
<img src="{html.escape(convergence_path.name)}" alt="Best feasible objective versus evaluation number">
<p>Per-evaluation data: <code>{html.escape(trials_path.name)}</code>.
Seed: {args.seed}. Different seeds change the sampled initial sweep subset,
candidate sequences, and GP sampling; comparing them shows how sensitive the
results are to randomness. A few seeds are useful evidence, not a guarantee.</p>
</body></html>"""
html_path.write_text(html_report, encoding="utf-8")
print(f"Scenario: {args.scenario} ({scenario.get('description', '')})")
print(f"Best feasible saved-surrogate result: {report_row(rf_best)}")
print(f"Best feasible GP result: {report_row(gp_best)}")
print(f"Trial data: {trials_path}")
print(f"Summary: {summary_path}")
print(f"Timing: {json.dumps(timing, sort_keys=True)}")
print(f"Design-space plot: {plot_path}")
print(f"Convergence plot: {convergence_path}")
print(f"HTML report: {html_path}")
