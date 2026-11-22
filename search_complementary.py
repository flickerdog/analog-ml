"""Sobol-initialized, ngspice-verified Bayesian gain search at CM=0.9 V.

Run from the repository root:
    .venv/bin/python search_complementary.py --sobol-m 5 --bo-iterations 20
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from scipy.spatial import cKDTree
from scipy.stats import qmc

from src.complementary_ota import evaluate
from src.specs import LOAD_CAP

ROOT = Path(__file__).resolve().parent
RANGES = {
    "nmos_w": (45.0, 50.0),
    "pmos_w": (80.0, 100.0),
    "input_l": (0.17, 0.20),
    "itail": (18e-6, 22e-6),
    "fold_w": (17.0, 19.0),
    "vb1": (0.39, 0.42),
}
LIMITS = {
    "pm_deg": ("min", 60.0),
    "sat_margin": ("min", 0.05),
    "v_tail": ("min", 0.15),
    "mirror_ratio_error": ("max", 0.015),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--sobol-m", type=int, default=5, help="initial Sobol count is 2**m")
parser.add_argument("--bo-iterations", type=int, default=20)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "data/complementary_center_local_trials.csv",
)
args = parser.parse_args()
if args.sobol_m < 3:
    parser.error("--sobol-m must be at least 3")
if args.bo_iterations < 0:
    parser.error("--bo-iterations must be nonnegative")

FEATURES = list(RANGES)
LOWER_LOG = np.log10([RANGES[name][0] for name in FEATURES])
UPPER_LOG = np.log10([RANGES[name][1] for name in FEATURES])
TARGETS = [
    "gain_db", "pm_deg", "sat_margin", "v_tail",
    "mirror_ratio_error",
]
torch.set_num_threads(1)
torch.set_default_dtype(torch.double)


def decode(unit_x):
    log_x = LOWER_LOG + np.asarray(unit_x) * (UPPER_LOG - LOWER_LOG)
    return 10**log_x


def evaluate_unit(unit_x):
    params = dict(zip(FEATURES, decode(unit_x)))
    result = evaluate(
        params["nmos_w"], params["input_l"],
        params["pmos_w"], params["input_l"],
        params["itail"], LOAD_CAP,
        fold_w=params["fold_w"], fold_l=0.15,
        vb1=params["vb1"], vb2=1.05, vb3=0.66,
    )
    return {**params, **result}


def is_feasible(row):
    return (
        row["pm_deg"] >= 60
        and row["sat_margin"] > 0.05
        and row["v_tail"] > 0.15
        and row["mirror_ratio_error"] <= 0.015
        and bool(row["valid"])
    )


def fit_model(x, y):
    tx = torch.as_tensor(x, dtype=torch.double)
    ty = torch.as_tensor(y, dtype=torch.double)
    models = [
        SingleTaskGP(tx, ty[:, i:i + 1], outcome_transform=Standardize(m=1))
        for i in range(ty.shape[1])
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def acquire_candidate(x, y, seed):
    model = fit_model(x, y)
    constraints = []
    for metric, (kind, threshold) in LIMITS.items():
        index = TARGETS.index(metric)
        if kind == "min":
            constraints.append(
                lambda samples, idx=index, bound=threshold: bound - samples[..., idx]
            )
        else:
            constraints.append(
                lambda samples, idx=index, bound=threshold: samples[..., idx] - bound
            )
    objective_index = TARGETS.index("gain_db")
    acquisition = qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=torch.as_tensor(x, dtype=torch.double),
        sampler=SobolQMCNormalSampler(
            sample_shape=torch.Size([128]), seed=seed
        ),
        objective=GenericMCObjective(
            lambda samples, X=None: samples[..., objective_index]
        ),
        constraints=constraints,
        prune_baseline=True,
    )
    pool = qmc.Sobol(d=len(FEATURES), scramble=True, seed=seed).random(2048)
    distances, _ = cKDTree(x).query(pool, k=1)
    pool = pool[distances > 1e-5]
    if not len(pool):
        raise RuntimeError("Bayesian candidate pool contains no unseen points")
    scores = []
    for offset in range(0, len(pool), 256):
        batch = torch.as_tensor(pool[offset:offset + 256], dtype=torch.double)
        with torch.no_grad():
            score = acquisition(batch.unsqueeze(-2))
        scores.extend(score.detach().cpu().numpy().reshape(-1))
    return pool[int(np.argmax(scores))]


def save_results(rows):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


def run():
    sampler = qmc.Sobol(d=len(FEATURES), scramble=True, seed=args.seed)
    initial = sampler.random_base2(m=args.sobol_m)
    rows = []
    units = []
    started = time.perf_counter()
    known_feasible = {
        "nmos_w": 50.0,
        "pmos_w": 100.0,
        "input_l": 0.19,
        "itail": 20e-6,
        "fold_w": 18.0,
        "vb1": 0.40,
    }
    known_unit = (
        np.log10([known_feasible[name] for name in FEATURES]) - LOWER_LOG
    ) / (UPPER_LOG - LOWER_LOG)
    seed_row = evaluate_unit(known_unit)
    seed_row.update(phase="known_feasible_seed", iteration=0)
    rows.append(seed_row)
    units.append(known_unit)
    save_results(rows)
    if not is_feasible(seed_row):
        raise RuntimeError(
            "the known center-point design no longer satisfies search constraints; "
            "recheck the evaluator and configuration before optimizing"
        )
    print(
        f"Seeded feasible design: {seed_row['gain_db']:.2f} dB; "
        f"sat_margin={seed_row['sat_margin']:.3f} V, "
        f"tail_headroom={seed_row['v_tail']:.3f} V",
        flush=True,
    )
    for index, unit_x in enumerate(initial, start=1):
        row = evaluate_unit(unit_x)
        row.update(phase="sobol", iteration=index)
        rows.append(row)
        units.append(unit_x)
        save_results(rows)
        if index % 8 == 0 or index == len(initial):
            print(
                f"Sobol {index}/{len(initial)}; "
                f"feasible={sum(is_feasible(item) for item in rows)}; "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    bo_rows = 0
    for iteration in range(1, args.bo_iterations + 1):
        successful = [row for row in rows if all(np.isfinite(row[name]) for name in TARGETS)]
        x = np.asarray(units)[[rows.index(row) for row in successful]]
        y = np.asarray([[row[name] for name in TARGETS] for row in successful])
        unit_x = acquire_candidate(x, y, args.seed + iteration)
        row = evaluate_unit(unit_x)
        row.update(phase="bayesian_optimization", iteration=iteration)
        rows.append(row)
        units.append(unit_x)
        save_results(rows)
        bo_rows += 1
        if iteration % 5 == 0 or iteration == args.bo_iterations:
            best = max(
                (item["gain_db"] for item in rows if is_feasible(item)),
                default=float("nan"),
            )
            print(
                f"BO {iteration}/{args.bo_iterations}; "
                f"feasible={sum(is_feasible(item) for item in rows)}; "
                f"best_feasible_gain={best:.2f} dB; "
                f"elapsed={time.perf_counter() - started:.0f}s",
                flush=True,
            )

    feasible = [row for row in rows if is_feasible(row)]
    best = max(feasible, key=lambda row: row["gain_db"]) if feasible else None
    summary = {
        "common_mode_v": 0.9,
        "output_bias_v": 0.9,
        "seed": args.seed,
        "sobol_points": len(initial),
        "bayesian_optimization_points": bo_rows,
        "total_ngspice_points": len(rows),
        "feasible_points": len(feasible),
        "best_feasible": best,
        "best_any": max(rows, key=lambda row: row["gain_db"]),
        "elapsed_seconds": time.perf_counter() - started,
        "trials_csv": str(args.output),
        "search_ranges": RANGES,
        "folding_device_length_um": 0.15,
        "fixed_cascode_biases_v": {"vb2": 1.05, "vb3": 0.66},
        "feasibility_limits": LIMITS,
    }
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(f"Results: {args.output}\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    run()
