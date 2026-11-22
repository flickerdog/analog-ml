"""Run the whole pipeline: simulate -> plots -> table.

    python run_all.py              # re-plot and rebuild the table from the existing data (fast)
    python run_all.py --sweep      # re-simulate first (about 8 min), then plot and report
    python run_all.py --surrogate  # train and evaluate the surrogate
    python run_all.py --docs       # also copy the figures to docs/ (for the README)

If data/ota5t_sobol.csv does not exist yet, the sweep runs automatically.
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

from src.topology import load_topology, topology_names

ROOT = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("--sweep", action="store_true", help="re-run the simulations first")
parser.add_argument("--surrogate", action="store_true", help="also train the surrogate")
parser.add_argument("--docs", action="store_true", help="copy the figures to docs/")
parser.add_argument("--topology", choices=topology_names(), default="ota5t")

if args.surrogate and not (ROOT / "train_surrogate.py").exists():
    parser.error("cannot tra
    args = parser.parse_args()
topology = load_topology(args.topology)
DATA = topology.DATA_FILEin surrogate: train_surrogate.py was not found")

if not DATA.exists() and not args.sweep:
    print(f"{DATA.relative_to(ROOT)} not found: running the sweep first.")
    args.sweep = True

steps = []
if args.sweep:
    steps.append("sweep_ota.py")
if args.topology == "ota5t":
    steps += ["plot_results.py", "report_sweep.py"]
if args.surrogate:
    steps.append("train_surrogate.py")

for step in steps:
    print(f"\n=== {step} ===", flush=True)
    t0 = time.perf_counter()
    # sys.executable = the Python running this script, so the venv is used everywhere
    command = [sys.executable, step]
    if step in {"sweep_ota.py", "train_surrogate.py"}:
        command.extend(["--topology", args.topology])
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"({time.perf_counter() - t0:.0f} s)")

if args.docs:
    (ROOT / "docs").mkdir(exist_ok=True)
    for png in (ROOT / "reports").glob("*.png"):
        shutil.copy(png, ROOT / "docs" / png.name)
        print("copied", png.name, "to docs/")

print("\nDone. Figures and table are in reports/.")
