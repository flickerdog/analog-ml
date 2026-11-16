import os
import subprocess
import tempfile
from pathlib import Path
from string import Template

import numpy as np

TIMEOUT = 10

# --- PDK location: the only machine-specific part ---------------------------
PDK_ROOT = Path.home() / ".ciel/sky130A"
TECH = PDK_ROOT / "libs.tech/ngspice"
SPICE = PDK_ROOT / "libs.ref/sky130_fd_pr/spice"

# Trimmed typical-corner models: only the 1.8 V NMOS and PMOS.
# The full `.lib ... tt` loads ~30 device files and takes ~50 s to parse.
MODELS = f"""\
.option scale=1.0u
.include {TECH}/parameters/lod.spice
.param mc_mm_switch=0
.param mc_pr_switch=0
.param lv_dlc_rotweak = .00e-9
.param sky130_fd_pr__nfet_01v8__dlc_rotweak = lv_dlc_rotweak
.param sky130_fd_pr__pfet_01v8__dlc_rotweak = lv_dlc_rotweak
.include {SPICE}/sky130_fd_pr__nfet_01v8__tt.pm3.spice
.include {SPICE}/sky130_fd_pr__nfet_01v8__mismatch.corner.spice
.include {SPICE}/sky130_fd_pr__pfet_01v8__tt.corner.spice
.include {SPICE}/sky130_fd_pr__pfet_01v8__mismatch.corner.spice
"""

NETLIST = Template("""* NMOS single-device DC sweep
$models

Vds d 0 $vds
Vgs g 0 0
XM1 d g 0 0 sky130_fd_pr__nfet_01v8 W=$w L=$l

.dc Vgs 0 1.8 0.01

.control
run
wrdata id.txt i(Vds)
quit
.endc
.end
""")

def run_dc_sweep(w, l, vds=0.9):
    """Simulate one NMOS and return (vgs, id) as numpy arrays."""
    netlist = NETLIST.substitute(models=MODELS, vds=vds, w=w, l=l)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "circuit.cir").write_text(netlist)
        subprocess.run(
            ["ngspice", "-b", "circuit.cir"],
            cwd=tmp, check=True, capture_output=True, text=True, timeout=120,
        )
        data = np.loadtxt(tmp / "id.txt")
    return data[:, 0], -data[:, 1]


def evaluate(w, l, vgs_target=0.6, vds=0.9):
    """Return the metrics of an NMOS biased at vgs_target."""
    vgs, id_ = run_dc_sweep(w, l, vds)
    gm = np.gradient(id_, vgs)            # gm = dId/dVgs
    i_t = np.interp(vgs_target, vgs, id_)
    gm_t = np.interp(vgs_target, vgs, gm)
    return {"id": i_t, "gm": gm_t, "gm_over_id": gm_t / i_t}

def simulate(netlist, outfile):
    """Run a netlist in a scratch folder; return (numbers in `outfile`, ngspice stdout)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "circuit.cir").write_text(netlist)
        result = subprocess.run(
            ["ngspice", "-b", "circuit.cir"],
            cwd=tmp, capture_output=True, text=True, timeout=10,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
        )
        if result.returncode != 0:
            raise RuntimeError(result.stdout + result.stderr)
        return np.loadtxt(tmp / outfile), result.stdout