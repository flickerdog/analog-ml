import subprocess
import tempfile
from pathlib import Path
from string import Template

import numpy as np

# One place to change if the PDK moves or you switch to another one
MODEL_LIB = Path.home() / ".ciel/sky130A/libs.tech/ngspice/sky130.lib.spice"
CORNER = "tt"

NETLIST = Template("""* NMOS single-device DC sweep
.lib $lib $corner

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
    netlist = NETLIST.substitute(lib=MODEL_LIB, corner=CORNER, vds=vds, w=w, l=l)
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