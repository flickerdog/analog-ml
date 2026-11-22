from pathlib import Path
from string import Template

import numpy as np

from src.metrics import ac_metrics, parse_op
from src.ngspice_backend import MODELS, simulate

VDD = 1.8
MARGIN = 0.05     # V of headroom above Vdsat
TAIL_MIN = 0.15   # V a real tail current source would need
TRAIN_POWER = False
SURROGATE_TARGETS = ("gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail")
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data/ota5t_sobol.csv"
MODEL_FILE = ROOT / "data/ota5t_surrogate.joblib"
SCENARIO_FILE = ROOT / "optimization_scenarios.yaml"
RANGES = {
    "w1": (7, 50),
    "l1": (0.15, 2.0),
    "w3": (7, 50),
    "l3": (0.15, 2.0),
    "itail": (100e-9, 100e-6),
}

DEVICES = {"m1": "nfet", "m2": "nfet", "m3": "pfet", "m4": "pfet"}


def vdsat_vec(dev):
    return f"@m.x{dev}.msky130_fd_pr__{DEVICES[dev]}_01v8[vdsat]"


OP_CMDS = (
    "save all " + " ".join(vdsat_vec(d) for d in DEVICES) + "\n"
    + "op\n"
    + "print v(n1) v(out) v(tail)\n"
    + "\n".join(f"print {vdsat_vec(d)}" for d in DEVICES)
)

NETLIST = Template("""* 5T OTA, open-loop AC through a DC feedback trick
$models

Vdd vdd 0 $vdd
Vin inp 0 dc 0.9 ac 1
Itail tail 0 dc $itail

XM1 n1  inp tail 0   sky130_fd_pr__nfet_01v8 W=$w1 L=$l1
XM2 out inn tail 0   sky130_fd_pr__nfet_01v8 W=$w1 L=$l1
XM3 n1  n1  vdd  vdd sky130_fd_pr__pfet_01v8 W=$w3 L=$l3
XM4 out n1  vdd  vdd sky130_fd_pr__pfet_01v8 W=$w3 L=$l3

* DC unity feedback, AC-grounded inverting input
Lfb out inn 1G
Cfb inn 0 1G
CL  out 0 $cl

.ac dec 20 1 10G

.control
$opcmds
run
wrdata ac.txt v(out)
quit
.endc
.end
""")


def op_checks(op):
    """Smallest (|Vds| - |Vdsat|) over the four devices, and the tail voltage."""
    vn1, vout, vtail = op["v(n1)"], op["v(out)"], op["v(tail)"]
    vds = {"m1": vn1 - vtail, "m2": vout - vtail, "m3": VDD - vn1, "m4": VDD - vout}
    sat_margin = min(abs(vds[d]) - abs(op[vdsat_vec(d)]) for d in DEVICES)
    return sat_margin, vtail


def evaluate(w1, l1, w3, l3, itail=20e-6, cl=1e-12):
    netlist = NETLIST.substitute(
        models=MODELS, vdd=VDD, w1=w1, l1=l1, w3=w3, l3=l3,
        itail=itail, cl=cl, opcmds=OP_CMDS,
    )
    data, stdout = simulate(netlist, "ac.txt")
    assert data.shape[1] == 3, f"expected 3 columns (f, re, im), got {data.shape}"
    freq = data[:, 0]
    h = data[:, 1] + 1j * data[:, 2]
    gain_db, gbw, pm = ac_metrics(freq, h)
    sat_margin, v_tail = op_checks(parse_op(stdout))
    return {
        "gain_db": gain_db, "gbw_hz": gbw, "pm_deg": pm, "power_w": VDD * itail,
        "sat_margin": sat_margin, "v_tail": v_tail,
        "valid": bool(sat_margin > MARGIN and v_tail > TAIL_MIN),
    }


def power_estimate(parameters):
    return VDD * parameters[..., list(RANGES).index("itail")]
