from pathlib import Path

import numpy as np
import yaml

from src.metrics import ac_metrics, parse_op
from src.ngspice_backend import MODELS, simulate
from src.specs import SAT_MARGIN, V_TAIL_MIN

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(__file__).with_name("complementary_input.yaml")
with CONFIG_FILE.open(encoding="utf-8") as file:
    CONFIG = yaml.safe_load(file)

VDD = CONFIG["supply_v"]
TRAIN_POWER = True
SURROGATE_TARGETS = (
    "gain_db", "gbw_hz", "pm_deg", "sat_margin", "v_tail", "power_w"
)
CM_MIN = CONFIG["input_common_mode"]["min_v"]
CM_MAX = CONFIG["input_common_mode"]["max_v"]
CM_STEP = CONFIG["input_common_mode"]["step_v"]
COMMON_MODE_RANGE_V = (CM_MIN, CM_MAX)
_CM_VALUES = np.linspace(
    CM_MIN,
    CM_MAX,
    round((CM_MAX - CM_MIN) / CM_STEP) + 1,
)
COMMON_MODE_POINTS = len(_CM_VALUES)
OUTPUT_BIAS = CONFIG["output_bias_v"]
FOLD_W = CONFIG["folding_devices"]["width_um"]
FOLD_L = CONFIG["folding_devices"]["length_um"]
VB1 = CONFIG["cascode_biases"]["vb1_v"]
VB2 = CONFIG["cascode_biases"]["vb2_v"]
VB3 = CONFIG["cascode_biases"]["vb3_v"]
M5_M3_CURRENT_RATIO = CONFIG["cascode_biases"]["m5_to_m3_current_ratio"]
M5_M3_RATIO_TOLERANCE = CONFIG["cascode_biases"][
    "current_ratio_tolerance_fraction"
]
M5_M3_MIN_CURRENT_RATIO = 1.20
RANGES = {
    name: (values["min"], values["max"])
    for name, values in CONFIG["parameters"].items()
}
DATA_FILE = ROOT / "data/complementary_input_sobol.csv"
MODEL_FILE = ROOT / "data/complementary_input_surrogate.joblib"
SCENARIO_FILE = ROOT / "optimization_scenarios_complementary.yaml"

_DEVICE_TYPES = {
    "m1": "pfet", "m2": "pfet", "m3": "nfet", "m4": "nfet",
    "m5": "pfet", "m6": "pfet", "m7": "pfet", "m8": "pfet",
    "m9": "nfet", "m10": "nfet", "m11": "nfet", "m12": "nfet",
}


def _device_path(index, device, parameter="vdsat"):
    kind = _DEVICE_TYPES[device]
    return f"@m.x{device}_{index}.msky130_fd_pr__{kind}_01v8[{parameter}]"


def _instance_netlist(
    index, cm, itail, cl, clamp_output, output_current=0.0,
    fold_w=FOLD_W, fold_l=FOLD_L, vb1=VB1, vb2=VB2, vb3=VB3,
):
    p = lambda name: f"{name}{index}"
    output_bias_fixture = f"Ioutbias{index} {p('out')} 0 dc 0\n"
    return f"""Vdd{index} {p('vdd')} 0 {VDD}
Ib1{index} {p('vdd')} {p('psrc')} dc {itail}
Ib2{index} {p('nsrc')} 0 dc {itail}
{output_bias_fixture}\
Vb1{index} {p('vb1')} 0 {vb1}
Vb2{index} {p('vb2')} 0 {vb2}
Vb3{index} {p('vb3')} 0 {vb3}
Vinp{index} {p('inp')} 0 dc {cm:.12g} ac 1
Vinn{index} {p('inn')} 0 dc {cm:.12g}
CL{index} {p('out')} 0 {cl}
XM1_{index} {p('lower1')} {p('inp')} {p('psrc')} {p('vdd')} sky130_fd_pr__pfet_01v8 W=$wp L=$lp
XM2_{index} {p('lower2')} {p('inn')} {p('psrc')} {p('vdd')} sky130_fd_pr__pfet_01v8 W=$wp L=$lp
XM3_{index} {p('pmid1')} {p('inp')} {p('nsrc')} 0 sky130_fd_pr__nfet_01v8 W=$wn L=$ln
XM4_{index} {p('pmid2')} {p('inn')} {p('nsrc')} 0 sky130_fd_pr__nfet_01v8 W=$wn L=$ln
* KCL: I5 = I3 + I7 at pmid1; I11 = I9 + I1 at lower1.
* The right branch mirrors these summing nodes around the output.
XM5_{index} {p('pmid1')} {p('upper1')} {p('vdd')} {p('vdd')} sky130_fd_pr__pfet_01v8 W={fold_w} L={fold_l}
XM6_{index} {p('pmid2')} {p('upper1')} {p('vdd')} {p('vdd')} sky130_fd_pr__pfet_01v8 W={fold_w} L={fold_l}
* Isolated PMOS wells are source-tied to avoid unnecessary body effect.
XM7_{index} {p('upper1')} {p('vb1')} {p('pmid1')} {p('pmid1')} sky130_fd_pr__pfet_01v8 W={fold_w} L={fold_l}
XM8_{index} {p('out')} {p('vb1')} {p('pmid2')} {p('pmid2')} sky130_fd_pr__pfet_01v8 W={fold_w} L={fold_l}
XM9_{index} {p('upper1')} {p('vb2')} {p('lower1')} 0 sky130_fd_pr__nfet_01v8 W={fold_w} L={fold_l}
XM10_{index} {p('out')} {p('vb2')} {p('lower2')} 0 sky130_fd_pr__nfet_01v8 W={fold_w} L={fold_l}
XM11_{index} {p('lower1')} {p('vb3')} 0 0 sky130_fd_pr__nfet_01v8 W={fold_w} L={fold_l}
XM12_{index} {p('lower2')} {p('vb3')} 0 0 sky130_fd_pr__nfet_01v8 W={fold_w} L={fold_l}
"""


def _operating_point_commands(index, clamp_output):
    nodes = (
        "nsrc", "psrc", "pmid1", "pmid2", "upper1", "lower1", "lower2",
        "out",
    )
    commands = [
        "print " + " ".join(f"v({node}{index})" for node in nodes),
        f"print i(vdd{index})",
        f"print i(vb1{index})",
        f"print i(vb2{index})",
        f"print i(vb3{index})",
    ]
    if clamp_output:
        commands.append(f"print i(ioutbias{index})")
    commands.extend(
        f"print {_device_path(index, device)}"
        for device in _DEVICE_TYPES
    )
    commands.extend(
        f"print {_device_path(index, device, 'vth')}"
        for device in _DEVICE_TYPES
    )
    commands.extend(
        f"print {_device_path(index, device, 'id')}"
        for device in _DEVICE_TYPES
    )
    return "\n".join(commands)


def _build_netlist(
    wn, ln, wp, lp, itail, cl, clamp_output, output_currents=None,
    fold_w=FOLD_W, fold_l=FOLD_L, vb1=VB1, vb2=VB2, vb3=VB3,
):
    output_currents = output_currents or [0.0] * len(_CM_VALUES)
    instances = "\n".join(
        _instance_netlist(
            index, cm, itail, cl, clamp_output, output_currents[index],
            fold_w, fold_l, vb1, vb2, vb3,
        )
        for index, cm in enumerate(_CM_VALUES)
    )
    outputs = [f"ac_{index}.txt" for index in range(len(_CM_VALUES))]
    control = [
        "set noaskquit",
        "op",
        *(
            _operating_point_commands(index, clamp_output)
            for index in range(len(_CM_VALUES))
        ),
    ]
    if not clamp_output:
        control.extend([
            "ac dec 20 1 10G",
            "run",
            *(f"wrdata {outfile} v(out{index})" for index, outfile in enumerate(outputs)),
        ])
    control.append("quit")
    netlist = f"""* Complementary folded-cascode OTA, common-mode characterization
{MODELS}
.option savecurrents
{instances}
.control
{"".join(command + chr(10) for command in control)}
.endc
.end
"""
    netlist = netlist.replace("$wn", str(wn)).replace("$ln", str(ln))
    netlist = netlist.replace("$wp", str(wp)).replace("$lp", str(lp))
    return netlist, outputs


def evaluate(
    wn, ln, wp, lp, itail=20e-6, cl=1e-12,
    fold_w=FOLD_W, fold_l=FOLD_L, vb1=VB1, vb2=VB2, vb3=VB3,
):
    """Evaluate worst-case metrics at every configured common-mode point.

    A pair counts as active above 1% of its nominal tail current. Tail
    headroom is the minimum across all active pairs. Every active MOSFET in
    the input and summing stages must satisfy the saturation check.
    """
    bias_netlist, _ = _build_netlist(
        wn, ln, wp, lp, itail, cl, clamp_output=True,
        fold_w=fold_w, fold_l=fold_l, vb1=vb1, vb2=vb2, vb3=vb3,
    )
    _, _ = simulate(bias_netlist, [])
    netlist, outputs = _build_netlist(
        wn, ln, wp, lp, itail, cl, clamp_output=False,
        fold_w=fold_w, fold_l=fold_l, vb1=vb1, vb2=vb2, vb3=vb3,
    )
    data_by_cm, stdout = simulate(netlist, outputs)
    op = parse_op(stdout)

    gains, gbws, phase_margins, mirror_current_ratios = [], [], [], []
    saturation_margins, tail_headroom_by_cm, powers = [], [], []
    output_voltages = []
    operating_details = {}
    cascode_branches_ok = True
    mirror_current_ratio_ok = True
    active_current_min = max(itail * 0.01, 1e-11)
    for index, outfile in enumerate(outputs):
        data = data_by_cm[outfile]
        if data.ndim != 2 or data.shape[1] != 3:
            raise RuntimeError(
                f"expected 3 AC columns for {outfile}, got shape {data.shape}"
            )
        gain, gbw, phase_margin = ac_metrics(
            data[:, 0], data[:, 1] + 1j * data[:, 2]
        )
        gains.append(gain)
        gbws.append(gbw if np.isfinite(gbw) else 0.0)
        phase_margins.append(phase_margin if np.isfinite(phase_margin) else 0.0)

        node = lambda name: op[f"v({name}{index})"]
        output_voltages.append(node("out"))
        operating_details.update({
            f"v_{name}": node(name)
            for name in ("nsrc", "psrc", "pmid1", "pmid2", "upper1",
                         "lower1", "lower2", "out")
        })
        operating_details.update({
            "i_supply": op[f"i(vdd{index})"],
            "i_vb1": op[f"i(vb1{index})"],
            "i_vb2": op[f"i(vb2{index})"],
            "i_vb3": op[f"i(vb3{index})"],
            "i_output": 0.0,
        })
        vds = {
            "m1": node("psrc") - node("lower1"),
            "m2": node("psrc") - node("lower2"),
            "m3": node("pmid1") - node("nsrc"),
            "m4": node("pmid2") - node("nsrc"),
            "m5": VDD - node("pmid1"),
            "m6": VDD - node("pmid2"),
            "m7": node("pmid1") - node("upper1"),
            "m8": node("pmid2") - node("out"),
            "m9": node("lower1") - node("upper1"),
            "m10": node("lower2") - node("out"),
            "m11": node("lower1"),
            "m12": node("lower2"),
        }
        terminals = {
            "m1": ("inp", "psrc", "lower1", "vdd"),
            "m2": ("inn", "psrc", "lower2", "vdd"),
            "m3": ("inp", "nsrc", "pmid1", "0"),
            "m4": ("inn", "nsrc", "pmid2", "0"),
            "m5": ("upper1", "vdd", "pmid1", "vdd"),
            "m6": ("upper1", "vdd", "pmid2", "vdd"),
            "m7": ("vb1", "pmid1", "upper1", "pmid1"),
            "m8": ("vb1", "pmid2", "out", "pmid2"),
            "m9": ("vb2", "lower1", "upper1", "0"),
            "m10": ("vb2", "lower2", "out", "0"),
            "m11": ("vb3", "0", "lower1", "0"),
            "m12": ("vb3", "0", "lower2", "0"),
        }
        for device in _DEVICE_TYPES:
            gate, source, drain, bulk = terminals[device]
            if gate == "vb1":
                gate_v = vb1
            elif gate == "vb2":
                gate_v = vb2
            elif gate == "vb3":
                gate_v = vb3
            elif gate == "inp":
                gate_v = _CM_VALUES[index]
            elif gate == "inn":
                gate_v = _CM_VALUES[index]
            else:
                gate_v = node(gate)
            source_v = VDD if source == "vdd" else 0.0 if source == "0" else node(source)
            drain_v = VDD if drain == "vdd" else 0.0 if drain == "0" else node(drain)
            current = op[_device_path(index, device, "id").lower()]
            vdsat = op[_device_path(index, device, "vdsat").lower()]
            vth = op[_device_path(index, device, "vth").lower()]
            operating_details.update({
                f"{device}_id_a": current,
                f"{device}_vgs_v": gate_v - source_v,
                f"{device}_vds_v": drain_v - source_v,
                f"{device}_vdsat_v": vdsat,
                f"{device}_vth_v": vth,
                f"{device}_sat_margin_v": abs(drain_v - source_v) - abs(vdsat),
            })
        operating_details.update({
            "i_tail_p_measured": sum(
                abs(op[_device_path(index, device, "id").lower()])
                for device in ("m1", "m2")
            ),
            "i_tail_n_measured": sum(
                abs(op[_device_path(index, device, "id").lower()])
                for device in ("m3", "m4")
            ),
        })
        active = []
        for device in _DEVICE_TYPES:
            current = op[_device_path(index, device, "id").lower()]
            if abs(current) > active_current_min:
                active.append(device)
        i3 = abs(op[_device_path(index, "m3", "id").lower()])
        i5 = abs(op[_device_path(index, "m5", "id").lower()])
        if i3 > active_current_min:
            mirror_ratio = i5 / i3
            mirror_current_ratios.append(mirror_ratio)
            mirror_current_ratio_ok &= mirror_ratio > M5_M3_MIN_CURRENT_RATIO
        else:
            mirror_current_ratio_ok = False
        active_set = set(active)
        if (
            active_set.intersection(("m1", "m2"))
            and active_set.intersection(("m3", "m4"))
        ):
            # At mixed common mode, all four cascode devices must conduct.
            cascode_branches_ok &= all(
                device in active_set for device in ("m7", "m8", "m9", "m10")
            )
        cm_saturation_margins = []
        for device in active:
            path = _device_path(index, device).lower()
            cm_saturation_margins.append(
                abs(vds[device]) - abs(op[path])
            )

        n_pair_current = sum(
            abs(op[_device_path(index, device, "id").lower()])
            for device in ("m1", "m2")
        )
        p_pair_current = sum(
            abs(op[_device_path(index, device, "id").lower()])
            for device in ("m3", "m4")
        )
        available_tail_headrooms = []
        if n_pair_current > active_current_min:
            available_tail_headrooms.append(node("nsrc"))
        if p_pair_current > active_current_min:
            available_tail_headrooms.append(VDD - node("psrc"))
        if available_tail_headrooms:
            tail_headroom_by_cm.append(
                min(available_tail_headrooms)
            )
        else:
            tail_headroom_by_cm.append(float("nan"))
            cm_saturation_margins.append(float("nan"))
        saturation_margins.extend(cm_saturation_margins)
        powers.append(VDD * abs(op[f"i(vdd{index})"]))

    def worst_case(values):
        array = np.asarray(values, dtype=float)
        return (
            float(np.min(array))
            if array.size and np.all(np.isfinite(array))
            else float("nan")
        )

    gain_db = worst_case(gains)
    gbw_hz = worst_case(gbws)
    pm_deg = worst_case(phase_margins)
    sat_margin = worst_case(saturation_margins)
    v_tail = worst_case(tail_headroom_by_cm)
    power_w = float(np.max(powers))
    return {
        "gain_db": gain_db,
        "gbw_hz": gbw_hz,
        "pm_deg": pm_deg,
        "power_w": power_w,
        "sat_margin": sat_margin,
        "v_tail": v_tail,
        **operating_details,
        "vout": worst_case(output_voltages),
        "valid": bool(
            sat_margin > SAT_MARGIN
            and v_tail > V_TAIL_MIN
            and cascode_branches_ok
            and mirror_current_ratio_ok
        ),
        "cascode_branches_ok": float(cascode_branches_ok),
        "cm_points": len(_CM_VALUES),
        "mirror_current_ratio": worst_case(mirror_current_ratios),
        "mirror_ratio_error": abs(
            worst_case(mirror_current_ratios) - M5_M3_CURRENT_RATIO
        ),
    }


def power_estimate(parameters):
    return 4.5 * VDD * parameters[..., list(RANGES).index("itail")]
