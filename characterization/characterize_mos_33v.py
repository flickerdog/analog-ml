#!/usr/bin/env python3
"""Generate SKY130 thick-oxide NMOS/PMOS gm/Id lookup tables."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from characterization.characterize_mos import _derive_metrics, _make_recommendations
from src.ngspice_backend import MODELS, SPICE

OUTPUT = Path(__file__).resolve().parent / "results_3v3"
WIDTHS_UM = (10.0, 20.0, 50.0)
LENGTHS_UM = (0.5, 1.0, 2.0)
VDS_VALUES_V = (0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.75, 1.0, 1.5, 2.0, 2.4)
VGS_MAX_V = 2.4
VGS_STEP_V = 0.01
SOURCE_V = {"n": 0.6, "p": 3.3}
BODY_V = {"n": (0.0, 0.3, 0.6), "p": (2.7, 3.0, 3.3)}
MODELS_33V = f"""{MODELS}
.param hv_dlc_rotweak = .00e-9
.param sky130_fd_pr__nfet_03v3_nvt__dlc_rotweak = hv_dlc_rotweak
.param sky130_fd_pr__pfet_g5v0d10v5__dlc_rotweak = hv_dlc_rotweak
.include {SPICE}/sky130_fd_pr__nfet_03v3_nvt__tt.corner.spice
.include {SPICE}/sky130_fd_pr__nfet_03v3_nvt__mismatch.corner.spice
.include {SPICE}/sky130_fd_pr__pfet_g5v0d10v5__tt.corner.spice
.include {SPICE}/sky130_fd_pr__pfet_g5v0d10v5__mismatch.corner.spice
"""


def _token(value):
    return str(value).replace(".", "p").replace("-", "m")


def _device_setup(polarity, body_v):
    if polarity == "n":
        model = "sky130_fd_pr__nfet_03v3_nvt"
        dc_source, gate_voltage, drain_voltage, drain_current = (
            "Vgs", "v(g,s)", "v(d,s)", "i(Vds)"
        )
        sources = (
            f"Vsrc s 0 {SOURCE_V['n']}",
            f"Vbody b 0 {body_v}",
            "Vgs g s 0",
            "Vds d s 0.6",
        )
        body_bias = SOURCE_V["n"] - body_v
    else:
        model = "sky130_fd_pr__pfet_g5v0d10v5"
        dc_source, gate_voltage, drain_voltage, drain_current = (
            "Vsg", "v(s,g)", "v(s,d)", "i(Vsd)"
        )
        sources = (
            f"Vsrc s 0 {SOURCE_V['p']}",
            f"Vbody b 0 {body_v}",
            "Vsg s g 0",
            "Vsd s d 0.6",
        )
        body_bias = body_v - SOURCE_V["p"]
    return (
        model, dc_source, gate_voltage, drain_voltage, drain_current,
        sources, body_bias,
    )


def _make_netlist(polarity, width, length, output_names):
    model, dc_source, _, _, drain_current, sources, _ = _device_setup(
        polarity, BODY_V[polarity][0]
    )
    control = ["set wr_singlescale"]
    for body_v in BODY_V[polarity]:
        control.append(f"alter Vbody = {body_v}")
        for vds, output_name in zip(VDS_VALUES_V, output_names[body_v]):
            control.extend([
                f"alter {'Vds' if polarity == 'n' else 'Vsd'} = {vds}",
                f"dc {dc_source} 0 {VGS_MAX_V} {VGS_STEP_V}",
                f"wrdata {output_name} {drain_current}",
            ])
    return f"""* SKY130 thick-oxide MOS DC characterization, typical corner
{MODELS_33V}
{chr(10).join(sources)}
Xdut d g s b {model} W={width} L={length}
.control
{chr(10).join(control)}
quit
.endc
.end
"""


def _run_geometry(polarity, width, length, workdir):
    output_names = {
        body_v: [
            f"{polarity}_w{_token(width)}_l{_token(length)}_b{_token(body_v)}"
            f"_vd{_token(vds)}.txt"
            for vds in VDS_VALUES_V
        ]
        for body_v in BODY_V[polarity]
    }
    circuit = workdir / f"{polarity}_w{_token(width)}_l{_token(length)}.cir"
    circuit.write_text(
        _make_netlist(polarity, width, length, output_names), encoding="utf-8"
    )
    result = subprocess.run(
        ["ngspice", "-b", circuit.name],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            f"ngspice failed for {circuit.name}:\n{result.stdout}\n{result.stderr}"
        )

    model, _, _, _, _, _, _ = _device_setup(polarity, BODY_V[polarity][0])
    records = []
    for body_v, names in output_names.items():
        _, _, _, _, _, _, body_bias_v = _device_setup(polarity, body_v)
        for vds, name in zip(VDS_VALUES_V, names):
            output_file = workdir / name
            if not output_file.exists():
                raise RuntimeError(f"ngspice did not create {output_file.name}")
            data = np.loadtxt(output_file, ndmin=2)
            if data.shape[1] != 2:
                raise RuntimeError(
                    f"expected gate voltage and drain-current columns in {name}, "
                    f"got {data.shape[1]}"
                )
            gate_voltage, signed_current = data.T
            records.extend({
                "polarity": model.removeprefix("sky130_fd_pr__"),
                "width_um": width,
                "length_um": length,
                "body_voltage_v": body_v,
                "body_bias_v": body_bias_v,
                "vgs_v": gate_v,
                "vds_v": vds,
                "id_a": abs(current),
            } for gate_v, current in zip(gate_voltage, signed_current))
    return pd.DataFrame.from_records(records)


def _plot_results(table, output_dir):
    plot_width = 20.0 if 20.0 in table["width_um"].values else table["width_um"].min()
    nominal = table[
        np.isclose(table["vds_v"], 0.6)
        & np.isclose(table["body_bias_v"], 0.0)
        & np.isclose(table["width_um"], plot_width)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for polarity, group in nominal.groupby("polarity"):
        for length, length_group in group.groupby("length_um"):
            label = f"{polarity}, W={plot_width:g}, L={length:g} um"
            axes[0].plot(
                length_group["vgs_v"],
                length_group["current_density_a_per_um"] * 1e6,
                label=label,
            )
            axes[1].plot(
                length_group["current_density_a_per_um"] * 1e6,
                length_group["gm_over_id_per_v"],
                label=label,
            )
    axes[0].set(xlabel="VGS / VSG (V)", ylabel="|ID| / W (uA/um)")
    axes[1].set(
        xlabel="|ID| / W (uA/um)",
        ylabel="gm / ID (1/V)",
        xscale="log",
        ylim=(0, 40),
    )
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle(f"SKY130 thick-oxide TT MOSFETs, W={plot_width:g} um, VDS=0.6 V")
    fig.tight_layout()
    fig.savefig(output_dir / "gm_id_overview.png", dpi=180)
    plt.close(fig)

    body = table[
        np.isclose(table["vds_v"], 0.6)
        & np.isclose(table["width_um"], plot_width)
        & np.isclose(table["length_um"], 0.5)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for _, group in body.groupby(
        ["polarity", "width_um", "length_um", "body_voltage_v"]
    ):
        group = group.sort_values("vgs_v")
        polarity = group["polarity"].iloc[0]
        bias = group["body_bias_v"].iloc[0]
        label = f"{polarity}, body bias={bias:g} V"
        axes[0].plot(
            group["vgs_v"], group["current_density_a_per_um"] * 1e6, label=label
        )
        axes[1].plot(
            group["vgs_v"], group["gm_over_id_per_v"], label=label
        )
    axes[0].set(xlabel="VGS / VSG (V)", ylabel="|ID| / W (uA/um)")
    axes[1].set(xlabel="VGS / VSG (V)", ylabel="gm / ID (1/V)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"Body-effect sensitivity, W={plot_width:g} um, L=0.5 um, VDS=0.6 V"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "body_effect.png", dpi=180)
    plt.close(fig)


def characterize(output_dir=OUTPUT, quick=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = (10.0,) if quick else WIDTHS_UM
    lengths = (0.5,) if quick else LENGTHS_UM
    frames = []
    with tempfile.TemporaryDirectory(prefix="sky130_33v_mos_") as temporary:
        workdir = Path(temporary)
        geometries = [
            (polarity, width, length)
            for polarity in ("n", "p")
            for width in widths
            for length in lengths
        ]
        for index, (polarity, width, length) in enumerate(geometries, start=1):
            print(
                f"[{index}/{len(geometries)}] "
                f"{polarity.upper()}MOS W={width:g} um L={length:g} um",
                flush=True,
            )
            frames.append(_run_geometry(polarity, width, length, workdir))
    table = _derive_metrics(pd.concat(frames, ignore_index=True))
    table.to_csv(output_dir / "mosfet_lut.csv", index=False)
    _make_recommendations(table).to_csv(
        output_dir / "gm_id_bias_recommendations.csv", index=False
    )
    _plot_results(table, output_dir)
    print(f"Wrote {len(table):,} LUT rows to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    characterize(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
