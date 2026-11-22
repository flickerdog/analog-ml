#!/usr/bin/env python3
"""Generate SKY130 1.8 V MOSFET gm/Id lookup tables with ngspice."""

import argparse
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ngspice_backend import MODELS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent / "results"

WIDTHS_UM = (7.0, 10.0, 20.0, 35.0, 50.0)
LENGTHS_UM = (0.15, 0.20, 0.30, 0.50, 1.0, 2.0)
VDS_VALUES_V = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0, 1.2)
VGS_MAX_V = 1.2
VGS_STEP_V = 0.01
SOURCE_V = {"n": 0.6, "p": 1.2}
BODY_V = {"n": (0.0, 0.3, 0.6), "p": (1.2, 1.5, 1.8)}


def _token(value):
    return str(value).replace(".", "p").replace("-", "m")


def _device_setup(polarity, body_v):
    if polarity == "n":
        model = "sky130_fd_pr__nfet_01v8"
        dc_source = "Vgs"
        gate_voltage = "v(g,s)"
        drain_voltage = "v(d,s)"
        drain_current = "i(Vds)"
        sources = (
            f"Vsrc s 0 {SOURCE_V['n']}",
            f"Vbody b 0 {body_v}",
            "Vgs g s 0",
            "Vds d s 0.6",
        )
        body_bias = SOURCE_V["n"] - body_v
    else:
        model = "sky130_fd_pr__pfet_01v8"
        dc_source = "Vsg"
        gate_voltage = "v(s,g)"
        drain_voltage = "v(s,d)"
        drain_current = "i(Vsd)"
        sources = (
            f"Vsrc s 0 {SOURCE_V['p']}",
            f"Vbody b 0 {body_v}",
            "Vsg s g 0",
            "Vsd s d 0.6",
        )
        body_bias = body_v - SOURCE_V["p"]
    return model, dc_source, gate_voltage, drain_voltage, drain_current, sources, body_bias


def _make_netlist(polarity, width, length, output_names):
    body_values = BODY_V[polarity]
    model, dc_source, gate_voltage, drain_voltage, drain_current, sources, _ = (
        _device_setup(polarity, body_values[0])
    )
    control = ["set wr_singlescale"]
    for body_v in body_values:
        control.append(f"alter Vbody = {body_v}")
        for vds, output_name in zip(VDS_VALUES_V, output_names[body_v]):
            control.extend(
                [
                    f"alter {'Vds' if polarity == 'n' else 'Vsd'} = {vds}",
                    f"dc {dc_source} 0 {VGS_MAX_V} {VGS_STEP_V}",
                    f"wrdata {output_name} {drain_current}",
                ]
            )
    return f"""* SKY130 1.8 V {polarity.upper()}MOS DC characterization
{MODELS}
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
    netlist = _make_netlist(polarity, width, length, output_names)
    circuit = workdir / f"{polarity}_w{_token(width)}_l{_token(length)}.cir"
    circuit.write_text(netlist, encoding="utf-8")
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
                    f"expected VGS and drain-current columns in {name}, "
                    f"got {data.shape[1]}"
                )
            vgs, signed_id = data.T
            records.extend(
                {
                    "polarity": "nmos" if polarity == "n" else "pmos",
                    "width_um": width,
                    "length_um": length,
                    "body_voltage_v": body_v,
                    "body_bias_v": body_bias_v,
                    "vgs_v": gate_v,
                    "vds_v": vds,
                    "id_a": abs(current),
                }
                for gate_v, current in zip(vgs, signed_id)
            )
    return pd.DataFrame.from_records(records)


def _derive_metrics(table):
    keys = ["polarity", "width_um", "length_um", "body_voltage_v", "vds_v"]
    table = table.sort_values(keys + ["vgs_v"]).reset_index(drop=True)
    table["gm_s"] = table.groupby(keys, sort=False)["id_a"].transform(
        lambda values: np.gradient(values.to_numpy(), VGS_STEP_V)
    )
    body_keys = ["polarity", "width_um", "length_um", "body_voltage_v", "vgs_v"]
    table["gds_s"] = table.groupby(body_keys, sort=False)["id_a"].transform(
        lambda values: np.gradient(
            values.to_numpy(),
            table.loc[values.index, "vds_v"].to_numpy(),
        )
    )
    table["current_density_a_per_um"] = table["id_a"] / table["width_um"]
    table["gm_over_id_per_v"] = np.where(
        table["id_a"] > 0,
        table["gm_s"] / table["id_a"],
        np.nan,
    )
    table["gm_over_gds"] = np.where(
        table["gds_s"] > 0,
        table["gm_s"] / table["gds_s"],
        np.nan,
    )
    table["vdsat_95pct_v"] = np.nan
    sat_keys = ["polarity", "width_um", "length_um", "body_voltage_v", "vgs_v"]
    for _, indices in table.groupby(sat_keys, sort=False).groups.items():
        group = table.loc[indices].sort_values("vds_v")
        max_current = group["id_a"].iloc[-1]
        if max_current <= 0:
            continue
        threshold_current = 0.95 * max_current
        crossing = np.flatnonzero(group["id_a"].to_numpy() >= threshold_current)
        if crossing.size:
            table.loc[indices, "vdsat_95pct_v"] = group["vds_v"].iloc[crossing[0]]

    # Constant-current threshold: |ID|/W = 100 nA/um, at each drain bias.
    threshold_keys = ["polarity", "width_um", "length_um", "body_voltage_v", "vds_v"]
    table["vth_cc_100na_per_um_v"] = np.nan
    for _, indices in table.groupby(threshold_keys, sort=False).groups.items():
        group = table.loc[indices].sort_values("vgs_v")
        current_density = group["current_density_a_per_um"].to_numpy()
        crossing = np.flatnonzero(current_density >= 100e-9)
        if not crossing.size:
            continue
        j = crossing[0]
        if j == 0:
            threshold = group["vgs_v"].iloc[0]
        else:
            x = group["vgs_v"].to_numpy()[j - 1 : j + 1]
            y = current_density[j - 1 : j + 1]
            threshold = np.interp(100e-9, y, x)
        table.loc[indices, "vth_cc_100na_per_um_v"] = threshold
    return table


def _make_recommendations(table):
    nominal = table[np.isclose(table["vds_v"], 0.6)].copy()
    targets = (5.0, 10.0, 15.0, 20.0)
    rows = []
    keys = ["polarity", "width_um", "length_um", "body_voltage_v"]
    for _, group in nominal.groupby(keys, sort=True):
        group = group.sort_values("vgs_v")
        gm_id = group["gm_over_id_per_v"].to_numpy()
        vgs = group["vgs_v"].to_numpy()
        density = group["current_density_a_per_um"].to_numpy()
        gmgds = group["gm_over_gds"].to_numpy()
        for target in targets:
            indices = np.flatnonzero(np.isfinite(gm_id) & (gm_id >= target))
            if not indices.size:
                continue
            j = indices[-1]
            if j == len(gm_id) - 1:
                continue
            gm_bracket = gm_id[j : j + 2][::-1]
            if gm_bracket[0] == gm_bracket[1]:
                continue
            rows.append(
                {
                    "polarity": group["polarity"].iloc[0],
                    "width_um": group["width_um"].iloc[0],
                    "length_um": group["length_um"].iloc[0],
                    "body_voltage_v": group["body_voltage_v"].iloc[0],
                    "body_bias_v": group["body_bias_v"].iloc[0],
                    "vds_v": 0.6,
                    "target_gm_over_id_per_v": target,
                    "vgs_v": np.interp(target, gm_bracket, vgs[j : j + 2][::-1]),
                    "current_density_a_per_um": np.interp(
                        target, gm_bracket, density[j : j + 2][::-1]
                    ),
                    "gm_over_gds": np.interp(
                        target, gm_bracket, gmgds[j : j + 2][::-1]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _plot_results(table, output_dir):
    nominal = table[
        np.isclose(table["vds_v"], 0.6)
        & np.isclose(table["body_bias_v"], 0.0)
        & np.isclose(table["width_um"], 20.0)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for polarity, group in nominal.groupby("polarity"):
        for length, length_group in group.groupby("length_um"):
            label = f"{polarity}, L={length:g} um"
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
    fig.suptitle("SKY130 TT MOSFET characterization, W=20 um, VDS=0.6 V")
    fig.tight_layout()
    fig.savefig(output_dir / "gm_id_overview.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    body_key = ["polarity", "width_um", "length_um", "body_voltage_v"]
    target = table[
        np.isclose(table["vds_v"], 0.6)
        & np.isclose(table["width_um"], 20.0)
        & np.isclose(table["length_um"], 0.5)
    ]
    for _, group in target.groupby(body_key):
        group = group.sort_values("vgs_v")
        polarity = group["polarity"].iloc[0]
        bias = group["body_bias_v"].iloc[0]
        axes[0].plot(
            group["vgs_v"],
            group["current_density_a_per_um"] * 1e6,
            label=f"{polarity}, body bias={bias:g} V",
        )
        axes[1].plot(
            group["vgs_v"],
            group["gm_over_id_per_v"],
            label=f"{polarity}, body bias={bias:g} V",
        )
    axes[0].set(xlabel="VGS / VSG (V)", ylabel="|ID| / W (uA/um)")
    axes[1].set(xlabel="VGS / VSG (V)", ylabel="gm / ID (1/V)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Body-effect sensitivity, W=20 um, L=0.5 um, VDS=0.6 V")
    fig.tight_layout()
    fig.savefig(output_dir / "body_effect.png", dpi=180)
    plt.close(fig)


def characterize(output_dir=OUTPUT, quick=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = (7.0, 20.0) if quick else WIDTHS_UM
    lengths = (0.15, 0.5) if quick else LENGTHS_UM
    frames = []
    with tempfile.TemporaryDirectory(prefix="sky130_mos_") as temporary:
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller geometry subset to validate setup and output format.",
    )
    args = parser.parse_args()
    characterize(args.output_dir, quick=args.quick)


if __name__ == "__main__":
    main()
