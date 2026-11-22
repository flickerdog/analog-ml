# SKY130 MOSFET characterization

Run from the repository root:

```bash
.venv/bin/python -m characterization.characterize_mos
```

The script uses the repository's SKY130 1.8 V typical-corner model setup and
creates `results/mosfet_lut.csv`, `results/gm_id_bias_recommendations.csv`,
and summary plots. Use `--quick` for a smaller geometry check.

The full table sweeps NMOS and PMOS widths of 7, 10, 20, 35, and 50 um;
lengths of 0.15, 0.20, 0.30, 0.50, 1, and 2 um; gate overdrive from 0 to
1.2 V in 10 mV steps; and drain-source voltage from 50 mV to 1.2 V. It
includes three reverse-body-bias conditions for each polarity. The NMOS
source is held at 0.6 V and the PMOS source at 1.2 V, keeping all terminals
within the 0–1.8 V supply range.

The CSV stores positive drain-current magnitude. `gm` and `gds` are extracted
from finite differences of the DC LUT, not from scalar operating-point
parameters. `vth_cc_100na_per_um_v` is the 100 nA/um constant-current
threshold, and `vdsat_95pct_v` is the drain voltage at which the current
reaches 95% of its value at 1.2 V. These definitions are explicit so the
results are reproducible and comparable across geometry.

Body-bias entries are model experiments, not a claim that every connection
is available in the standard process. Standard NMOS devices share the P
substrate and normally use ground as bulk; the additional NMOS body-bias
curves quantify sensitivity only where a suitable isolated-body structure is
available. A PMOS source-tied body represents an individually contacted
N-well; a body tied to 1.8 V represents the common rail-tied alternative.
The results are typical-corner, room-temperature characterizations; process
corner and temperature runs should follow after the nominal operating region
is selected.

## 3.3 V thick-oxide device set

Run the separate thick-oxide characterization with:

```bash
.venv/bin/python -m characterization.characterize_mos_33v
```

The installed SKY130A PDK has no 3.3 V PMOS primitive. This sweep therefore
characterizes the available `nfet_03v3_nvt` together with the higher-rated
`pfet_g5v0d10v5` as a safe PMOS counterpart for a 3.3 V experimental supply.
It is a mixed device-voltage-class design, not a matched 3.3 V complementary
pair. Both devices have model minimum dimensions of W=10 um and L=0.5 um.

The sweep uses W=10, 20, and 50 um; L=0.5, 1, and 2 um; VGS/VSG from 0 to
2.4 V in 10 mV steps; and VDS/VSD from 50 mV to 2.4 V. NMOS source is 0.6 V,
keeping its gate and drain at or below 3.0 V; PMOS source is 3.3 V, with body
at 2.7, 3.0, or 3.3 V. The CSV stores positive current magnitude and includes
gm/Id, gm/gds, constant-current threshold, and the 95%-current saturation
voltage. Results go to `characterization/results_3v3/` so they do not
overwrite the 1.8 V characterization.
