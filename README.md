# analog-ml

Machine-learning-assisted sizing of analog circuits, using only open-source tools:
[ngspice](https://ngspice.sourceforge.io/) and the open
[SKY130](https://github.com/google/skywater-pdk) PDK.

The goal is to find out how far surrogate models and optimizers can go on real
analog design problems, starting with a deliberately simple circuit.
**Work in progress:** the data pipeline and first nominal surrogate are in
place; using the surrogate to search for better designs is next.

## What it does

1. **Simulates** thousands of sizings of a 5-transistor OTA with ngspice and
   extracts DC gain, gain-bandwidth product (GBW), phase margin (PM) and power.
2. **Classifies** every design against the requirements in
   [`specs.yaml`](specs.yaml): *invalid* (a transistor is out of saturation, or
   the tail node has no headroom), *valid but misses a spec*, or *passes*.
3. **Plots** where the passing designs live in the design space, explores how
   gain varies with channel length, and writes a summary table.
4. **Trains a first surrogate** to estimate nominal circuit metrics from the
   design variables, and checks its predictions on held-out sweep results.
   The evaluation is written as an HTML report with plots and a table of
   predicted-pass designs that failed ngspice verification.
5. *(next)* Uses the surrogate to search for promising designs, then verifies
   those designs with ngspice.

![Performance space](docs/performance_space.png)
![Fraction of designs passing along each design variable](docs/passrate_vs_variable.png)
![DC gain versus minimum channel length, colored by input-pair current density](docs/gain_vs_l.png)

## The circuit and the design space

A 5-transistor OTA: NMOS differential pair (M1, M2), PMOS current-mirror load
(M3, M4) and an **ideal tail current source**. Open-loop AC response is measured
with the usual trick: DC unity feedback through a huge inductor, inverting
input grounded for AC by a huge capacitor.

| Variable | Meaning | Range (sampled log-uniformly) |
|---|---|---|
| `w1`, `l1` | input pair width, length | 7 to 50 µm, 0.15 to 2 µm |
| `w3`, `l3` | load pair width, length | 7 to 50 µm, 0.15 to 2 µm |
| `itail` | tail current | 100 nA to 100 µA |

Designs are drawn with a Sobol sequence (4096 points) so the space is covered
evenly. The load capacitance, performance requirements, and validity
thresholds are configured in [`specs.yaml`](specs.yaml); the data file records
the load capacitance used for each simulation.

Edit the requirements in `specs.yaml` and re-plot to reclassify existing data;
no re-simulation is needed. A design is *valid* if all four OTA transistors
meet the configured saturation-margin threshold and the tail node meets its
configured minimum voltage.

## Method notes

- **What the surrogate learns.** `train_surrogate.py` uses scikit-learn's
  `RandomForestRegressor`. It trains one model for each of five simulated
  outputs: gain, GBW, phase margin, saturation margin, and tail voltage. Power
  is calculated directly from the tail current, so it does not need a model.
- **How a random forest works.** Each decision tree learns simple split rules
  on the five inputs and predicts by averaging the training results in the
  matching leaf (the final region reached by those rules). By default, the
  forest averages 300 trees to make predictions less sensitive to any one
  tree. The inputs are converted to `log10` because the sweep varies them over
  multiplicative ranges; GBW is also learned in `log10` space.
- **How it is evaluated.** By default, the script holds out 20% of the sweep,
  trains on the other 80%, and compares predictions with ngspice results for
  those unseen rows. It reports mean absolute error (MAE), root mean squared
  error (RMSE), and R², alongside a simple mean-only prediction baseline.
  After this check, it trains a final model on all rows for later use. A random
  holdout measures performance within this sweep; it does not guarantee
  accuracy in new regions or near every spec boundary.
- **Trimmed model files.** Loading the full SKY130 typical corner makes ngspice
  parse about 30 device families and takes roughly 50 s per run. This project
  includes only the 1.8 V NMOS and PMOS models, which takes about 0.1 s per
  run and gave identical results on a test transistor.
- **Model binning.** SKY130 models are split into width/length bins. Measured
  steps in gm/Id across bin edges were a few tenths of a percent at most, so W
  and L are sampled continuously.
- **Serial simulation.** On the development machine, running ngspice in
  parallel was slower than a plain loop, so the sweep runs serially
  (about 8 minutes for 4096 designs).

## Limitations

- Ideal tail current source, so the tail device is not modelled.
- Typical corner and 27 °C only: no PVT, mismatch, noise or layout parasitics.
- The current sweep is nominal and does not model Monte Carlo variation or
  input offset; the surrogate trained from it cannot estimate offset.
- Lower tail currents limit achievable GBW, particularly with larger load
  capacitance.

## Reproduce

```bash
# system packages (Debian/Ubuntu)
sudo apt install -y python3-venv git ngspice

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pre-built SKY130 PDK via ciel
ciel ls-remote --pdk-family=sky130            # list versions
ciel enable --pdk-family=sky130 <hash>        # version used for the results: <hash>

# Simulate, plot, write the table
python run_all.py --sweep

# Train and evaluate the surrogate using the saved sweep data
python run_all.py --surrogate
```

The surrogate command creates `reports/surrogate_report.html`, which you can
open in a browser. It includes a metric table (average prediction errors and
a simple baseline), plots comparing predictions with ngspice on held-out
designs, an error plot, and a table of predicted passes that failed simulation.
`reports/surrogate_test_predictions.csv` contains the inputs and actual and
predicted values for every held-out design. These files are generated locally
and are not tracked by git. The plots below are a snapshot of the current
4,096-row sweep; rerun `python run_all.py --surrogate --docs` to refresh them
after changing the data.

In the predicted-versus-simulated plots, each dot is one test design and dots
closer to the dashed diagonal mean more accurate predictions. GBW uses
logarithmic axes. The error plots show `ngspice result − prediction`; zero
means no error. MAE is the average absolute error, RMSE emphasizes larger
errors, and R² compares the model with a mean-only prediction. The
classification table is important for design search: a predicted pass is
only a candidate, and must still be checked with ngspice. This first model
uses nominal sweep data only; it cannot estimate Monte Carlo variation or 3σ
input offset.

![Surrogate predictions compared with ngspice on held-out designs](docs/surrogate_actual_vs_predicted.png)
![Surrogate prediction errors on held-out designs](docs/surrogate_prediction_errors.png)

### Surrogate training flow

```mermaid
flowchart LR
    A["Sweep inputs<br/>w1, l1, w3, l3, itail"] --> B["ngspice<br/>simulated metrics"]
    B --> C["Saved sweep CSV"]
    A --> C
    C --> D["Log-transform inputs<br/>and split 80% / 20%"]
    D --> E["Train five random-forest<br/>regressors on 80%"]
    E --> F["Predict the held-out 20%"]
    F --> G["Compare with ngspice<br/>metrics and spec results"]
    G --> H["HTML report, plots,<br/>and prediction CSV"]
    C --> I["Refit five models<br/>on all available rows"]
    I --> J["Saved model for<br/>later design search"]
```

In the diagram, the held-out 20% (the default) is used only to evaluate
predictions, not to fit those evaluated models. The final saved model is then
retrained using the full dataset. At present, the saved model is not yet
connected to an optimizer. The test fraction and number of trees can be
changed with `--test-size` and `--trees` when running
`python train_surrogate.py --help`.

## Scripts

| File | Role |
|---|---|
| `run_all.py` | One command for the whole pipeline: sweep, plots, table. Options: `--sweep`, `--surrogate`, `--docs` |
| `specs.yaml` | Requirements, validity thresholds and test conditions |
| `sweep_ota.py` | Simulates the design points and writes `data/ota5t_sobol.csv` |
| `plot_results.py` | Four figures in `reports/` |
| `report_sweep.py` | HTML tables in `reports/` |
| `train_surrogate.py` | Trains and evaluates nominal metric models; writes an HTML report and plots to `reports/`, and a model to `data/` |
| `try_ota.py` | Smoke test: simulates one design in about 0.1 s |
| `src/ngspice_backend.py` | Model includes and the function that runs ngspice in a scratch folder |
| `src/ota5t.py` | OTA netlist, metric extraction and `evaluate()` |
| `src/specs.py` | Reads `specs.yaml` and classifies designs |
| `src/tables.py` | Unit conversion and HTML table helpers |
| `src/plotting.py` | Plot style and log-axis helper |
| `sims/` | First single-transistor experiments |

`data/` and `reports/` are generated and not tracked by git. The figures shown
above are copied to `docs/` with `python run_all.py --docs`.
Surrogate training is implemented; using it to optimize designs is still
planned.

## Data and licensing

Only open-source models and tools are used. The repository contains no
proprietary PDK data.
