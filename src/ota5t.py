from string import Template

import numpy as np

from src.ngspice_backend import MODELS, simulate

VDD = 1.8

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
run
wrdata ac.txt v(out)
quit
.endc
.end
""")


def ac_metrics(freq, h):
    """DC gain (dB), unity-gain frequency (Hz) and phase margin (deg) from H(f)."""
    mag_db = 20 * np.log10(np.abs(h))
    phase = np.degrees(np.unwrap(np.angle(h)))
    logf = np.log10(freq)
    dc_gain = mag_db[0]

    below = np.where(mag_db < 0)[0]          # points where gain < 0 dB
    if len(below) == 0 or below[0] == 0:     # never crosses 0 dB
        return dc_gain, np.nan, np.nan
    i = below[0]
    lf = np.interp(0.0, [mag_db[i], mag_db[i - 1]], [logf[i], logf[i - 1]])
    ph = np.interp(lf, [logf[i - 1], logf[i]], [phase[i - 1], phase[i]])
    return dc_gain, 10 ** lf, 180 + (ph - phase[0])


def evaluate(w1, l1, w3, l3, itail=20e-6, cl=1e-12):
    netlist = NETLIST.substitute(
        models=MODELS, vdd=VDD, w1=w1, l1=l1, w3=w3, l3=l3, itail=itail, cl=cl
    )
    data = simulate(netlist, "ac.txt")
    assert data.shape[1] == 3, f"expected 3 columns (f, re, im), got {data.shape}"
    freq = data[:, 0]
    h = data[:, 1] + 1j * data[:, 2]
    gain_db, gbw, pm = ac_metrics(freq, h)
    return {"gain_db": gain_db, "gbw_hz": gbw, "pm_deg": pm, "power_w": VDD * itail}