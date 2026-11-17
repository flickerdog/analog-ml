import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ngspice_backend import MODELS, simulate
from src.ota5t import DEVICES, NETLIST, VDD, ac_metrics, parse_op

COLS = ["w1", "l1", "w3", "l3", "itail"]


def vec(dev, p):
    return f"@m.x{dev}.msky130_fd_pr__{DEVICES[dev]}_01v8[{p}]"


EXTRA = [vec(d, p) for d in DEVICES for p in ("gm", "gds", "id")]
OPCMDS = ("save all " + " ".join(EXTRA) + "\nop\n"
          + "\n".join(f"print {v}" for v in EXTRA))


def inspect(row):
    net = NETLIST.substitute(models=MODELS, vdd=VDD, cl=1e-12, opcmds=OPCMDS,
                             **row[COLS].to_dict())
    data, out = simulate(net, "ac.txt")
    gain_ac = ac_metrics(data[:, 0], data[:, 1] + 1j * data[:, 2])[0]
    op = parse_op(out)
    gm1 = op[vec("m1", "gm")]
    gds2, gds4 = op[vec("m2", "gds")], op[vec("m4", "gds")]
    id1 = abs(op[vec("m1", "id")])
    return {
        "gain_ac_dB": gain_ac,
        "gain_op_dB": 20 * np.log10(gm1 / (gds2 + gds4)),
        "gm1/Id": gm1 / id1,
        "gm/gds M1": gm1 / op[vec("m1", "gds")],
        "gm/gds M4": op[vec("m4", "gm")] / gds4,
    }


df = pd.read_csv("data/ota5t_sobol.csv")
ok = df[df["valid"] == True].sort_values("gain_db").reset_index(drop=True)
picks = {"lowest": ok.iloc[0], "median": ok.iloc[len(ok) // 2], "highest": ok.iloc[-1]}

rows = []
for name, r in picks.items():
    rows.append({"design": name, **r[COLS].to_dict(), **inspect(r)})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Gain vs the shorter of the two lengths, coloured by inversion level
lmin = np.minimum(ok.l1, ok.l3)
dens = np.log10(ok.itail / 2 * ok.l1 / ok.w1)   # log10 Id/(W/L) of the input pair
plt.figure(figsize=(7, 4.5))
sc = plt.scatter(lmin, ok.gain_db, c=dens, s=6)
plt.xscale("log")
plt.xlabel("min(L1, L3) (um)")
plt.ylabel("DC gain (dB)")
plt.colorbar(sc, label="log10( Id / (W/L) ), higher = stronger inversion")
plt.tight_layout()
plt.savefig("data/gain_vs_l.png", dpi=150)