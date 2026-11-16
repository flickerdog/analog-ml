import subprocess
import tempfile
from pathlib import Path

from src.ngspice_backend import MODELS
from src.ota5t import NETLIST, VDD

net = NETLIST.substitute(models=MODELS, vdd=VDD, w1=20, l1=0.5,
                         w3=20, l3=0.5, itail=20e-6, cl=1e-12)
head = net.split(".ac dec")[0]          # keep the circuit, drop the AC setup

params = ["vdsat", "vth", "id"]
vecs = [f"@m.xm{i}.msky130_fd_pr__{kind}_01v8[{p}]"
        for i, kind in [(1, "nfet"), (2, "nfet"), (3, "pfet"), (4, "pfet")]
        for p in params]

control = (".control\nsave all " + " ".join(vecs) + "\nop\n"
           "print v(n1) v(out) v(tail) v(inn)\n"
           + "".join(f"print {v}\n" for v in vecs) + "quit\n.endc\n.end\n")

with tempfile.TemporaryDirectory() as tmp:
    Path(tmp, "c.cir").write_text(head + control)
    r = subprocess.run(["ngspice", "-b", "c.cir"], cwd=tmp, capture_output=True, text=True)
print(r.stdout)
print(r.stderr)