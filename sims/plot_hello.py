import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("id.txt")
vgs = data[:, 0]
id_ = -data[:, 1]   # SPICE reports the source current with the opposite sign

plt.semilogy(vgs, id_)
plt.xlabel("Vgs (V)")
plt.ylabel("Id (A)")
plt.grid(True)
plt.savefig("hello.png", dpi=150)
plt.show()