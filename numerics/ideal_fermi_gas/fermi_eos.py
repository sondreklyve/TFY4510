import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# dimensionless Fermi momentum range
xF = np.logspace(-2, 2, 500)

# functions for epsilon(xF) and p(xF)
def epsilon(x):
    return (1/(8*np.pi**2)) * (x*np.sqrt(1+x**2)*(2*x**2+1) - np.arcsinh(x))

def pressure(x):
    return (1/(24*np.pi**2)) * (
        x*np.sqrt(1+x**2)*(2*x**2-3) + 3*np.arcsinh(x)
    )

eps = epsilon(xF)
p = pressure(xF)
ratio = p / eps

# --- use viridis-like colors ---
cmap = plt.cm.viridis
color_main = cmap(0.3)

plt.figure(figsize=(8, 5.5))

# main curve
plt.loglog(xF, ratio, color=color_main, linewidth=2.2, label=r"$P/\epsilon$")

# ultra-relativistic limit
plt.axhline(
    1/3,
    color="black",
    linestyle="--",
    linewidth=1.8,
    label=r"Ultra-relativistic limit $P = \epsilon/3$"
)

plt.xlabel(r"Fermi momentum ratio $x_F = p_F/m$", fontsize=18)
plt.ylabel(r"Stiffness ratio $P/\epsilon$", fontsize=18)
plt.title(r"EoS in the Non-Relativistic and Ultra-Relativistic Limits", fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14, frameon=True)
plt.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
# plt.savefig("fermi_eos.pdf", dpi=300)
plt.show()
