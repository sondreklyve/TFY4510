import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# Parameters
mu = 1.0  # chemical potential
T_values = [0.1, 0.2, 0.5]  # different temperatures

# Energy range
E = np.linspace(0, 2, 500)

# Fermi-Dirac distribution function
def fermi_dirac(E, mu, T):
    return 1.0 / (np.exp((E - mu) / T) + 1.0)

# Use viridis-like colors for the finite-T curves
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.8, len(T_values)))

plt.figure(figsize=(8, 5.5))

for T, col in zip(T_values, colors):
    f = fermi_dirac(E, mu, T)
    plt.plot(E, f, color=col, label=rf"$T = {T}$", linewidth=2.2)

# Add T=0 limit (step function) in black dashed
f_zero = np.heaviside(mu - E, 1.0)
plt.plot(E, f_zero, "k--", label=r"$T = 0$", linewidth=2.2)

plt.xlabel(r"Energy $E$", fontsize=18)
plt.ylabel(r"Occupation $f(E)$", fontsize=18)
plt.title(r"Fermi--Dirac Distribution", fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(loc="best", fontsize=14, frameon=True)
plt.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
# plt.savefig("fermi_dirac_distribution.pdf", dpi=300)
plt.show()
