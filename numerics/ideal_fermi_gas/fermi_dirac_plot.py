import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 1.0  # chemical potential
T_values = [0.1, 0.2, 0.5]  # different temperatures

# Energy range
E = np.linspace(0, 2, 500)

# Fermi-Dirac distribution function
def fermi_dirac(E, mu, T):
    return 1.0 / (np.exp((E - mu) / T) + 1.0)

# Plot
plt.figure(figsize=(6,4))
for T in T_values:
    f = fermi_dirac(E, mu, T)
    plt.plot(E, f, label=f"$T={T}$")

# Add T=0 limit (step function)
f_zero = np.heaviside(mu - E, 1.0)
plt.plot(E, f_zero, "k--", label="$T=0$")

plt.xlabel(r"Energy $E$")
plt.ylabel(r"Occupation $f(E)$")
plt.title("Fermi--Dirac Distribution")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("fermi_dirac_distribution.pdf", dpi=300)
plt.show()
