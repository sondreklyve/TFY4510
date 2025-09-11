import numpy as np
import matplotlib.pyplot as plt

# dimensionless Fermi momentum range
xF = np.logspace(-2, 2, 500)

# functions for epsilon(xF) and p(xF) with g=2, m=1 (just scaling)
def epsilon(x):
    return (1/(8*np.pi**2)) * (x*np.sqrt(1+x**2)*(2*x**2+1) - np.arcsinh(x))

def pressure(x):
    return (1/(24*np.pi**2)) * (x*np.sqrt(1+x**2)*(2*x**2-3) + 3*np.arcsinh(x))

eps = epsilon(xF)
p = pressure(xF)

ratio = p / eps

# make the plot
plt.figure(figsize=(6,4))
plt.loglog(xF, ratio, label=r'$p/\epsilon$')
plt.axhline(1/3, color='gray', linestyle='--', label=r'Ultra-relativistic $p=\epsilon/3$')
plt.axhline(0, color='black', linewidth=0.5)

plt.xlabel(r'$x_F = p_F/m$')
plt.ylabel(r'$p/\epsilon$')
plt.title('Equation of state stiffness across regimes')
plt.legend()
plt.tight_layout()

plt.savefig("fermi_eos.pdf")
plt.close()
