#!/usr/bin/env python3
# Analytic incompressible star (uniform density): GR vs Newtonian
# Plots m/M, P/epsilon0 (GR & Newtonian), and metric functions e^{2alpha}, e^{-2beta}
# Units: G = c = 1

import numpy as np
import matplotlib.pyplot as plt

def profiles_uniform_density(C, R=1.0, npts=400):
    """
    Analytic interior profiles for a uniform-density star at compactness C=2M/R < 8/9.
    Returns dict with arrays sampled on x=r/R in [0,1].
    """
    if not (0.0 < C < 8.0/9.0):
        raise ValueError("Compactness must satisfy 0 < C < 8/9 for a regular fluid solution.")

    M = 0.5 * C * R
    eps0 = 3.0 * M / (4.0 * np.pi * R**3)  # = 3C/(8π) for R=1

    x = np.linspace(0.0, 1.0, npts)
    r = x * R

    # Mass function and Phi
    m = (4.0*np.pi/3.0) * eps0 * r**3                  # = M * x^3
    m_over_M = m / M
    Phi = np.sqrt(1.0 - C * x**2)
    Phi_R = np.sqrt(1.0 - C)

    # GR pressure
    P = eps0 * (Phi - Phi_R) / (3.0*Phi_R - Phi)       # eq. (const_density_pressure)
    P_over_eps0 = P / eps0

    # Newtonian pressure (same rho=eps0 in c=1)
    Pn = (3.0/(8.0*np.pi)) * (M**2 / R**4) * (1.0 - x**2)
    Pn_over_eps0 = Pn / eps0   # equals C/4 at x=0

    # Metric functions
    e2alpha = 0.25 * (3.0*Phi_R - Phi)**2
    e2beta  = 1.0 / (1.0 - C * x**2)                   # = (1 - 2m/r)^{-1}
    einv2beta = 1.0 / e2beta                           # = 1 - C x^2

    return dict(
        x=x,
        m_over_M=m_over_M,
        P_over_eps0=P_over_eps0,
        Pn_over_eps0=Pn_over_eps0,
        e2alpha=e2alpha,
        e2beta=e2beta,
        einv2beta=einv2beta,
        C=C,
        Phi_R=Phi_R
    )

def plot_all(compactness_list=(0.1, 0.3, 0.6), R=1.0):
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(compactness_list)))

    # Figure 1: Pressure profiles (GR vs Newtonian)
    plt.figure(figsize=(6.2, 5.2))
    for color, C in zip(colors, compactness_list):
        prof = profiles_uniform_density(C, R=R)
        x = prof["x"]
        plt.plot(x, prof["P_over_eps0"], lw=2, color=color, label=fr"$\mathcal{{C}}={C:.2f}$")
        plt.plot(x, prof["Pn_over_eps0"], ls="--", lw=1.6, color=color, alpha=0.9)

    plt.xlabel(r"$r/R$")
    plt.ylabel(r"$P/\epsilon_0$")
    plt.title("Uniform-density star: pressure profiles")
    # plt.title("Uniform-density star: Buchdal limit")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, ncol=1)
    plt.tight_layout()

    # # Figure 2: Mass profile m/M
    # plt.figure(figsize=(6.2, 5.2))
    # # m/M = x^3, independent of C — plot once
    # x = np.linspace(0, 1, 400)
    # plt.plot(x, x**3, lw=2)
    # plt.xlabel(r"$r/R$")
    # plt.ylabel(r"$m(r)/M$")
    # plt.title(r"Uniform-density star: $m(r)/M = (r/R)^3$")
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()

    # # Figure 3: Metric functions
    # plt.figure(figsize=(6.6, 5.2))
    # for color, C in zip(colors, compactness_list):
    #     prof = profiles_uniform_density(C, R=R)
    #     x = prof["x"]
    #     # e^{2alpha} and 1/e^{2beta} = 1 - C x^2
    #     plt.plot(x, prof["e2alpha"], lw=2, color=color, label=fr"$e^{{2\alpha}}$, $\mathcal{{C}}={C:.2f}$")
    #     plt.plot(x, prof["einv2beta"], lw=1.6, ls="--", color=color, alpha=0.9, label=r"$1/e^{2\beta}$")

    # plt.xlabel(r"$r/R$")
    # plt.ylabel(r"Metric components")
    # plt.title(r"Uniform-density interior: $e^{2\alpha}$ (solid), $1/e^{2\beta}=1-\mathcal{C}(r/R)^2$ (dashed)")
    # plt.grid(True, alpha=0.3)
    # plt.legend(frameon=False, ncol=2)
    # plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    # Choose any set of compactness values strictly below 8/9 ≈ 0.888...
    plot_all(compactness_list=(0.05, 0.10, 0.25))
