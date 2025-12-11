import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14


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
    m = (4.0*np.pi/3.0) * eps0 * r**3
    m_over_M = m / M
    Phi = np.sqrt(1.0 - C * x**2)
    Phi_R = np.sqrt(1.0 - C)

    # GR pressure
    P = eps0 * (Phi - Phi_R) / (3.0*Phi_R - Phi)
    P_over_eps0 = P / eps0

    # Newtonian pressure
    Pn = (3.0/(8.0*np.pi)) * (M**2 / R**4) * (1.0 - x**2)
    Pn_over_eps0 = Pn / eps0

    # Metric functions
    e2alpha = 0.25 * (3.0*Phi_R - Phi)**2
    e2beta  = 1.0 / (1.0 - C * x**2)
    einv2beta = 1.0 / e2beta

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


def plot_pressure_profiles(compactness_list=(0.1, 0.3, 0.6), R=1.0):
    """
    Plot GR and Newtonian pressure profiles P(r)/eps0 for a set of compactness values.
    """
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(compactness_list)))

    plt.figure(figsize=(8, 5.5))
    for C, col in zip(compactness_list, colors):
        prof = profiles_uniform_density(C, R=R)
        x = prof["x"]

        # GR pressure
        plt.plot(
            x, prof["P_over_eps0"],
            lw=2.0, color=col,
            label=rf"$\mathcal{{C}} = {C:.2f}$"
        )

        # Newtonian pressure
        plt.plot(
            x, prof["Pn_over_eps0"],
            ls="--", lw=1.5, color=col, alpha=0.9,
        )

    plt.xlabel(r"$r/R$", fontsize=18)
    plt.ylabel(r"$P/\epsilon_0$", fontsize=18)
    plt.title(r"Pressure Profiles for Uniform-Density Stars", fontsize=20)

    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=14, frameon=True, ncol=1)
    plt.tight_layout()
    plt.show()


def plot_buchdahl_limit(compactness_list=(0.7, 0.8, 0.86, 0.88), R=1.0):
    """
    Plot GR pressure profiles for compactness values approaching the Buchdahl limit (C → 8/9).
    """
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(compactness_list)))

    plt.figure(figsize=(8, 5.5))
    for C, col in zip(compactness_list, colors):
        prof = profiles_uniform_density(C, R=R)
        x = prof["x"]

        plt.plot(
            x, prof["P_over_eps0"], lw=2.0, color=col,
            label=rf"$\mathcal{{C}} = {C:.2f}$"
        )

    plt.xlabel(r"$r/R$", fontsize=18)
    plt.ylabel(r"$P/\epsilon_0$", fontsize=18)
    plt.title(r"Approach to the Buchdahl Limit ($\mathcal{C} \to 8/9$)", fontsize=20)

    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=14, frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_pressure_profiles()
    plot_buchdahl_limit()
