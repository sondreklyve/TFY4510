import argparse
import numpy as np
import matplotlib.pyplot as plt

from numerics.npemu.core import (
    solve_composition,
    make_eos,
    dMdr,
    dPdr,
    e0,
    rho0,
    MeV4togcm3,
    dynetoMeV4,
)
from numerics.npemu.eos_models import (
    build_eos_uniform,
    build_eos_rmf_plus_crust,
    build_eos_polytrope_crust
)


def plot_eos(eps_uniform, P_uniform, eps_core, P_core, eps_crust, P_crust, model: str):
    """
    Plot core npeμ EoS and (if present) crust/polytrope branch
    in physical units: log10 P vs log10 epsilon.
    """
    # core
    eps_core_phys = eps_core * e0 * MeV4togcm3
    P_core_phys = P_core * e0 / dynetoMeV4
    eps_uniform_phys = eps_uniform * e0 * MeV4togcm3
    P_uniform_phys = P_uniform * e0 / dynetoMeV4
    log_eps_core = np.log10(eps_core_phys)
    log_P_core = np.log10(P_core_phys)
    log_eps_uniform = np.log10(eps_uniform_phys)
    log_P_uniform = np.log10(P_uniform_phys)

    plt.figure(figsize=(5.5, 4))
    plt.plot(log_eps_core, log_P_core, 'k-', label='core (npeμ)')
    plt.plot(log_eps_uniform, log_P_uniform, 'k--', label='extended core (npeμ)')

    # crust / polytrope branch, if present
    if eps_crust is not None and P_crust is not None:
        eps_crust_phys = eps_crust * e0 * MeV4togcm3
        P_crust_phys = P_crust * e0 / dynetoMeV4
        log_eps_crust = np.log10(eps_crust_phys)
        log_P_crust = np.log10(P_crust_phys)
        plt.plot(log_eps_crust, log_P_crust, 'r-', label='polytrope')

    plt.grid(True)
    plt.xlabel(r'$\log_{10} \epsilon\;(\mathrm{g/cm}^3)$')
    plt.ylabel(r'$\log_{10} P\;(\mathrm{dyne/cm}^2)$')
    plt.title(f'EoS: {model}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_composition():
    """
    Plot log10(rho_i / rho) vs baryon density rho (fm^-3)
    for i = n, p, e, μ with clean labels.
    """

    comp = solve_composition()

    rhos = comp["rhos"]                 # total baryon density (MeV^3)
    rhonnorm = comp["rhonnorm"]         # ρ_n / ρ
    rhoesnorm = comp["rhoesnorm"]       # ρ_e / ρ
    rhomuonsnorm = comp["rhomuonsnorm"] # ρ_μ / ρ

    # proton fraction is 1 - neutron fraction
    rhopnorm = 1.0 - rhonnorm

    # Convert total baryon density to fm^-3
    rhos_fm3 = rhos / rho0

    # log10 of the fractions
    log_n = np.log10(rhonnorm)
    log_p = np.log10(rhopnorm)
    log_e = np.log10(rhoesnorm)
    log_mu = np.log10(rhomuonsnorm)

    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.grid(True, linestyle=":", linewidth=0.7)

    plt.plot(rhos_fm3, log_n,  'b',    label="n")
    plt.plot(rhos_fm3, log_p,  'r',    label="p")
    plt.plot(rhos_fm3, log_e,  'b--',  label="e")
    plt.plot(rhos_fm3, log_mu, 'k--',  label=r"$\mu$")

    plt.xlabel(r"$\rho\;(\mathrm{fm}^{-3})$")
    plt.ylabel(r"$\log_{10}(\rho_i/\rho)$")
    plt.title("npeμ composition")

    plt.xlim(0, 1.0)
    plt.ylim(-3, 0)

    plt.legend()
    plt.tight_layout()
    plt.show()



def run_tov(EoS, Pcstart, Pcend, Pcstep, tol, r_max=100.0, rstep=0.001):
    Mlist = []
    Rlist = []
    centralpressures = []

    Pc = Pcstart
    while Pc < Pcend:
        P = Pc
        M = 0.0
        r = 0.0
        centralpressures.append(Pc)

        while P > tol and r < r_max:
            r += rstep
            eps = float(EoS(P))
            M += rstep * dMdr(r, eps)
            P += rstep * dPdr(r, M, P, eps)

        Rlist.append(r)
        Mlist.append(M)
        Pc *= Pcstep

    Mlist = np.array(Mlist)
    Rlist = np.array(Rlist)
    centralpressures = np.array(centralpressures)

    idx_max = int(np.argmax(Mlist))
    StableM = Mlist[: idx_max + 1]
    StableR = Rlist[: idx_max + 1]
    UnstableM = Mlist[idx_max + 1 :]
    UnstableR = Rlist[idx_max + 1 :]

    centraldensities = np.array([float(EoS(Pc)) * e0 * MeV4togcm3 for Pc in centralpressures])
    Stablecd = centraldensities[: idx_max + 1]
    Unstablecd = centraldensities[idx_max + 1 :]

    return dict(
        Mlist=Mlist,
        Rlist=Rlist,
        centralpressures=centralpressures,
        centraldensities=centraldensities,
        StableM=StableM,
        StableR=StableR,
        UnstableM=UnstableM,
        UnstableR=UnstableR,
        Stablecd=Stablecd,
        Unstablecd=Unstablecd,
        idx_max=idx_max,
    )


def run_model(model: str):
    # 1) Composition
    comp = solve_composition()
    rhos = comp["rhos"]
    rhons = comp["rhons"]
    gsigmas = comp["gsigmas"]
    kes = comp["kes"]
    kmuons = comp["kmuons"]

    # 2) Core npeμ EoS
    energy, pressure = make_eos(rhos, rhons, gsigmas, kes, kmuons)

    # 3) Build global EoS according to chosen model
    if model == "crust":
        data = build_eos_rmf_plus_crust(energy, pressure)
    elif model == "uniform":
        data = build_eos_uniform(energy, pressure)
    elif model == "polytrope":
        data = build_eos_polytrope_crust(energy, pressure)
    else:
        raise ValueError(f"Unknown model '{model}'")

    data_uniform = build_eos_uniform(energy, pressure)
    EoS = data["EoS"]
    joined_energy = data["joined_energy"]
    joined_pressure = data["joined_pressure"]

    plot_eos(
        data_uniform["eps_core"],
        data_uniform["P_core"],
        data["eps_core"],
        data["P_core"],
        data["eps_crust"],
        data["P_crust"],
        model=model,
    )

    # 4) TOV integration
    Pcstart = 4e33 * dynetoMeV4 / e0  # dimensionless
    Pcend = joined_pressure[-2]
    Pcstep = 1.05
    tol = joined_pressure[0]

    tov = run_tov(EoS, Pcstart, Pcend, Pcstep, tol)

    idx_max = tov["idx_max"]
    print(f"Model: {model}")
    print("Radius at maximum mass (km):", tov["Rlist"][idx_max])
    print("Maximum mass (M_sun):", tov["Mlist"][idx_max])

    # 5) Plots (M–R and M–ε_c)
    StableM = tov["StableM"]
    StableR = tov["StableR"]
    UnstableM = tov["UnstableM"]
    UnstableR = tov["UnstableR"]
    Stablecd = tov["Stablecd"]
    Unstablecd = tov["Unstablecd"]

    Stablecd_log = np.log10(Stablecd)
    Unstablecd_log = np.log10(Unstablecd)

    fig, axarr = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

    # ---- Left: Mass–Radius ----
    axarr[0].grid(True)
    axarr[0].plot(StableR, StableM, 'k-', label='stable')
    axarr[0].plot(UnstableR, UnstableM, 'k--', label='unstable')
    axarr[0].set_xlabel('Radius (km)')
    axarr[0].set_ylabel(r'$M/M_\odot$')
    axarr[0].set_xlim([8, 25])
    axarr[0].set_title("Mass–Radius Relation")
    axarr[0].legend()

    # ---- Right: Mass vs central density ----
    axarr[1].grid(True)
    axarr[1].plot(Stablecd_log, StableM, 'k-')
    axarr[1].plot(Unstablecd_log, UnstableM, 'k--')
    axarr[1].set_xlabel(r'$\log_{10}\,\epsilon_c\;(\mathrm{g/cm}^3)$')
    axarr[1].set_ylabel(r'$M/M_\odot$')
    axarr[1].set_xlim([14.45, 15.8])
    axarr[1].set_title("Mass vs Central Energy Density")

    # Overall figure title (optional)
    model_title = model if model != "polytrope" else f"polytrope (Γ=1.2)"
    fig.suptitle(
        f"Model: {model_title}",
        fontsize=18,
        y=0.9,       # lift it slightly above the plots
    )


    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    plot_composition()
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["crust", "uniform", "polytrope"],
        default="crust",
        help="Which global EoS model to use.",
    )

    args = parser.parse_args()

    run_model(args.model)
