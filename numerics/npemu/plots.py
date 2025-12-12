# plots.py

import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.constants import c
from core import rho0

from tov import run_model


def apply_style():
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.linestyle"] = ":"
    plt.rcParams["grid.alpha"] = 0.5
    plt.rcParams["figure.figsize"] = (8, 5.5)


def load_band(fname, rho_grid, low=5, high=95):
    """
    Load an ensemble EoS file and return low/high percentile bands
    evaluated on a given density grid.

    Parameters
    ----------
    fname : str
        Path to the h5 file.
    rho_grid : array
        Target density grid [g/cm^3].
    low, high : float
        Percentiles for the lower/upper band.

    Returns
    -------
    P_low, P_high : arrays
        Lower/upper pressures [dyn/cm^2] at rho_grid.
    """
    with h5py.File(fname, "r") as f:
        eos_group = f["eos"]
        pressures = []

        for eos_id in eos_group.keys():
            rho = eos_group[eos_id]["energy_densityc2"][:]
            p = eos_group[eos_id]["pressurec2"][:]
            p = p * (c*100) **2

            order = np.argsort(rho)
            rho = rho[order]
            p = p[order]

            p_grid = np.interp(rho_grid, rho, p, left=np.nan, right=np.nan)
            pressures.append(p_grid)

    pressures = np.array(pressures)
    P_low = np.nanpercentile(pressures, low, axis=0)
    P_high = np.nanpercentile(pressures, high, axis=0)
    return P_low, P_high


def plot_composition(comp):
    """
    Plot npeμ composition (number fractions) vs mass density.
    """

    apply_style()

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

    plt.figure(figsize=(8,5.5))

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, 4))
    plt.plot(rhos_fm3, log_n, color=colors[0],    label=r"$n$")
    plt.plot(rhos_fm3, log_p, color=colors[1],    label=r"$p$")
    plt.plot(rhos_fm3, log_e, color=colors[2],  label=r"$e$")
    plt.plot(rhos_fm3, log_mu, color=colors[3],  label=r"$\mu$")

    plt.xlabel(r"$\rho\;(\mathrm{fm}^{-3})$", fontsize=18)
    plt.ylabel(r"$\log_{10}(\rho_i/\rho)$", fontsize=18)
    plt.title(r"$npe\mu$ composition", fontsize=20)

    plt.xlim(0, 1.0)
    plt.ylim(-3, 0)
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_eos_band_comparison(eos_uniform, band_file, low=5, high=95):
    apply_style()

    rho = np.asarray(eos_uniform["rho"]) 
    P   = np.asarray(eos_uniform["P"])

    log_rho = np.log10(rho)
    log_P   = np.log10(P)

    P_low, P_high = load_band(band_file, rho, low=low, high=high)
    log_P_low    = np.log10(P_low)
    log_P_high   = np.log10(P_high)

    plt.figure(figsize=(8, 5.5))

    plt.plot(log_rho, log_P, color="k", lw=2, label=r"$npe\mu$ core EoS")

    plt.fill_between(
        log_rho,
        log_P_low,
        log_P_high,
        color="green",
        alpha=0.3,
        label=r"PSR+GW+X-Ray 90\% band",
    )

    plt.xlabel(r"$\log_{10}\rho\,[\mathrm{g\,cm^{-3}}]$", fontsize=18)
    plt.ylabel(r"$\log_{10}P\,[\mathrm{dyne\,cm^{-2}}]$", fontsize=18)
    plt.title(r"$npe\mu$ EoS vs Ng et al. posterior band", fontsize=20)

    plt.xlim(13.58, 15.55)
    plt.ylim(32, 37)

    plt.grid(True, which="both", alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_tov_results(
    R_main,
    M_main,
    stable_main,
    unstable_main,
    ec_main,
    M_ec_main,
    model
):
    apply_style()

    R_main = np.asarray(R_main)
    M_main = np.asarray(M_main)
    stable_main = np.asarray(stable_main, dtype=bool)
    unstable_main = ~stable_main

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        R_main[stable_main],
        M_main[stable_main],
        color="k",
        lw=2,
        label="stable",
    )
    ax1.plot(
        R_main[unstable_main],
        M_main[unstable_main],
        color="k",
        ls="--",
        lw=2,
        label="unstable",
    )

    ax1.set_xlim([8, 25])
    ax1.set_xlabel("Radius (km)", fontsize=18)
    ax1.set_ylabel(r"$M/M_\odot$", fontsize=18)
    ax1.set_title("Mass–Radius Relation", fontsize=18)
    ax1.legend(fontsize=14)

    ax2.plot(
        np.log10(ec_main[stable_main]),
        M_ec_main[stable_main],
        color="k",
        lw=2,
        label="stable",
    )
    ax2.plot(
        np.log10(ec_main[unstable_main]),
        M_ec_main[unstable_main],
        color="k",
        ls="--",
        lw=2,
        label="unstable",
    )

    ax2.set_xlim([14.45, 15.8])
    ax2.set_xlabel(r"$\log_{10}\,\epsilon_c\;(\mathrm{g/cm^3})$", fontsize=18)
    ax2.set_ylabel(r"$M/M_\odot$", fontsize=18)
    ax2.set_title("Mass vs Central Energy Density", fontsize=18)

    if model == "rmf_crust":
        fig.suptitle("FPS crust", fontsize=20)
    elif model == "polytrope":
        fig.suptitle(r"Polytrope $(\Gamma=1.2)$", fontsize=20)
    fig.tight_layout()
    plt.show()


def plot_eos_comparison(eos_main, eos_uniform, model):
    apply_style()

    rho_m = np.asarray(eos_main["rho"])
    P_m   = np.asarray(eos_main["P"])

    rho_u = np.asarray(eos_uniform["rho"])
    P_u   = np.asarray(eos_uniform["P"])

    idx_m = np.argsort(rho_m)
    rho_m, P_m = rho_m[idx_m], P_m[idx_m]

    idx_u = np.argsort(rho_u)
    rho_u, P_u = rho_u[idx_u], P_u[idx_u]

    eps_cut = 1e14

    mask_crust = rho_m <= eps_cut

    mask_core_solid = rho_u >= eps_cut
    mask_core_dashed = rho_u < eps_cut

    plt.figure(figsize=(8,5.5))

    # --- Crust / polytrope ---
    if np.any(mask_crust):
        plt.plot(
            np.log10(rho_m[mask_crust]),
            np.log10(P_m[mask_crust]),
            color="tab:blue",
            lw=2.0,
            label="FPS crust" if model=="rmf_crust" else "polytrope",
        )

    if np.any(mask_core_dashed):
        plt.plot(
            np.log10(rho_u[mask_core_dashed]),
            np.log10(P_u[mask_core_dashed]),
            "k--",
            lw=2.0,
            label=r"extended core $(npe\mu)$",
        )

    if np.any(mask_core_solid):
        plt.plot(
            np.log10(rho_u[mask_core_solid]),
            np.log10(P_u[mask_core_solid]),
            "k-",
            lw=2.0,
            label=r"core $(npe\mu)$",
        )

    plt.xlabel(r"$\log_{10}\,\epsilon\;(\mathrm{g/cm^3})$", fontsize=18)
    plt.ylabel(r"$\log_{10}\,P\;(\mathrm{dyne/cm^2})$", fontsize=18)

    if model == "polytrope":
        plt.title(r"EoS: Polytrope $(\Gamma = 1.2)$", fontsize=20)
    elif model == "rmf_crust":
        plt.title("EoS: FPS crust", fontsize=20)
    else:
        plt.title(f"EoS: {model}", fontsize=20)

    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_mr_band_comparison(R, M, stable_mask):
    apply_style()

    R = np.asarray(R)
    M = np.asarray(M)
    stable_mask = np.asarray(stable_mask, dtype=bool)
    unstable_mask = ~stable_mask

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.plot(
        R[stable_mask],
        M[stable_mask],
        "k-",
        lw=2,
        label=r"$npe\mu$ core (stable)",
    )

    ax.plot(
        R[unstable_mask],
        M[unstable_mask],
        "k--",
        lw=2,
        label=r"$npe\mu$ core (unstable)",
    )

    # --- NICER J0030 ---
    M_0030 = 1.34
    M_0030_low, M_0030_high = 0.16, 0.15
    R_0030 = 12.71
    R_0030_low, R_0030_high = 1.19, 1.14

    # --- NICER J0740 ---
    M_0740 = 2.08
    M_0740_low = M_0740_high = 0.07
    R_0740 = 12.49
    R_0740_low, R_0740_high = 0.88, 1.28

    # --- Massive pulsars ---
    M_0348, dM_0348 = 2.01, 0.04
    M_1614, dM_1614 = 1.97, 0.04
    M_2215, dM_2215 = 2.27, 0.16
    M_0952, dM_0952 = 2.35, 0.17

    R_min, R_max = 8.1, 14.0
    M_min, M_max = 1.0, 2.6

    ax.set_xlim(R_min, R_max)
    ax.set_ylim(M_min, M_max)

    # Helper
    yband = lambda M, dM: (M - dM, M + dM)

    # Horizontal mass bands
    ax.fill_betweenx(
        yband(M_0348, dM_0348),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J0348+0432",
    )
    ax.fill_betweenx(
        yband(M_1614, dM_1614),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J1614−2230",
    )
    ax.fill_betweenx(
        yband(M_2215, dM_2215),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J2215+5135",
    )
    ax.fill_betweenx(
        yband(M_0952, dM_0952),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J0952−0607",
    )

    # NICER error bars
    ax.errorbar(
        R_0030,
        M_0030,
        xerr=[[R_0030_low], [R_0030_high]],
        yerr=[[M_0030_low], [M_0030_high]],
        fmt="o",
        color="tab:orange",
        label="PSR J0030+0451",
    )
    ax.errorbar(
        R_0740,
        M_0740,
        xerr=[[R_0740_low], [R_0740_high]],
        yerr=[[M_0740_low], [M_0740_high]],
        fmt="o",
        color="tab:blue",
        label="PSR J0740+6620",
    )

    ax.set_xlabel(r"$R\,[\mathrm{km}]$", fontsize=18)
    ax.set_ylabel(r"$M/M_\odot$", fontsize=18)
    ax.set_title("Mass–radius relation with observational constraints", fontsize=20)

    ax.grid(True, which="both", alpha=0.5)
    ax.legend(loc="lower left", fontsize="small")

    fig.tight_layout()
    plt.show()


def make_all_plots(res, model, band_eos_file=None):
    """
    Generate all plots for the npemu model.

    Parameters
    ----------
    res : dict
        Dictionary returned by tov.run_model().
    model : str
        Used to give plots titles
    band_eos_file : str or None
        Path to EoS band h5 file, or None to skip that plot.
    """

    eos_main = res["eos_main"]
    eos_uniform = res["eos_uniform"]
    R_main = res["R_main"]
    M_main = res["M_main"]
    stable_main = res["stable_main"]
    unstable_main = res["unstable_main"]
    ec_main = res["ec_main"]
    M_ec_main = res["M_ec_main"]
    comp = res["comp"]

    plot_composition(comp)

    plot_eos_comparison(eos_main, eos_uniform, model)

    plot_tov_results(
        R_main,
        M_main,
        stable_main,
        unstable_main,
        ec_main,
        M_ec_main,
        model
    )

    if band_eos_file is not None:
        plot_eos_band_comparison(eos_uniform, band_eos_file)

    plot_mr_band_comparison(R_main, M_main, stable_main)


def main():
    # model = "polytrope"
    model = "rmf_crust"
    res = run_model(model)

    make_all_plots(
        res,
        model,
        band_eos_file="numerics/npemu/bands.h5",
    )


if __name__ == "__main__":
    main()
