import numpy as np
import matplotlib.pyplot as plt
from eos_models import build_eos_uniform, build_eos_rmf_plus_crust
from core import solve_composition, make_eos, e0, MeV4togcm3, dynetoMeV4
from extract_data import load_band, load_mr_band
from run_npemu import run_tov

def plot_npemu_vs_ng_band(
    eos_dict,
    eps0_to_rho_cgs,
    P0_to_dyncm2,
    rho_band,
    P_lower,
    P_upper,
    label_npemu="npemu EoS",
    color_npemu="k",
    ax=None,
):
    """
    Plot an npemu EoS (from build_eos_uniform) against the Ng et al. pressure–density band.

    Parameters
    ----------
    eos_dict : dict
        Output of build_eos_uniform(energy, pressure). Uses:
        - 'eps_core' : np.array of energy densities (dimensionless, in units of eps0)
        - 'P_core'   : np.array of pressures (dimensionless, in units of P0)
    eps0_to_rho_cgs : float
        Conversion factor from dimensionless energy to rho = epsilon/c^2 in g/cm^3.
        i.e. rho_cgs = eps_dimless * eps0_to_rho_cgs
    P0_to_dyncm2 : float
        Conversion factor from dimensionless pressure to P in dyn/cm^2.
        i.e. P_cgs = P_dimless * P0_to_dyncm2
    rho_band : np.ndarray
        Density grid used for the Ng et al. band (g/cm^3).
    P_lower, P_upper : np.ndarray
        5th and 95th percentile pressures on rho_band (dyn/cm^2).
    label_npemu : str
        Legend label for the npemu curve.
    color_npemu : str
        Matplotlib color for the npemu curve.
    ax : matplotlib.axes.Axes or None
        If given, plot into this axes; otherwise create a new figure.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the plot.
    """
    eps_core = np.asarray(eos_dict["eps_core"])
    P_core   = np.asarray(eos_dict["P_core"])

    # Convert npemu dimensionless units to cgs
    rho_npemu = eps_core * eps0_to_rho_cgs      # g/cm^3 (epsilon/c^2)
    P_npemu   = P_core   * P0_to_dyncm2         # dyn/cm^2
    log_rho = np.log10(rho_npemu)
    log_P   = np.log10(P_npemu)

    # Sort npemu EoS by density to get a nice monotonic curve
    order = np.argsort(rho_npemu)
    rho_npemu = rho_npemu[order]
    P_npemu   = P_npemu[order]
    log_rho_band  = np.log10(rho_band)
    log_P_lower   = np.log10(P_lower)
    log_P_upper   = np.log10(P_upper)


    # Create axes if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_title("npemu EoS vs Ng et al. posterior band")

    ax.plot(
        log_rho,
        log_P,
        color=color_npemu,
        lw=2,
        label=label_npemu,
    )
    ax.fill_between(
        log_rho_band,
        log_P_lower,
        log_P_upper,
        color="green",
        alpha=0.3,
        label="PSR+GW+X-Ray 90% band"
    )

    ax.set_xlabel(r"$\log_{10}\rho\,[{\rm g\,cm^{-3}}]$")
    ax.set_ylabel(r"$\log_{10}P\,[{\rm dyn\,cm^{-2}}]$")

    ax.set_xlim((13.58, 15.55))
    ax.set_ylim((32, 37))

    ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()

    return ax


def plot_npemu_mr():
    """
    Build the npeμ+crust EoS, run the TOV solver, and plot the resulting
    mass–radius relation together with selected observational constraints:

      - NICER radius measurements for PSR J0030+0451 and PSR J0740+6620
        shown as error bars in the M–R plane.
      - Mass measurements for PSR J0348+0432, J1614−2230, J2215+5135,
        and J0952−0607 shown as horizontal shaded bands (±1σ).

    Radius on x-axis (km), mass on y-axis (M_sun).
    """

    # ------------------------------------------------------------------
    # 1) Composition and core EoS (same as in run_model)
    # ------------------------------------------------------------------
    comp = solve_composition()
    rhos    = comp["rhos"]
    rhons   = comp["rhons"]
    gsigmas = comp["gsigmas"]
    kes     = comp["kes"]
    kmuons  = comp["kmuons"]

    energy, pressure = make_eos(rhos, rhons, gsigmas, kes, kmuons)

    # npeμ core + crust
    data = build_eos_rmf_plus_crust(energy, pressure)
    EoS = data["EoS"]
    joined_pressure = data["joined_pressure"]

    # ------------------------------------------------------------------
    # 2) TOV integration (copied from your run_model logic)
    # ------------------------------------------------------------------
    Pcstart = 4e33 * dynetoMeV4 / e0   # dimensionless central pressure
    Pcend   = joined_pressure[-2]      # stop near highest tabulated pressure
    Pcstep  = 1.05
    tol     = joined_pressure[0]

    tov = run_tov(EoS, Pcstart, Pcend, Pcstep, tol)

    StableM    = tov["StableM"]
    StableR    = tov["StableR"]
    UnstableM  = tov["UnstableM"]
    UnstableR  = tov["UnstableR"]

    # ------------------------------------------------------------------
    # 3) Observational constraints (hard-coded numbers)
    # ------------------------------------------------------------------
    # NICER radius measurements (asymmetric errors)
    # PSR J0030+0451  (Riley/Miller 2019; choose one set consistently)
    M_0030 = 1.34
    M_0030_err_low  = 0.16
    M_0030_err_high = 0.15
    R_0030 = 12.71
    R_0030_err_low  = 1.19
    R_0030_err_high = 1.14

    # PSR J0740+6620 (Salmi/Dittmann 2024)
    M_0740 = 2.08
    M_0740_err_low  = 0.07
    M_0740_err_high = 0.07
    R_0740 = 12.49
    R_0740_err_low  = 0.88
    R_0740_err_high = 1.28

    # Massive pulsars: central mass ± 1σ, shown as horizontal bands
    M_0348, dM_0348 = 2.01, 0.04   # PSR J0348+0432
    M_1614, dM_1614 = 1.97, 0.04   # PSR J1614−2230
    M_2215, dM_2215 = 2.27, 0.16   # PSR J2215+5135
    M_0952, dM_0952 = 2.35, 0.17   # PSR J0952−0607

    # ------------------------------------------------------------------
    # 4) Plot: radius on x-axis, mass on y-axis
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))

    # Your MR curve
    ax.plot(StableR,   StableM,   "k-", label="npemu+crust (stable)")
    ax.plot(UnstableR, UnstableM, "k--", label="npemu+crust (unstable)")

    # Axis limits used also for horizontal bands
    R_min, R_max = 8.1, 14.0
    ax.set_xlim(R_min, R_max)
    ax.set_ylim(1.0, 2.6)

    # Horizontal mass bands (±1σ)
    y = lambda M, dM: np.linspace(M - dM, M + dM, 2)

    ax.fill_betweenx(
        y(M_0348, dM_0348),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J0348+0432",
    )
    ax.fill_betweenx(
        y(M_1614, dM_1614),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J1614−2230",
    )
    ax.fill_betweenx(
        y(M_2215, dM_2215),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J2215+5135",
    )
    ax.fill_betweenx(
        y(M_0952, dM_0952),
        R_min,
        R_max,
        alpha=0.3,
        label="PSR J0952−0607",
    )

    # NICER points with error bars
    ax.errorbar(
        R_0030,
        M_0030,
        xerr=[[R_0030_err_low], [R_0030_err_high]],
        yerr=[[M_0030_err_low], [M_0030_err_high]],
        fmt="o",
        label="PSR J0030+0451",
    )
    ax.errorbar(
        R_0740,
        M_0740,
        xerr=[[R_0740_err_low], [R_0740_err_high]],
        yerr=[[M_0740_err_low], [M_0740_err_high]],
        fmt="o",
        label="PSR J0740+6620",
    )

    ax.set_xlabel("Radius (km)")
    ax.set_ylabel(r"$M/M_\odot$")
    ax.set_title("Mass–radius relation with observational constraints")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower left", fontsize="small")
    fig.tight_layout()

    return fig, ax

if __name__ == "__main__":
    # Example call
    fig, ax = plot_npemu_mr()
    plt.show()

"""
if __name__ == '__main__':
    rho_grid = np.logspace(13, 16, 400)   # 1e13–1e16 g/cm^3
    rho_band  = rho_grid      # from Ng band extraction
    green_file = "numerics/neutron_star_bands/NLSLTR_green.h5"
    P_lower, P_upper = load_band(green_file, rho_grid, 5.0, 95.0)

    comp = solve_composition()
    rhos    = comp["rhos"]
    rhons   = comp["rhons"]
    gsigmas = comp["gsigmas"]
    kes     = comp["kes"]
    kmuons  = comp["kmuons"]

    energy, pressure = make_eos(rhos, rhons, gsigmas, kes, kmuons)
    eos_npemu = build_eos_uniform(energy, pressure)

    eps0_to_rho_cgs = e0 * MeV4togcm3
    P0_to_dyncm2    = e0 / dynetoMeV4

    ax = plot_npemu_vs_ng_band(
        eos_npemu,
        eps0_to_rho_cgs=eps0_to_rho_cgs,
        P0_to_dyncm2=P0_to_dyncm2,
        rho_band=rho_band,
        P_lower=P_lower,
        P_upper=P_upper,
    )
    plt.show()
"""