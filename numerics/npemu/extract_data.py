import h5py
import numpy as np
import matplotlib.pyplot as plt

c = 2.99792458e10  # cm/s

def load_band(fname, rho_grid, low=2.5, high=97.5):
    """Load an ensemble EoS file and return low/high percentile bands."""
    with h5py.File(fname, "r") as f:
        eos_group = f["eos"]
        pressures = []

        for eos_id in eos_group.keys():
            rho = eos_group[eos_id]["energy_densityc2"][:]  # ε/c^2 = ρ [g/cm^3]
            p   = eos_group[eos_id]["pressurec2"][:]        # P/c^2
            p   = p * c**2                                  # → dyn/cm^2

            order = np.argsort(rho)
            rho = rho[order]
            p   = p[order]

            p_grid = np.interp(rho_grid, rho, p, left=np.nan, right=np.nan)
            pressures.append(p_grid)

    pressures = np.array(pressures)
    lower = np.nanpercentile(pressures, low, axis=0)
    upper = np.nanpercentile(pressures, high, axis=0)
    return lower, upper


def load_mr_band(fname, M_grid, low=5.0, high=95.0):
    """
    Load an ensemble HDF5 file and return lower/upper percentile bands
    for the mass–radius relation R(M).
    """

    with h5py.File(fname, "r") as f:
        ns_group = f["ns"]
        ids = list(ns_group.keys())

        R_all = []

        for eos_id in ids:
            ds = ns_group[eos_id]  # structured dataset

            M = ds["M"][()]   # Msun
            R = ds["R"][()]   # km

            # Sort by mass and enforce increasing branch (up to max mass)
            order = np.argsort(M)
            M = M[order]
            R = R[order]

            # Interpolate radius on the common mass grid
            R_grid = np.interp(M_grid, M, R, left=np.nan, right=np.nan)
            R_all.append(R_grid)

    R_all = np.array(R_all)  # shape: (N_eos, len(M_grid))

    R_low  = np.nanpercentile(R_all, low,  axis=0)
    R_high = np.nanpercentile(R_all, high, axis=0)

    return R_low, R_high


def plot_bands():
    # shared density grid
    rho_grid = np.logspace(13, 16, 400)   # 1e13–1e16 g/cm^3

    # green-constrained band
    green_file = "numerics/neutron_star_bands/NLSLTR_green.h5"
    lower_green, upper_green = load_band(green_file, rho_grid)

    # full band
    full_file = "numerics/neutron_star_bands/NLSLTR_full.h5"
    lower_full, upper_full = load_band(full_file, rho_grid)

    plt.figure(figsize=(6, 4))

    # green band (shaded)
    plt.fill_between(
        rho_grid, lower_green, upper_green,
        color="green", alpha=0.3,
        label="NLSLTR (green subset)"
    )

    # full band (purple dotted envelope)
    plt.plot(
        rho_grid, lower_full,
        color="purple", linestyle="--",
        label="NLSLTR (full, 5–95%)"
    )
    plt.plot(
        rho_grid, upper_full,
        color="purple", linestyle="--"
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.xlim((4*1e13, 2.7*1e15))
    plt.ylim((1e32, 1e37))
    plt.xlabel(r"$\rho$ [g/cm$^3$]")
    plt.ylabel(r"$P$ [dyn/cm$^2$]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_mr_bands():
    # Shared mass grid in Msun
    M_grid = np.linspace(0.5, 2.4, 300)

    # Green-constrained band (PSR+GW+XRay)
    green_file = "numerics/neutron_star_bands/NLSLTR_green.h5"
    R_lower_green, R_upper_green = load_mr_band(green_file, M_grid)

    # Full-astro band (full constraints)
    full_file = "numerics/neutron_star_bands/NLSLTR_full.h5"
    R_lower_full, R_upper_full = load_mr_band(full_file, M_grid)

    plt.figure(figsize=(6, 4))

    # Green band (shaded) — fill horizontally between radii
    plt.fill_betweenx(
        M_grid,                # y
        R_lower_green,         # x-left
        R_upper_green,         # x-right
        color="green", alpha=0.3,
        label="NLSLTR MR (green subset)"
    )

    # Full band (purple dotted envelope)
    plt.plot(
        R_lower_full, M_grid,
        color="purple", linestyle="--",
        label="NLSLTR MR (full, 2.5–97.5%)"
    )
    plt.plot(
        R_upper_full, M_grid,
        color="purple", linestyle="--"
    )

    # Axes labels
    plt.xlabel(r"$R\,[\mathrm{km}]$")
    plt.ylabel(r"$M\,[M_\odot]$")

    # Typical MR limits (tweak freely)
    plt.xlim((5.5, 16.5))
    plt.ylim((1.0, 2.5))

    plt.grid(True, which="both", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
