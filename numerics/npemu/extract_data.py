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


def plot_bands():
    # shared density grid
    rho_grid = np.logspace(13, 16, 400)   # 1e13–1e16 g/cm^3

    # green-constrained band
    green_file = "neutron_star_bands/NLSLTR_green.h5"
    lower_green, upper_green = load_band(green_file, rho_grid)

    # full band
    full_file = "neutron_star_bands/NLSLTR_full.h5"
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
