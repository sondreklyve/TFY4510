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
