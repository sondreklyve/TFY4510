import numpy as np
from scipy import interpolate, optimize

from core import e0, dynetoMeV4, MeV4togcm3, gcm3toMeV4

# ----------------------------
# Uniform npeμ model (no crust)
# ----------------------------
def build_eos_uniform(energy, pressure):
    """
    EoS epsilon(P) using ONLY the npeμ core EoS.
    energy, pressure are dimensionless (in units of e0).
    """
    eps_core = np.asarray(energy)
    P_core = np.asarray(pressure)

    idx = np.argsort(P_core)
    P_sorted = P_core[idx]
    eps_sorted = eps_core[idx]

    unique_mask = np.concatenate(([True], np.diff(P_sorted) > 0.0))
    P_unique = P_sorted[unique_mask]
    eps_unique = eps_sorted[unique_mask]

    EoS = interpolate.interp1d(
        P_unique,
        eps_unique,
        kind="cubic",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    return dict(
        EoS=EoS,
        joined_energy=eps_core,
        joined_pressure=P_core,
        eps_core=eps_core,
        P_core=P_core,
        eps_crust=None,
        P_crust=None,
    )


# ----------------------------
# Crust EoS fit (Lattimer–Swesty-like logistic joins)
# ----------------------------
def f0(x):
    return 1.0 / (np.exp(x) + 1.0)


# Coefficients 'a' from the original code
a = [
    6.22, 6.121, 0.005925, 0.16326, 6.48,
    11.4971, 19.105, 0.8938, 6.54,
    11.4950, -22.775, 1.5707, 4.3,
    14.08, 27.80, -1.653, 1.50, 14.67,
]


def PfromepsLD(log10_eps_gcm3):
    """
    Crust: log10 P(dyne/cm^2) as a function of log10 epsilon(g/cm^3).
    Input/Output are logarithmic (fits are given that way).
    """
    eps = log10_eps_gcm3
    return (
        (a[0] + a[1] * eps + a[2] * eps ** 3.0)
        * f0(a[4] * a[5] * (eps / a[5] - 1.0))
        / (1.0 + a[3] * eps)
        + (a[6] + a[7] * eps) * f0(a[8] * a[9] * (1.0 - eps / a[9]))
        + (a[10] + a[11] * eps) * f0(a[12] * a[13] * (1.0 - eps / a[13]))
        + (a[14] + a[15] * eps) * f0(a[16] * a[17] * (1.0 - eps / a[17]))
    )


def epsLDroot(log10_eps, log10P):
    return PfromepsLD(log10_eps) - log10P


def EoSLD(P, guess):
    """
    Invert crust relation to get epsilon(P).
    Input P is dimensionless (in units of e0), returns epsilon/e0.
    """
    log10P_dyne = np.log10(P * e0 / dynetoMeV4)
    log10eps = optimize.newton(epsLDroot, guess, args=(log10P_dyne,))
    return 10.0 ** log10eps * gcm3toMeV4 / e0


def build_eos_rmf_plus_crust(energy, pressure, n_ld_points: int = 1000):
    """
    Original RMF core + crust fit joined EoS.
    energy, pressure are dimensionless (in units of e0).

    Returns
    -------
    data : dict with keys
        'EoS'            : epsilon(P) interpolator (dimensionless)
        'joined_energy'  : full joined epsilon array
        'joined_pressure': full joined P array
        'eps_core','P_core'   : RMF core branch
        'eps_crust','P_crust' : crust branch
    """
    energy = np.asarray(energy)
    pressure = np.asarray(pressure)

    energy_gcm3_log10 = np.log10(energy * e0 * MeV4togcm3)
    pressure_dyne_log10 = np.log10(pressure * e0 / dynetoMeV4)

    # Find matching point where RMF P(ε) crosses crust P(ε)
    counter = 0
    while (
        counter < len(energy_gcm3_log10) - 1
        and PfromepsLD(energy_gcm3_log10[counter]) > pressure_dyne_log10[counter]
    ):
        counter += 1

    crossenergy = energy[counter]

    # Low-density branch (crust), tabulated in ε then mapped to P
    eps_crust = np.linspace(1.0e-13, crossenergy, n_ld_points)
    energyLD_log10_gcm3 = np.log10(eps_crust * e0 * MeV4togcm3)
    P_crust_log10 = [PfromepsLD(x) for x in energyLD_log10_gcm3]
    P_crust = np.array([10.0 ** p * dynetoMeV4 / e0 for p in P_crust_log10])

    eps_core = energy[counter:]
    P_core = pressure[counter:]

    joined_energy = np.concatenate([eps_crust, eps_core])
    joined_pressure = np.concatenate([P_crust, P_core])

    EoS = interpolate.interp1d(
        joined_pressure,
        joined_energy,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=False,
    )

    return dict(
        EoS=EoS,
        joined_energy=joined_energy,
        joined_pressure=joined_pressure,
        eps_core=eps_core,
        P_core=P_core,
        eps_crust=eps_crust,
        P_crust=P_crust,
    )


def build_eos_polytrope_crust(energy, pressure, gamma_crust=1.2):
    """
    EoS with RMF npeμ core and a *single polytropic crust* with
    user-controlled gamma_crust.

    Steps:
      - build the full RMF+tabulated crust EoS
      - read off the crust energy/pressure arrays and the core arrays
      - replace the tabulated crust by P = K * eps^gamma_crust, with K
        chosen so that P matches at the top of the crust
      - join polytropic crust + npeμ core and build epsilon(P) interpolator
    """
    # Get reference RMF+crust model
    full = build_eos_rmf_plus_crust(energy, pressure)

    eps_core = np.asarray(full["eps_core"])
    P_core = np.asarray(full["P_core"])
    eps_crust_tab = np.asarray(full["eps_crust"])
    P_crust_tab = np.asarray(full["P_crust"])

    # Match point: top of the crust table
    eps_match = eps_crust_tab[-1]
    P_match = P_crust_tab[-1]

    # Choose stiffness of the polytropic crust
    Gamma = gamma_crust

    # Constant K fixed by continuity at eps_match
    K = P_match / (eps_match ** Gamma)

    # Polytropic crust on the same epsilon grid as the tabulated crust
    eps_crust_poly = eps_crust_tab.copy()
    P_crust_poly = K * eps_crust_poly ** Gamma

    # Join polytropic crust + npeμ core
    joined_energy = np.concatenate([eps_crust_poly, eps_core])
    joined_pressure = np.concatenate([P_crust_poly, P_core])

    # epsilon(P) interpolator for TOV
    idxP = np.argsort(joined_pressure)
    P_sorted = joined_pressure[idxP]
    eps_sorted = joined_energy[idxP]

    # Ensure strictly increasing P
    unique = np.concatenate(([True], np.diff(P_sorted) > 0.0))
    P_unique = P_sorted[unique]
    eps_unique = eps_sorted[unique]

    EoS = interpolate.interp1d(
        P_unique,
        eps_unique,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    return dict(
        EoS=EoS,
        joined_energy=joined_energy,
        joined_pressure=joined_pressure,
        eps_core=eps_core,
        P_core=P_core,
        eps_crust=eps_crust_poly,
        P_crust=P_crust_poly,
    )
