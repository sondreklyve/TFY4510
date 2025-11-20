import numpy as np
import scipy.constants as sc
import scipy.optimize as op

# ----------------------------
# Constants and unit helpers
# ----------------------------
G = sc.G
hbar = sc.hbar
Mo = 1.9891e30  # solar mass in kg
R0 = G * Mo / (1000.0 * sc.c**2.0)  # length scale in km

MeVtoJoules = 1.0e6 * sc.eV
JoulestoMeV = 1.0 / MeVtoJoules

# Momentum unit: 1/m in MeV (ℏ=c=1 style conversions)
MeVtoperm = mtoperMeV = MeVtoJoules / (sc.c * hbar)
permtoMeV = perMeVtom = 1.0 / MeVtoperm

# Energy density conversions
MeV4togcm3 = 1000.0 * MeVtoJoules / sc.c**2.0 * (MeVtoperm / 100.0) ** 3.0
MeV4toJouleskm3 = MeVtoJoules * (1000.0 * MeVtoperm) ** 3.0
MeV4tokgkm3 = MeV4toJouleskm3 / sc.c**2.0
dynetoMeV4 = 0.1 * JoulestoMeV * (permtoMeV) ** 3.0
gcm3toMeV4 = 1.0 / MeV4togcm3

# Particle masses (MeV)
mn = 938.0  # approximate mass used in sigma self-interactions
m = (939.5654133 + 938.2720813) / 2.0  # average nucleon mass
me = 0.5109989461
mmu = 105.6583745

# Density normalization rho0 ~ 1 fm^-3 in MeV units
rho0 = (1.0e15 * permtoMeV) ** 3.0
e0 = m ** 4.0

# Scaled mass density prefactor (for TOV): solar masses per km^3
beta = 4.0 * np.pi * e0 * MeV4tokgkm3 / Mo

# ----------------------------
# RMF parameters (σ-ω-ρ, with σ self-interactions b,c)
# ----------------------------
gmsigma2 = 9.927 * (1.0e-15 * mtoperMeV) ** 2.0
gmomega2 = 4.820 * (1.0e-15 * mtoperMeV) ** 2.0
gmrho2 = 4.791 * (1.0e-15 * mtoperMeV) ** 2.0
b = 0.008621
c = -0.002321

# ----------------------------
# Simple helpers
# ----------------------------
def cube(x):
    """Real cube root (handles negatives)."""
    if x >= 0:
        return x ** (1.0 / 3.0)
    else:
        return -abs(x) ** (1.0 / 3.0)


def gsigmaintegral(fgsigma, k):
    """
    Integral appearing in the σ-field equation.
    Returns normalized (g_sigma/m)^3 contribution.
    """
    xnorm = k / (1.0 - fgsigma)
    return (1.0 - fgsigma) ** 3.0 * (
        xnorm * np.sqrt(xnorm ** 2.0 + 1.0) - np.arcsinh(xnorm)
    ) / (2.0 * np.pi ** 2.0)


# ----------------------------
# Nonlinear system for composition before/after muon onset
# Unknowns x = [rho_n, g_sigma, k_e]
# ----------------------------
_bnds = ((0, None), (0, None), (0, None))  # rho_n, g_sigma, k_e >= 0


def squaresbeforemuons(x, rho):
    kp = cube(3.0 * np.pi ** 2.0 * (rho - x[0]))  # proton Fermi momentum
    kn = cube(3.0 * np.pi ** 2.0 * x[0])          # neutron Fermi momentum
    grho = gmrho2 * (0.5 * rho - x[0])
    mstar = (1.0 - x[1] / m) * m

    term_sigma = x[1] - gmsigma2 * (
        -b * mn * x[1] ** 2.0
        - c * x[1] ** 3.0
        + m ** 3.0 * (
            gsigmaintegral(x[1] / m, kn / m)
            + gsigmaintegral(x[1] / m, kp / m)
        )
    )
    term_ke = kp - x[2]
    term_beta = (
        grho
        + np.sqrt(kp ** 2.0 + mstar ** 2.0)
        - np.sqrt(kn ** 2.0 + mstar ** 2.0)
        + np.sqrt(x[2] ** 2.0 + me ** 2.0)
    )
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0


def squaresaftermuons(x, rho):
    kp = cube(3.0 * np.pi ** 2.0 * (rho - x[0]))
    kn = cube(3.0 * np.pi ** 2.0 * x[0])
    grho = gmrho2 * (0.5 * rho - x[0])
    mstar = (1.0 - x[1] / m) * m
    mue2 = me ** 2.0 + x[2] ** 2.0
    kmu = np.sqrt(max(mue2 - mmu ** 2.0, 0.0))  # protect against small negatives

    term_sigma = x[1] - gmsigma2 * (
        -b * mn * x[1] ** 2.0
        - c * x[1] ** 3.0
        + m ** 3.0 * (
            gsigmaintegral(x[1] / m, kn / m)
            + gsigmaintegral(x[1] / m, kp / m)
        )
    )
    term_ke = kp - cube(x[2] ** 3.0 + kmu ** 3.0)  # charge neutrality with muons
    term_beta = (
        grho
        + np.sqrt(kp ** 2.0 + mstar ** 2.0)
        - np.sqrt(kn ** 2.0 + mstar ** 2.0)
        + np.sqrt(mue2)
    )
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0


def solve_composition(
    n_points: int = 2000,
    rho_min_factor: float = 0.01,
    rho_max_factor: float = 2.0,
):
    """
    Solve beta-equilibrated npeμ composition as function of baryon density.

    Returns
    -------
    result : dict
        Keys:
        - 'rhos'      : baryon density grid (MeV^3)
        - 'rhons'     : neutron densities
        - 'gsigmas'   : sigma mean-field (g_sigma * sigma)
        - 'kes'       : electron Fermi momenta
        - 'kmuons'    : muon Fermi momenta
        - 'rhonnorm'  : rho_n / rho
        - 'rhoesnorm' : rho_e / rho
        - 'rhomuons'  : muon densities
        - 'rhomuonsnorm' : rho_mu / rho
    """
    n = n_points
    startrho = rho_min_factor * rho0
    rhos = np.linspace(startrho, rho_max_factor * rho0, n)

    # Initial guess for [rho_n, g_sigma, k_e]
    guess = np.array([
        startrho,
        gmsigma2 * startrho,
        0.12 * m * (startrho / rho0) ** (2.0 / 3.0),
    ])

    rhons = []
    gsigmas = []
    rhonnorm = []
    rhomuons = []
    rhomuonsnorm = []
    rhoesnorm = []
    kes = []
    kmuons = []

    firsttimemuons = False

    for rho in rhos:
        if (guess[2] ** 2.0 + me ** 2.0) >= mmu ** 2.0 and not firsttimemuons:
            firsttimemuons = True

        if not firsttimemuons:
            roots = op.minimize(squaresbeforemuons, guess, (rho,), bounds=_bnds)
        else:
            roots = op.minimize(squaresaftermuons, guess, (rho,), bounds=_bnds)

        guess = roots.x

        rhons.append(guess[0])
        rhonnorm.append(guess[0] / rho)
        gsigmas.append(guess[1])
        kes.append(guess[2])
        rhoesnorm.append(guess[2] ** 3.0 / (rho * 3.0 * np.pi ** 2.0))

        if firsttimemuons:
            km = np.sqrt(max(guess[2] ** 2.0 + me ** 2.0 - mmu ** 2.0, 0.0))
            kmuons.append(km)
            rm = km ** 3.0 / (3.0 * np.pi ** 2.0)
            rhomuons.append(rm)
            rhomuonsnorm.append(rm / rho)
        else:
            kmuons.append(0.0)
            rhomuons.append(0.0)
            rhomuonsnorm.append(0.0)

    return dict(
        rhos=np.array(rhos),
        rhons=np.array(rhons),
        gsigmas=np.array(gsigmas),
        kes=np.array(kes),
        kmuons=np.array(kmuons),
        rhonnorm=np.array(rhonnorm),
        rhoesnorm=np.array(rhoesnorm),
        rhomuons=np.array(rhomuons),
        rhomuonsnorm=np.array(rhomuonsnorm),
    )


# ----------------------------
# EoS integrals (fermion gas)
# ----------------------------
def pressureintegral(x):
    return (
        (2.0 * x ** 3.0 - 3.0 * x) * np.sqrt(1.0 + x ** 2.0)
        + 3.0 * np.arcsinh(x)
    ) / 8.0


def energyintegral(x):
    return (
        (2.0 * x ** 3.0 + x) * np.sqrt(1.0 + x ** 2.0)
        - np.arcsinh(x)
    ) / 8.0


def make_eos(rholist, rhonlist, gsigmalist, kelist, kmulist):
    """
    Return (energy_density_list, pressure_list) in units of e0.
    """
    energydensitylist = []
    pressurelist = []
    for idx in range(len(rholist)):
        rho = rholist[idx]
        rhon = rhonlist[idx]
        gs = gsigmalist[idx]
        ke = kelist[idx]
        kmu = kmulist[idx]

        kp = cube(3.0 * np.pi ** 2.0 * (rho - rhon))
        kn = cube(3.0 * np.pi ** 2.0 * rhon)

        selfinteractions = b * mn * gs ** 3.0 / 3.0 + c * gs ** 4.0 / 4.0
        msigmaterm = 0.5 * gs ** 2.0 / gmsigma2
        momegaterm = 0.5 * gmomega2 * rho ** 2.0
        mrhoterm = 0.5 * gmrho2 * (0.5 * rho - rhon) ** 2.0

        pressureints = (
            (m - gs) ** 4.0
            * (pressureintegral(kp / (m - gs)) + pressureintegral(kn / (m - gs)))
            + me ** 4.0 * pressureintegral(ke / me)
            + mmu ** 4.0 * pressureintegral(max(kmu, 0.0) / mmu)
        ) / (3.0 * np.pi ** 2.0)

        energyints = (
            (m - gs) ** 4.0
            * (energyintegral(kp / (m - gs)) + energyintegral(kn / (m - gs)))
            + me ** 4.0 * energyintegral(ke / me)
            + mmu ** 4.0 * energyintegral(max(kmu, 0.0) / mmu)
        ) / (np.pi ** 2.0)

        currentenergydensity = selfinteractions + momegaterm + msigmaterm + mrhoterm + energyints
        currentpressure = -selfinteractions + momegaterm - msigmaterm + mrhoterm + pressureints

        energydensitylist.append(currentenergydensity / e0)
        pressurelist.append(currentpressure / e0)

    return np.array(energydensitylist), np.array(pressurelist)


# ----------------------------
# TOV right-hand sides
# ----------------------------
def dMdr(r, eps):
    # eps is dimensionless (in units of e0); beta carries units
    return beta * r ** 2.0 * eps  # solar masses per km


def dPdr(r, M, P, eps):
    # Pressure gradient in TOV equation
    return (
        -(R0 * eps * M) / r ** 2.0
        * (P / eps + 1.0)
        * ((beta * r ** 3.0 * P) / M + 1.0)
        / (1.0 - 2.0 * R0 * M / r)
    )
