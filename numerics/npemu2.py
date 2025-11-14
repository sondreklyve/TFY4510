
import numpy as np
import scipy
import scipy.constants as sc
import scipy.optimize as op
from scipy import interpolate
import matplotlib.pyplot as plt

# ----------------------------
# Constants and unit helpers (match npemu.py)
# ----------------------------
G = sc.G
hbar = sc.hbar
Mo = 1.9891e30  # solar mass in kg

MeVtoJoules = 1.0e6 * sc.eV
JoulestoMeV = 1.0 / MeVtoJoules

# Momentum unit: 1/m in MeV (ℏ=c=1 style conversions)
MeVtoperm = MeVtoJoules / (sc.c * hbar)
permtoMeV = 1.0 / MeVtoperm

# Energy density conversions
MeV4togcm3 = 1000.0 * MeVtoJoules / sc.c**2.0 * (MeVtoperm / 100.0) ** 3.0
MeV4toJoules_per_m3 = MeVtoJoules * (MeVtoperm) ** 3.0  # direct to SI J/m^3
dynetoMeV4 = 0.1 * JoulestoMeV * (permtoMeV) ** 3.0
gcm3toMeV4 = 1.0 / MeV4togcm3

# Particle masses (MeV)
mn = 938.0  # approximate mass used in sigma self-interactions
m = (939.5654133 + 938.2720813) / 2.0  # average nucleon mass
me = 0.5109989461
mmu = 105.6583745

# Density normalization rho0 ~ 1 fm^-3 in MeV units
rho0 = (1.0e15 * permtoMeV) ** 3.0
e0 = m ** 4.0  # energy density unit in MeV^4

# ----------------------------
# RMF parameters (σ-ω-ρ, with σ self-interactions b,c) — same as npemu.py
# ----------------------------
gmsigma2 = 9.927 * (1.0e-15 * MeVtoperm) ** 2.0
gmomega2 = 4.820 * (1.0e-15 * MeVtoperm) ** 2.0
gmrho2 = 4.791 * (1.0e-15 * MeVtoperm) ** 2.0
b = 0.008621
c = -0.002321

# ----------------------------
# Helpers
# ----------------------------
def cube(x):
    """Real cube root (handles negatives)."""
    return np.sign(x) * (np.abs(x) ** (1.0/3.0))

def gsigmaintegral(fgsigma, k):
    """
    Integral appearing in the σ-field equation.
    Returns normalized (g_sigma/m)^3 contribution.
    """
    xnorm = k / (1.0 - fgsigma + 1e-30)
    return (1.0 - fgsigma) ** 3.0 * (
        xnorm * np.sqrt(xnorm ** 2.0 + 1.0) - np.arcsinh(xnorm)
    ) / (2.0 * np.pi ** 2.0)

# ----------------------------
# Nonlinear system for composition before/after muon onset
# Unknowns x = [rho_n, g_sigma, k_e]
# ----------------------------
def squaresbeforemuons(x, rho):
    rho_n, gsig, ke = x
    kp = cube(3.0 * np.pi ** 2.0 * (rho - rho_n))  # proton Fermi momentum
    kn = cube(3.0 * np.pi ** 2.0 * rho_n)          # neutron Fermi momentum
    grho = gmrho2 * (0.5 * rho - rho_n)
    mstar = (1.0 - gsig / m) * m

    term_sigma = gsig - gmsigma2 * (
        -b * mn * gsig ** 2.0
        - c * gsig ** 3.0
        + m ** 3.0 * (gsigmaintegral(gsig / m, kn / m) + gsigmaintegral(gsig / m, kp / m))
    )
    term_ke = kp - ke
    term_beta = (
        grho
        + np.sqrt(kp ** 2.0 + mstar ** 2.0)
        - np.sqrt(kn ** 2.0 + mstar ** 2.0)
        + np.sqrt(ke ** 2.0 + me ** 2.0)
    )
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0

def squaresaftermuons(x, rho):
    rho_n, gsig, ke = x
    kp = cube(3.0 * np.pi ** 2.0 * (rho - rho_n))
    kn = cube(3.0 * np.pi ** 2.0 * rho_n)
    grho = gmrho2 * (0.5 * rho - rho_n)
    mstar = (1.0 - gsig / m) * m
    mue2 = me ** 2.0 + ke ** 2.0
    kmu = np.sqrt(max(mue2 - mmu ** 2.0, 0.0))

    term_sigma = gsig - gmsigma2 * (
        -b * mn * gsig ** 2.0
        - c * gsig ** 3.0
        + m ** 3.0 * (gsigmaintegral(gsig / m, kn / m) + gsigmaintegral(gsig / m, kp / m))
    )
    term_ke = kp - cube(ke ** 3.0 + kmu ** 3.0)  # charge neutrality with muons
    term_beta = grho + np.sqrt(kp ** 2.0 + mstar ** 2.0) - np.sqrt(kn ** 2.0 + mstar ** 2.0) + np.sqrt(mue2)
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0

# ----------------------------
# Fermion-gas integrals for EOS
# ----------------------------
def pressureintegral(x):
    return ((2.0 * x ** 3.0 - 3.0 * x) * np.sqrt(1.0 + x ** 2.0) + 3.0 * np.arcsinh(x)) / 8.0

def energyintegral(x):
    return ((2.0 * x ** 3.0 + x) * np.sqrt(1.0 + x ** 2.0) - np.arcsinh(x)) / 8.0

def makeEoS(rholist, rhonlist, gsigmalist, kelist, kmulist):
    """Return (energy_density_list, pressure_list) in units of e0 (dimensionless)."""
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
# Crust EoS fit (Lattimer–Swesty-like logistic joins)
# ----------------------------
def f0(x):
    return 1.0 / (np.exp(x) + 1.0)

# Coefficients from user code
a = [6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 6.54,
     11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50, 14.67]

def PfromepsLD(log10_eps_gcm3):
    eps = log10_eps_gcm3
    return (
        (a[0] + a[1] * eps + a[2] * eps ** 3.0) * f0(a[4] * a[5] * (eps / a[5] - 1.0)) / (1.0 + a[3] * eps)
        + (a[6] + a[7] * eps) * f0(a[8] * a[9] * (1.0 - eps / a[9]))
        + (a[10] + a[11] * eps) * f0(a[12] * a[13] * (1.0 - eps / a[13]))
        + (a[14] + a[15] * eps) * f0(a[16] * a[17] * (1.0 - eps / a[17]))
    )

# ----------------------------
# Build core composition grid and EOS (no plotting, no side effects)
# ----------------------------
def build_core_eos(npts=1200, rhomin_frac=0.01, rhomax_frac=2.0):
    bnds = ((0, None), (0, None), (0, None))  # rho_n, g_sigma, k_e >= 0

    startrho = rhomin_frac * rho0
    rhos = np.linspace(startrho, rhomax_frac * rho0, npts).tolist()

    # Initial guess for [rho_n, g_sigma, k_e]
    guess = np.array([startrho, gmsigma2 * startrho, 0.12 * m * (startrho / rho0) ** (2.0 / 3.0)])

    rhons = []
    gsigmas = []
    rhoesnorm = []
    kes = []
    kmuons = []

    muons_on = False

    for rho in rhos:
        if (guess[2] ** 2.0 + me ** 2.0) >= mmu ** 2.0 and not muons_on:
            muons_on = True
        if not muons_on:
            roots = op.minimize(squaresbeforemuons, guess, (rho,), bounds=bnds, method='L-BFGS-B')
        else:
            roots = op.minimize(squaresaftermuons, guess, (rho,), bounds=bnds, method='L-BFGS-B')

        guess = roots.x

        rhons.append(guess[0])
        gsigmas.append(guess[1])
        kes.append(guess[2])
        rhoesnorm.append(guess[2] ** 3.0 / (rho * 3.0 * np.pi ** 2.0))

        if muons_on:
            km = np.sqrt(max(guess[2] ** 2.0 + me ** 2.0 - mmu ** 2.0, 0.0))
            kmuons.append(km)
        else:
            kmuons.append(0.0)

    rhos = np.array(rhos)
    rhons = np.array(rhons)
    gsigmas = np.array(gsigmas)
    kes = np.array(kes)
    kmuons = np.array(kmuons)

    energy, pressure = makeEoS(rhos, rhons, gsigmas, kes, kmuons)  # dimensionless (units of e0)

    return rhos, energy, pressure

# ----------------------------
# Join crust + core in (epsilon, P), both dimensionless in units of e0
# ----------------------------
def build_joined_eos():
    rhos, energy, pressure = build_core_eos()

    energy_gcm3_log10 = np.log10(np.maximum(energy * e0 * MeV4togcm3, 1e-99))
    pressure_dyne_log10 = np.log10(np.maximum(pressure * e0 / dynetoMeV4, 1e-99))

    # Find matching point where RMF P(ε) crosses crust P(ε)
    idx = 0
    while idx < len(energy_gcm3_log10) - 1 and PfromepsLD(energy_gcm3_log10[idx]) > pressure_dyne_log10[idx]:
        idx += 1

    crossenergy = energy[idx]
    # Build low-density branch (crust)
    energyLD = np.linspace(1.0e-13, crossenergy, 1000)
    energyLD_log10_gcm3 = np.log10(np.maximum(energyLD * e0 * MeV4togcm3, 1e-99))
    pressureLD_log10 = np.array([PfromepsLD(x) for x in energyLD_log10_gcm3])
    pressureLD = (10.0 ** pressureLD_log10) * dynetoMeV4 / e0

    # Join: (epsilon, P), both dimensionless
    joined_energy = np.concatenate([energyLD, energy[idx:]])
    joined_pressure = np.concatenate([pressureLD, pressure[idx:]])

    # Monotonic sort by P for interpolation safety
    s = np.argsort(joined_pressure)
    joined_pressure = joined_pressure[s]
    joined_energy = joined_energy[s]

    return joined_pressure, joined_energy

# ----------------------------
# ε(P) adapters for TOV
# ----------------------------
class EpsOfP_SI:
    """Return ε(P) in SI [J/m^3] given P in SI [Pa], built from dimensionless arrays."""
    def __init__(self, Ps_dimless, Es_dimless):
        # store SI versions
        self.Ps = Ps_dimless * e0 * MeV4toJoules_per_m3
        self.Es = Es_dimless * e0 * MeV4toJoules_per_m3
        self.f = interpolate.interp1d(self.Ps, self.Es, kind='linear', bounds_error=False,
                                      fill_value=(self.Es[0], self.Es[-1]), assume_sorted=False)
    def __call__(self, P_SI):
        return float(self.f(P_SI))

class EpsOfP_Dimless:
    """Return ε(P) dimensionless (units of e0) given P in same dimensionless units."""
    def __init__(self, Ps_dimless, Es_dimless):
        self.Ps = Ps_dimless
        self.Es = Es_dimless
        self.f = interpolate.interp1d(self.Ps, self.Es, kind='linear', bounds_error=False,
                                      fill_value=(self.Es[0], self.Es[-1]), assume_sorted=False)
    def __call__(self, P_dimless):
        return float(self.f(P_dimless))

# ----------------------------
# Run with user's TOV solver
# ----------------------------
def run_with_tov(tov_expects_SI=True):
    Ps_dim, Es_dim = build_joined_eos()

    if tov_expects_SI:
        eps_of_P = EpsOfP_SI(Ps_dim, Es_dim)
        Pmin, Pmax = Ps_dim[5]*e0*MeV4toJoules_per_m3, Ps_dim[-5]*e0*MeV4toJoules_per_m3
    else:
        eps_of_P = EpsOfP_Dimless(Ps_dim, Es_dim)
        Pmin, Pmax = Ps_dim[5], Ps_dim[-5]

    from tov import massradiusplot  # import your solver

    Ps_out, Ms_out, Rs_out = massradiusplot(
        eps_of_P, (float(Pmin), float(Pmax)),
        tolD=5e-4, tolP=1e-6, maxdr=2e-3, Psurf=0.0, nmodes=0, newtonian=False, outfile=""
    )

    # Try to guess units for plotting
    Ms, Rs = np.array(Ms_out), np.array(Rs_out)

    # Guess mass units
    if np.nanmax(Ms) > 1e29:
        Ms_plot = Ms / Mo
        Mylabel = r"M [$M_\odot$]"
    else:
        Ms_plot = Ms
        Mylabel = r"M (solver units)"

    # Guess radius units
    if np.nanmax(Rs) > 1e4:  # >10 km in meters
        Rs_plot = Rs / 1000.0
        Rxlabel = "R [km]"
    else:
        Rs_plot = Rs
        Rxlabel = "R (solver units)"

    plt.figure(figsize=(5,4))
    plt.plot(Rs_plot, Ms_plot, '-k')
    plt.xlabel(Rxlabel)
    plt.ylabel(Mylabel)
    plt.title("npemu RMF EOS — Mass–Radius (joined crust+core)")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Try SI first. If the curve is clearly wrong in magnitude, set tov_expects_SI=False.
    run_with_tov(tov_expects_SI=False)
