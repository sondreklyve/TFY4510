import numpy as np
import scipy
import scipy.constants as sc
import scipy.optimize as op
from scipy import interpolate
import matplotlib.pyplot as plt

# ----------------------------
# Constants and unit helpers
# ----------------------------
G = sc.G
hbar = sc.hbar
Mo = 1.9891e30  # solar mass in kg
R0 = G * Mo / (1000.0 * sc.c**2.0)  # length scale factor in km

MeVtoJoules = 1.0e6 * sc.eV
JoulestoMeV = 1.0 / MeVtoJoules

# Momentum unit: 1/m in MeV (ℏ=c=1)
MeVtoperm = MeVtoJoules / (sc.c * hbar)
permtoMeV = 1.0 / MeVtoperm

# Energy density conversions (handy for plotting)
MeV4togcm3 = 1000.0 * MeVtoJoules / sc.c**2.0 * (MeVtoperm / 100.0) ** 3.0
MeV4toJouleskm3 = MeVtoJoules * (1000.0 * MeVtoperm) ** 3.0
MeV4tokgkm3 = MeV4toJouleskm3 / sc.c**2.0
dynetoMeV4 = 0.1 * JoulestoMeV * (permtoMeV) ** 3.0
gcm3toMeV4 = 1.0 / MeV4togcm3

# Particle masses (MeV)
mn = 938.0  # mass entering sigma self-interactions (as in chapter)
m = (939.5654133 + 938.2720813) / 2.0  # average nucleon mass
me = 0.5109989461
mmu = 105.6583745

# Density normalization ρ0 ~ 1 fm^-3 in MeV units
rho0 = (1.0e15 * permtoMeV) ** 3.0
e0 = m ** 4.0

# Scaled mass density prefactor for TOV: solar masses per km^3
beta = 4.0 * np.pi * e0 * MeV4tokgkm3 / Mo

# ----------------------------
# RMF parameters (σ-ω-ρ) with σ self-interactions b,c
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
    return np.sign(x) * (abs(x) ** (1.0 / 3.0))

def gsigmaintegral(fgsigma, k):
    """
    Integral in the σ-field equation.
    Returns normalized (g_sigma/m)^3 contribution.
    """
    xnorm = k / (1.0 - fgsigma)
    return (1.0 - fgsigma) ** 3.0 * (
        xnorm * np.sqrt(xnorm ** 2.0 + 1.0) - np.arcsinh(xnorm)
    ) / (2.0 * np.pi ** 2.0)

# ----------------------------
# Nonlinear systems (before / after muons)
# Unknowns x = [rho_n, g_sigma, k_e]
# ----------------------------
def squaresbeforemuons(x, rho):
    kp = cube(3.0 * np.pi ** 2.0 * (rho - x[0]))  # proton Fermi momentum
    kn = cube(3.0 * np.pi ** 2.0 * x[0])          # neutron Fermi momentum
    grho = gmrho2 * (0.5 * rho - x[0])
    mstar = (1.0 - x[1] / m) * m

    term_sigma = x[1] - gmsigma2 * (
        -b * mn * x[1] ** 2.0
        - c * x[1] ** 3.0
        + m ** 3.0 * (gsigmaintegral(x[1] / m, kn / m) + gsigmaintegral(x[1] / m, kp / m))
    )
    term_ke = kp - x[2]  # charge neutrality (no muons)
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
    kmu = np.sqrt(max(mue2 - mmu ** 2.0, 0.0))  # protect against tiny negatives

    term_sigma = x[1] - gmsigma2 * (
        -b * mn * x[1] ** 2.0
        - c * x[1] ** 3.0
        + m ** 3.0 * (gsigmaintegral(x[1] / m, kn / m) + gsigmaintegral(x[1] / m, kp / m))
    )
    term_ke = kp - cube(x[2] ** 3.0 + kmu ** 3.0)  # charge neutrality with muons
    term_beta = grho + np.sqrt(kp ** 2.0 + mstar ** 2.0) - np.sqrt(kn ** 2.0 + mstar ** 2.0) + np.sqrt(mue2)
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0

# ----------------------------
# EoS integrals (fermion gas)
# ----------------------------
def pressureintegral(x):
    return ((2.0 * x ** 3.0 - 3.0 * x) * np.sqrt(1.0 + x ** 2.0) + 3.0 * np.arcsinh(x)) / 8.0

def energyintegral(x):
    return ((2.0 * x ** 3.0 + x) * np.sqrt(1.0 + x ** 2.0) - np.arcsinh(x)) / 8.0

def makeEoS(rholist, rhonlist, gsigmalist, kelist, kmulist):
    """Return (energy_density_list, pressure_list) in units of e0."""
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
    # TOV pressure gradient
    return -(R0 * eps * M) / r ** 2.0 * (P / eps + 1.0) * ((beta * r ** 3.0 * P) / M + 1.0) / (1.0 - 2.0 * R0 * M / r)

# ----------------------------
# Solve composition over a density grid (uniform npeμ everywhere)
# ----------------------------
bnds = ((0, None), (0, None), (0, None))  # rho_n, g_sigma, k_e >= 0

n = 2000
rho_min = 0.005 * rho0      # push lower to extend EoS to smaller P (still RMF)
rho_max = 2.0 * rho0
rhos = np.linspace(rho_min, rho_max, n)

# Initial guess for [rho_n, g_sigma, k_e]
guess = np.array([rho_min * 0.9, gmsigma2 * rho_min, 0.12 * m * (rho_min / rho0) ** (2.0 / 3.0)])

rhons = np.empty_like(rhos)
gsigmas = np.empty_like(rhos)
kes = np.empty_like(rhos)
kmuons = np.empty_like(rhos)

firsttimemuons = False
for i, rho in enumerate(rhos):
    # detect muon threshold using previous guess
    if (guess[2] ** 2.0 + me ** 2.0) >= mmu ** 2.0 and not firsttimemuons:
        firsttimemuons = True

    if not firsttimemuons:
        roots = op.minimize(squaresbeforemuons, guess, (rho,), bounds=bnds, method="L-BFGS-B")
    else:
        roots = op.minimize(squaresaftermuons, guess, (rho,), bounds=bnds, method="L-BFGS-B")

    guess = roots.x
    rhons[i] = guess[0]
    gsigmas[i] = guess[1]
    kes[i] = guess[2]
    if firsttimemuons:
        km = np.sqrt(max(guess[2] ** 2.0 + me ** 2.0 - mmu ** 2.0, 0.0))
        kmuons[i] = km
    else:
        kmuons[i] = 0.0

# ----------------------------
# Build uniform RMF EoS only (no crust)
# ----------------------------
energy, pressure = makeEoS(rhos, rhons, gsigmas, kes, kmuons)

# Clean monotonicity for inversion ε(P) → used by TOV
# Sort by pressure and drop non-increasing tails if needed
order = np.argsort(pressure)
P_sorted = pressure[order]
E_sorted = energy[order]

# Enforce strict monotonic increase for interp1d (remove duplicates)
mask = np.diff(P_sorted, prepend=P_sorted[0] - 1e99) > 0
P_mono = P_sorted[mask]
E_mono = E_sorted[mask]

# Interpolators for forward/backward use
E_of_P = interpolate.interp1d(P_mono, E_mono, kind="linear", fill_value="extrapolate", assume_sorted=True)

# ----------------------------
# Plots: composition and EoS from uniform model
# ----------------------------
rhop_over_rho = 1.0 - rhons / rhos
rhosnorm = rhos / rho0

plt.figure()
plt.ylabel(r'$\log_{10}(\rho_i/\rho)$')
plt.xlabel(r'$\rho\;(\mathrm{fm}^{-3})$')
plt.ylim(-3, 0)
plt.xlim(0, 1)
plt.plot(rhosnorm, np.log10(np.clip(rhons / rhos, 1e-30, 1.0)), 'b', label='n')
plt.plot(rhosnorm, np.log10(np.clip(rhop_over_rho, 1e-30, 1.0)), 'r', label='p')

rho_e_over_rho = kes**3 / (3.0 * np.pi**2 * rhos)
rho_mu_over_rho = np.where(kmuons > 0, kmuons**3 / (3.0 * np.pi**2 * rhos), 0.0)
plt.plot(rhosnorm, np.log10(np.clip(rho_e_over_rho, 1e-30, 1.0)), 'b--', label='e')
plt.plot(rhosnorm, np.log10(np.clip(rho_mu_over_rho, 1e-30, 1.0)), 'k--', label=r'$\mu$')
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# EoS plot (uniform model only)
log_eps_gcm3 = np.log10(E_mono * e0 * MeV4togcm3)
log_P_dyne = np.log10(P_mono * e0 / dynetoMeV4)
plt.figure()
plt.ylabel(r'$\log_{10} P\;(\mathrm{dyne/cm}^2)$')
plt.xlabel(r'$\log_{10} \epsilon\;(\mathrm{g/cm}^3)$')
plt.plot(log_eps_gcm3, log_P_dyne, 'k')
plt.grid(True); plt.tight_layout(); plt.show()

# ----------------------------
# TOV integration (uniform npeμ all the way)
# ----------------------------
# We'll integrate outward until P drops to the minimum pressure available from the model.
P_surface = float(P_mono[0])          # lowest pressure from uniform RMF
P_max = float(P_mono[-1])             # highest pressure available
Pc_start = min(4e33 * dynetoMeV4 / e0, 0.8 * P_max)  # safe start inside domain
Pc_end = 0.98 * P_max                 # avoid extrapolation at the top
Pc_step = 1.05

Mlist, Rlist, Pc_list = [], [], []

rstep = 0.001  # km
Pc = Pc_start
while Pc < Pc_end:
    P = Pc
    M = 0.0
    r = 1e-6  # avoid r=0 in denominators
    while P > P_surface and r < 100.0:
        eps = float(E_of_P(P))
        # simple Euler; small rstep keeps errors modest. Replace with RK if desired.
        M += rstep * dMdr(r, eps)
        P += rstep * dPdr(r, M, P, eps)
        r += rstep
    Mlist.append(M)
    Rlist.append(r)
    Pc_list.append(Pc)
    Pc *= Pc_step

# Split at turning point (max M)
idx_max = int(np.argmax(Mlist))
StableM = Mlist[: idx_max + 1]
StableR = Rlist[: idx_max + 1]
UnstableM = Mlist[idx_max + 1 :]
UnstableR = Rlist[idx_max + 1 :]

print("Radius at maximum mass (km):", Rlist[idx_max])
print("Maximum mass (M_sun):", Mlist[idx_max])

# Plot M–R and M–ε_c (ε from E(Pc))
central_dens = [float(E_of_P(Pc)) * e0 * MeV4togcm3 for Pc in Pc_list]
cd_log = np.log10(central_dens)
Stable_cd_log = cd_log[: idx_max + 1]
Unstable_cd_log = cd_log[idx_max + 1 :]

fig, axarr = plt.subplots(1, 2, sharey=True, figsize=(9, 4))
axarr[0].grid(True)
axarr[0].plot(StableR, StableM, 'k-', label='stable')
axarr[0].plot(UnstableR, UnstableM, 'k--', label='unstable')
axarr[0].set_xlabel('Radius (km)')
axarr[0].set_ylabel(r'$M/M_\odot$')
axarr[0].legend()

axarr[1].grid(True)
axarr[1].plot(Stable_cd_log, StableM, 'k-')
axarr[1].plot(Unstable_cd_log, UnstableM, 'k--')
axarr[1].set_xlabel(r'$\log_{10}\,\epsilon_c\;(\mathrm{g/cm}^3)$')

plt.tight_layout(); plt.show()
