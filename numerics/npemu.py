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
# Helpers
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
        + m ** 3.0 * (gsigmaintegral(x[1] / m, kn / m) + gsigmaintegral(x[1] / m, kp / m))
    )
    term_ke = kp - cube(x[2] ** 3.0 + kmu ** 3.0)  # charge neutrality with muons
    term_beta = grho + np.sqrt(kp ** 2.0 + mstar ** 2.0) - np.sqrt(kn ** 2.0 + mstar ** 2.0) + np.sqrt(mue2)
    return term_sigma ** 2.0 + term_ke ** 2.0 + term_beta ** 2.0


# ----------------------------
# TOV right-hand sides
# ----------------------------
def dMdr(r, eps):
    # eps is (dimensionless) energy density (in units of e0); beta carries units
    return beta * r ** 2.0 * eps  # solar masses per km


def dPdr(r, M, P, eps):
    # Pressure gradient in TOV (J/km^2 units carried through scalings)
    return -(R0 * eps * M) / r ** 2.0 * (P / eps + 1.0) * ((beta * r ** 3.0 * P) / M + 1.0) / (1.0 - 2.0 * R0 * M / r)


# ----------------------------
# Solve composition as function of baryon density
# ----------------------------
bnds = ((0, None), (0, None), (0, None))  # rho_n, g_sigma, k_e >= 0

n = 2000
startrho = 0.01 * rho0
rhos = np.linspace(startrho, 2.0 * rho0, n).tolist()

# Initial guess for [rho_n, g_sigma, k_e]
guess = np.array([startrho, gmsigma2 * startrho, 0.12 * m * (startrho / rho0) ** (2.0 / 3.0)])

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
    if (guess[2] ** 2.0 + me ** 2.0) >= mmu ** 2.0 and firsttimemuons is False:
        firsttimemuons = True

    if firsttimemuons is False:
        roots = op.minimize(squaresbeforemuons, guess, (rho,), bounds=bnds)
    else:
        roots = op.minimize(squaresaftermuons, guess, (rho,), bounds=bnds)

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

# ----------------------------
# Plot composition (Fig. 5.3)
# ----------------------------
rhopnorm = [1.0 - item for item in rhonnorm]
rhosnorm = [item / rho0 for item in rhos]
logrhopnorm = [np.log10(max(item, 1e-30)) for item in rhopnorm]
logrhonnorm = [np.log10(max(item, 1e-30)) for item in rhonnorm]
logrhoesnorm = [np.log10(max(item, 1e-30)) for item in rhoesnorm]
logrhomuonsnorm = [np.log10(max(item, 1e-30)) for item in rhomuonsnorm]

plt.ylabel(r'$\log (\rho_i/\rho)$')
plt.xlabel(r'$\rho\;(\mathrm{fm}^{-3})$')
plt.ylim(-3, 0)
plt.xlim(0, 1)
plt.plot(rhosnorm, logrhonnorm, 'b', label='n')
plt.plot(rhosnorm, logrhopnorm, 'r', label='p')
plt.plot(rhosnorm, logrhoesnorm, 'b--', label='e')
plt.plot(rhosnorm, logrhomuonsnorm, 'k--', label=r'$\mu$')
plt.grid(True)
plt.legend()
plt.show()

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

    return energydensitylist, pressurelist


# ----------------------------
# Crust EoS fit (Lattimer–Swesty-like logistic joins)
# ----------------------------
def f0(x):
    return 1.0 / (np.exp(x) + 1.0)


# Coefficients 'a' from the PDF
a = [6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 6.54,
     11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50, 14.67]


def PfromepsLD(log10_eps_gcm3):
    """
    Crust: log10 P(dyne/cm^2) as a function of log10 epsilon(g/cm^3).
    Input/Output are logarithmic (fits are given that way).
    """
    eps = log10_eps_gcm3
    return (
        (a[0] + a[1] * eps + a[2] * eps ** 3.0) * f0(a[4] * a[5] * (eps / a[5] - 1.0)) / (1.0 + a[3] * eps)
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
    log10eps = scipy.optimize.newton(epsLDroot, guess, args=(log10P_dyne,))
    return 10.0 ** log10eps * gcm3toMeV4 / e0


# ----------------------------
# Build the joined EoS (core RMF + crust fit)
# ----------------------------
energy, pressure = makeEoS(rhos, rhons, gsigmas, kes, kmuons)

energy_gcm3_log10 = [np.log10(item * e0 * MeV4togcm3) for item in energy]
pressure_dyne_log10 = [np.log10(item * e0 / dynetoMeV4) for item in pressure]

# Find matching point where RMF P(ε) crosses crust P(ε)
counter = 0
while counter < len(energy_gcm3_log10) - 1 and PfromepsLD(energy_gcm3_log10[counter]) > pressure_dyne_log10[counter]:
    counter += 1

crossenergy = energy[counter]
crosspressure = pressure[counter]

# Low-density branch (crust), tabulated in ε then mapped to P
energyLD = np.linspace(1.0e-13, crossenergy, 1000).tolist()
energyLD_log10_gcm3 = [np.log10(item * e0 * MeV4togcm3) for item in energyLD]
pressureLD_log10 = [PfromepsLD(item) for item in energyLD_log10_gcm3]
pressureLD = [10.0 ** p * dynetoMeV4 / e0 for p in pressureLD_log10]

# Join: (ε, P)
joined_energy = energyLD + energy[counter:]
joined_pressure = pressureLD + pressure[counter:]

# Plot EoS (Fig. 5.1)
joindenerplot = [item * e0 * MeV4togcm3 for item in joined_energy]
joindpresplot = [item * e0 / dynetoMeV4 for item in joined_pressure]
logjoindenerplot = [np.log10(item) for item in joindenerplot]
logjoindpresplot = [np.log10(item) for item in joindpresplot]

plt.ylabel(r'$\log_{10} P\;(\mathrm{dyne/cm}^2)$')
plt.xlabel(r'$\log_{10} \epsilon\;(\mathrm{g/cm}^3)$')
plt.ylim(26, 38)
plt.xlim(9, 16)
plt.plot(logjoindenerplot, logjoindpresplot, 'k')
plt.grid(True)
plt.show()

# Interpolate ε(P) for TOV (domain: joined_pressure, range: joined_energy)
EoS = interpolate.interp1d(joined_pressure, joined_energy, kind='linear', fill_value="extrapolate", assume_sorted=False)

# ----------------------------
# TOV integration to build M–R curve (np e μ matter)
# ----------------------------
Pcstart = 4e33 * dynetoMeV4 / e0  # dimensionless
Pcend = joined_pressure[-2]
Pcstep = 1.05

Mlist = []
Rlist = []
centralpressures = []

rstep = 0.001  # km
tol = joined_pressure[0]
Pc = Pcstart

while Pc < Pcend:
    P = Pc
    M = 0.0
    r = 0.0
    centralpressures.append(Pc)

    # Integrate outward until pressure ~ surface
    while P > tol and r < 100.0:  # hard stop radius just in case
        r += rstep
        eps = float(EoS(P))
        M += rstep * dMdr(r, eps)
        P += rstep * dPdr(r, M, P, eps)

    Rlist.append(r)
    Mlist.append(M)
    Pc *= Pcstep

# Split stable/unstable by turning point (max M)
idx_max = int(np.argmax(Mlist))
StableM = Mlist[: idx_max + 1]
StableR = Rlist[: idx_max + 1]
UnstableM = Mlist[idx_max + 1 :]
UnstableR = Rlist[idx_max + 1 :]

# Report smallest radius at max mass and the maximum mass itself
print("Radius at maximum mass (km):", Rlist[idx_max])
print("Maximum mass (M_sun):", Mlist[idx_max])

# Plot M–R and M–epsilon_c (Fig. 5.2)
centraldensities = [float(EoS(Pc)) * e0 * MeV4togcm3 for Pc in centralpressures]
Stablecd = centraldensities[: idx_max + 1]
Unstablecd = centraldensities[idx_max + 1 :]
Stablecd_log = [np.log10(x) for x in Stablecd]
Unstablecd_log = [np.log10(x) for x in Unstablecd]

fig, axarr = plt.subplots(1, 2, sharey=True, figsize=(9, 4))

axarr[0].grid(True)
axarr[0].plot(StableR, StableM, 'k-', label='stable')
axarr[0].plot(UnstableR, UnstableM, 'k--', label='unstable')
axarr[0].set_xlabel('Radius (km)')
axarr[0].set_ylabel(r'$M/M_\odot$')
axarr[0].set_xlim([8, 25])
axarr[0].legend()

axarr[1].grid(True)
axarr[1].plot(Stablecd_log, StableM, 'k-')
axarr[1].plot(Unstablecd_log, UnstableM, 'k--')
axarr[1].set_xlabel(r'$\log_{10}\,\epsilon_c\;(\mathrm{g/cm}^3)$')
axarr[1].set_xlim([14.45, 15.8])

plt.ylim(0.2, 2.5)
plt.show()
