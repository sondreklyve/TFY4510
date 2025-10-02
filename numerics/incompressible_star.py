#!/usr/bin/env python3
"""
Constant-density ("incompressible") TOV: analytic vs numeric check.

What it does
------------
1) Defines an incompressible EoS: ε(P) = ε0 for P>0, else 0.
2) For a chosen compactness C = 2M/R (< 8/9), builds the analytic solution:
   - m(r) = (4π/3) ε0 r^3
   - Φ(r) = sqrt(1 - C (r/R)^2), Φ_R = sqrt(1 - C)
   - P(r) = ε0 [Φ_R - Φ(r)] / [3 Φ(r) - Φ_R]
   - e^{2α(r)} = ¼ (3 Φ_R - Φ(r))^2
3) Runs your TOV integrator with central pressure P_c from the analytic relation
      P_c/ε0 = (1 - Φ_R) / (3 Φ_R - 1)
   and compares dimensionless profiles: (r/R, P/P_c, m/M).
4) Sweeps a few compactness values to illustrate agreement & check approach to the
   Buchdahl limit C → 8/9 from below.

Outputs
-------
- numerics/data/const_eos_check/profile_C{value}.pdf   (P/Pc and m/M vs r/R)
- numerics/data/const_eos_check/errors.csv             (per-C max errors)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Project-local imports (match your repo)
from tov import soltov           # numeric TOV solver: (ϵ(P), P0) -> r, m, P, α, ϵ
from utils import writecols      # small CSV helper used elsewhere

# -------------------------
# Model & analytic helpers
# -------------------------

def eps_const_factory(eps0):
    """Return ε(P) function for an incompressible star: ε=ε0 for P>0, 0 otherwise."""
    def eps_const(P):
        return eps0 if P > 0 else 0.0
    return eps_const

def pc_over_eps0_from_C(C):
    """Analytic central pressure ratio P_c/ε0 as function of compactness C = 2M/R.
       Uses Φ_R = sqrt(1 - C). Valid for C < 8/9."""
    phi_R = np.sqrt(1.0 - C)
    # Guard against crossing the Buchdahl bound
    if not (0.0 < C < 8.0/9.0):
        raise ValueError("Compactness must satisfy 0 < C < 8/9 for a regular solution.")
    return (1.0 - phi_R)/(3.0*phi_R - 1.0)

def R_M_from_eps0_C(eps0, C):
    """Given ε0 and compactness C, return (R, M) from C = 2M/R = (8π/3) ε0 R^2."""
    R = np.sqrt( (3.0*C) / (8.0*np.pi*eps0) )
    M = 0.5 * C * R
    return R, M

def analytic_profiles(eps0, C, r_grid):
    """Analytic m(r), P(r), alpha(r) for incompressible star at fixed C and ε0."""
    R, M = R_M_from_eps0_C(eps0, C)
    x = r_grid / R
    phi_R = np.sqrt(1.0 - C)
    phi_r = np.sqrt(1.0 - C * x**2)

    # Pressure profile
    P = eps0 * (phi_r - phi_R) / (3.0*phi_R - phi_r)

    # Mass profile
    m = (4.0*np.pi/3.0) * eps0 * r_grid**3  # = (M/R^3) r^3

    # Time redshift potential
    e2alpha = 0.25 * (3.0*phi_R - phi_r)**2

    # Central values (handy to return)
    Pc = eps0 * (1.0 - phi_R) / (3.0*phi_R - 1.0)

    return R, M, Pc, m, P, e2alpha

# -------------------------
# Comparison utility
# -------------------------

def compare_numeric_vs_analytic(eps0, C_target, npts=512,
                                outdir="numerics/data/const_eos_check"):
    os.makedirs(outdir, exist_ok=True)

    # ---- 1) Get Pc from analytic relation for the *target* compactness (geom. units) ----
    # This only seeds the solver; we will later measure the achieved compactness C_num.
    Pc_over_eps0 = pc_over_eps0_from_C(C_target)
    Pc_guess = Pc_over_eps0 * eps0

    # ---- 2) Run numeric TOV with incompressible EoS ----
    eps_const = eps_const_factory(eps0)
    rN, mN, PN, aN, epsN = soltov(eps_const, Pc_guess)
    RN, MN = rN[-1], mN[-1]

    # ---- 3) Infer the numeric compactness C_num (handle geometric vs SI mass) ----
    C_num = 2.0 * MN / RN            # works if m is geometric (so that 1 - 2m/r appears)
    if not (0.0 < C_num < 1.0):      # fall back to SI→geom conversion if needed
        from constants import G, c
        C_num = 2.0 * G * MN / (RN * c**2)

    # Basic sanity
    assert 0.0 < C_num < 8.0/9.0, f"Compactness out of range: C_num={C_num}"

    # ---- 4) Build analytic *dimensionless* shapes for this C_num ----
    xA = np.linspace(0.0, 1.0, npts)
    phiR = np.sqrt(1.0 - C_num)
    phi  = np.sqrt(1.0 - C_num * xA**2)

    # Analytic P/Pc and m/M (independent of units)
    PA_bar = (phi - phiR) / (3.0*phiR - phi)
    mA_bar = xA**3

    # ---- 5) Dimensionless numerics against their own scales ----
    xN     = rN / RN
    PN_bar = PN / PN[0]     # normalize by *numeric* central pressure
    mN_bar = mN / MN

    # ---- 6) Error measures on the same x-grid ----
    from numpy import interp
    PA_on_N = interp(xN, xA, PA_bar)
    mA_on_N = interp(xN, xA, mA_bar)

    p_err_max = np.max(np.abs(PN_bar - PA_on_N))
    m_err_max = np.max(np.abs(mN_bar - mA_on_N))

    # ---- 7) Plot ----
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(xA, PA_bar, lw=2, label="Analytic $P/P_c$")
    #ax.plot(xN, PN_bar, lw=1.6, ls="--", label="Numeric $P/P_c$")
    ax.set_xlabel(r"$r/R$")
    ax.set_ylabel(r"$P/P_c$")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(xA, mA_bar, lw=2, color="tab:orange", label="Analytic $m/M$")
    #ax2.plot(xN, mN_bar, lw=1.6, ls="--", color="tab:red", label="Numeric $m/M$")
    ax2.set_ylabel(r"$m/M$")

    lines, labels = [], []
    for axis in (ax, ax2):
        L = axis.get_legend_handles_labels()
        lines += L[0]; labels += L[1]
    ax.legend(lines, labels, loc="lower right", frameon=True)
    ax.set_title(f"Incompressible star — C={C_target:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"profile_C{C_target:.3f}.pdf"))
    plt.close(fig)

    return {
        "C_target": C_target,
        "C_num": C_num,
        "Pc_over_eps0_target": Pc_over_eps0,
        "RN": RN, "MN": MN,
        "max_abs_err_P_over_Pc": p_err_max,
        "max_abs_err_m_over_M": m_err_max,
    }


def main():
    import csv, os

    # This eps0 is only used by the older (dimensionless) compare() signature.
    eps0 = 1.0

    # Compactness sweep (must be < 8/9)
    Cs = [0.10, 0.30, 0.60, 0.85]

    rows = []
    for C in Cs:
        row = compare_numeric_vs_analytic(eps0, C)
        rows.append(row)

        # Backward/forward-compatible field accessors:
        C_print   = row.get("C", row.get("C_target", float("nan")))
        Pc_ratio  = row.get("Pc_over_eps0", row.get("Pc_over_eps0_target", float("nan")))

        # Old API provided RN_over_R and MN_over_M; new API often provides RN and MN.
        RN_over_R = row.get("RN_over_R", None)
        MN_over_M = row.get("MN_over_M", None)

        # Pretty print line:
        if RN_over_R is not None and MN_over_M is not None:
            print(
                f"C={C_print:.3f}  Pc/eps0={Pc_ratio:.6f}  "
                f"RN/R={RN_over_R:.5f}  MN/M={MN_over_M:.5f}  "
                f"max|Δ(P/Pc)|={row['max_abs_err_P_over_Pc']:.3e}  "
                f"max|Δ(m/M)|={row['max_abs_err_m_over_M']:.3e}"
            )
        else:
            # Fall back to printing absolute RN, MN if ratios are not returned
            RN = row.get("RN", float("nan"))
            MN = row.get("MN", float("nan"))
            C_num = row.get("C_num", float("nan"))
            print(
                f"C_target={C_print:.3f}  C_num={C_num:.3f}  Pc/eps0={Pc_ratio:.6f}  "
                f"RN={RN:.5e}  MN={MN:.5e}  "
                f"max|Δ(P/Pc)|={row['max_abs_err_P_over_Pc']:.3e}  "
                f"max|Δ(m/M)|={row['max_abs_err_m_over_M']:.3e}"
            )

    # ---- Write a CSV in a robust way (no dependency on writecols signature) ----
    outcsv = "numerics/data/const_eos_check/errors.csv"
    os.makedirs(os.path.dirname(outcsv), exist_ok=True)

    # Choose a header that works for both APIs
    fieldnames = [
        # compactness / pressure ratio
        "C", "C_target", "C_num", "Pc_over_eps0", "Pc_over_eps0_target",
        # radii/masses (either ratios or absolutes)
        "RN_over_R", "MN_over_M", "RN", "MN",
        # errors
        "max_abs_err_P_over_Pc", "max_abs_err_m_over_M",
    ]

    # Normalize rows to this header (missing keys -> blank)
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {outcsv} and per-C plots in numerics/data/const_eos_check/")


if __name__ == "__main__":
    main()
