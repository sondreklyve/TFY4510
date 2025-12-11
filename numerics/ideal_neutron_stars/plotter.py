import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

file_prefix = "numerics/data"

files = {
    "NR (simple EoS)": f"{file_prefix}/nrnewt.dat",
    "NR (full EoS)": f"{file_prefix}/grnewt.dat",
    "GR (simple EoS)": f"{file_prefix}/nr.dat",
    "GR (full EoS)": f"{file_prefix}/gr.dat",
}

fig, ax = plt.subplots(figsize=(8, 7))

all_logPc = []

for label, fname in files.items():
    data = np.loadtxt(fname, skiprows=1)
    Pc = data[:, 0]
    all_logPc.extend(np.log10(Pc))

all_logPc = np.array(all_logPc)
norm = plt.Normalize(vmin=all_logPc.min(), vmax=all_logPc.max())
cmap = plt.cm.viridis_r

max_mass_R = None
max_mass_M = None

for label, fname in files.items():
    data = np.loadtxt(fname, skiprows=1)
    Pc, M, R = data[:, 0], data[:, 1], data[:, 2]

    order = np.argsort(Pc)
    Pc, M, R = Pc[order], M[order], R[order]
    logPc = np.log10(Pc)

    # Plot segments with color by Pc
    for i in range(len(Pc) - 1):
        ax.plot(
            R[i : i + 2],
            M[i : i + 2],
            color=cmap(norm(logPc[i])),
            lw=1.8,
        )

    # Track maximum mass only for GR (full EoS)
    if label == "GR (full EoS)":
        idx_max = np.argmax(M)
        max_mass_R = R[idx_max]
        max_mass_M = M[idx_max]

# Hardcoded label positions (R, M)
label_positions = {
    "NR (simple EoS)": (1.3, 1.9),
    "NR (full EoS)": (0.55, 1.4),
    "GR (simple EoS)": (0.3, 1.0),
    "GR (full EoS)": (0.35, 0.25),
}

for label, (x_lab, y_lab) in label_positions.items():
    ax.text(
        x_lab,
        y_lab,
        label,
        fontsize=12,
        color="0.1",
    )

# Mark only the OV maximum mass
if max_mass_R is not None:
    ax.plot(max_mass_R, max_mass_M, "ko", markersize=6)
    ax.annotate(
        rf"$M_{{\max}}\approx {max_mass_M:.2f}\,M_\odot$",
        xy=(max_mass_R, max_mass_M),        # point to the max
        xytext=(1.25, 0.1),                  # <-- choose a clean spot
        textcoords="data",
        arrowprops=dict(arrowstyle="->", lw=1.0, color="0.3"),
        fontsize=14,
        color="0.2",
        ha="left",
        va="center",
    )

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.18)
cbar.set_label(r"$\log_{10}(P_c/\mathrm{Pa})$", fontsize=16)

# Labels + title
ax.set_xlabel(r"$R\,[\mathrm{km}]$", fontsize=18)
ax.set_ylabel(r"$M/M_\odot$", fontsize=18)
ax.set_title(r"Mass--radius diagram for ideal neutron stars", fontsize=20)

ax.grid(True, linestyle=":", alpha=0.5)
plt.tight_layout()
plt.show()

# -------------------------
# Pressure profiles
# -------------------------
data = np.genfromtxt(f"{file_prefix}/pressures.dat", skip_header=1)

P0_vals = data[:14, 0]        # first 14 central pressures
X = data[:, 1:15]             # r/R
P = data[:, 15:29]            # P(r)/P(0)

logP = np.log10(P0_vals)
norm = plt.Normalize(vmin=logP.min(), vmax=logP.max())
cmap = plt.cm.viridis_r

fig, ax = plt.subplots(figsize=(8, 7))

for n in range(14):
    color = cmap(norm(logP[n]))
    ax.plot(X[:, n], P[:, n], color=color, lw=2.0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15)
cbar.set_label(r"$\log_{10}(P_c)$", fontsize=16)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$r/R$", fontsize=18)
ax.set_ylabel(r"$P(r)/P(0)$", fontsize=18)
ax.set_title("Pressure profiles colored by central pressure", fontsize=20)
ax.grid(True, alpha=0.3, linestyle=":")

plt.tight_layout()
plt.show()
