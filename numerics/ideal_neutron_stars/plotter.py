import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -------------------------
# Mass–radius diagram
# -------------------------
file_prefix = "numerics/data"

files = {
    "Newtonian $dP/dr$, non-rel. $\\epsilon(P)$": f"{file_prefix}/nrnewt.dat",
    "Newtonian $dP/dr$, general $\\epsilon(P)$": f"{file_prefix}/grnewt.dat",
    "Relativistic $dP/dr$, non-rel. $\\epsilon(P)$": f"{file_prefix}/nr.dat",
    "Relativistic $dP/dr$, general $\\epsilon(P)$": f"{file_prefix}/gr.dat",
}

fig, ax = plt.subplots(figsize=(7,6))

all_logPc = []

# First pass: collect all Pc values for normalization
for label, fname in files.items():
    data = np.loadtxt(fname, skiprows=1)
    Pc = data[:,0]
    all_logPc.extend(np.log10(Pc))
    
all_logPc = np.array(all_logPc)
norm = plt.Normalize(vmin=all_logPc.min(), vmax=all_logPc.max())
cmap = cm.plasma.reversed()   # inverted colormap

# Second pass: plot each curve
for label, fname in files.items():
    data = np.loadtxt(fname, skiprows=1)
    Pc, M, R = data[:,0], data[:,1], data[:,2]
    
    order = np.argsort(Pc)
    Pc, M, R = Pc[order], M[order], R[order]
    logPc = np.log10(Pc)
    
    for i in range(len(Pc)-1):
        ax.plot(R[i:i+2], M[i:i+2],
                color=cmap(norm(logPc[i])), lw=1.5, alpha=0.9,
                label=None)
    
    # add one label at the last point of each curve
    ax.text(R[-1]+0.1, M[-1], label, fontsize=8, va="center")

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation="horizontal", pad=0.15)
cbar.set_label(r"$\log_{10}(P_c/\mathrm{Pa})$")

ax.set_xlabel(r"$R\,[\mathrm{km}]$")
ax.set_ylabel(r"$M/M_\odot$")
ax.set_title("Mass–radius diagram for ideal neutron stars")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


file_prefix = "numerics/data"
data = np.genfromtxt(f"{file_prefix}/pressures.dat", skip_header=1)

# Extract first 14 pressures (rows 0–13, col 0)
P0_vals = data[:14, 0]   # 1e-06 ... 1e7
X = data[:, 1:15]
P = data[:, 15:29]

# Normalize log pressures for colormap
logP = np.log10(P0_vals)
norm = plt.Normalize(vmin=logP.min(), vmax=logP.max())
cmap = cm.plasma.reversed()   # invert colormap here

fig, ax = plt.subplots(figsize=(7,6))

# Plot each curve with color from its P0
for n in range(14):
    color = cmap(norm(logP[n]))
    ax.plot(X[:, n], P[:, n], color=color, lw=1.8)

# Add a colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15)
cbar.set_label(r"$\log_{10}(P_c)$")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$r/R$")
ax.set_ylabel(r"$P(r)/P(0)$")
ax.set_title("Pressure profiles colored by central pressure")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
