import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -------------------------
# Mass-radius diagram
# -------------------------
file_prefix = "numerics/data"

files = {
    "Newtonian dP/dr, non-rel. ε(P)": f"{file_prefix}/nrnewt.dat",
    "Newtonian dP/dr, general ε(P)": f"{file_prefix}/grnewt.dat",
    "relativistic dP/dr, non-rel. ε(P)": f"{file_prefix}/nr.dat",
    "relativistic dP/dr, general ε(P)": f"{file_prefix}/gr.dat",
}

plt.figure(figsize=(7,6))

for label, fname in files.items():
    data = np.loadtxt(fname, skiprows=1)  # skip header
    Pc = data[:,0]   # central pressure
    M  = data[:,1]   # mass [Msun]
    R  = data[:,2]   # radius [km]
    plt.plot(R, M, label=label)

plt.xlabel(r"$R\,[\mathrm{km}]$")
plt.ylabel(r"$M/M_\odot$")
plt.title("Mass-radius diagram for ideal neutron stars")
plt.xlim(0, 50)
plt.ylim(0, 1.1)
plt.legend()
plt.grid()
plt.show()



# -------------------------
# Pressure profiles (lines, not filled)
# -------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data  = np.loadtxt(f"{file_prefix}/pressures.dat", skiprows=1)

Pc    = data[:, 0]         # central pressures
xcols = data[:, 1:15]      # r/R samples (if these are NOT normalized, add the normalization line below)
pcols = data[:, 15:]       # P(r)/P(0)

# If xcols are absolute radii, uncomment this to normalize each row to r/R:
# xcols = xcols / xcols.max(axis=1, keepdims=True)

# Sort by Pc so the colorbar runs smoothly with Pc
order = np.argsort(Pc)
Pc, xcols, pcols = Pc[order], xcols[order], pcols[order]

fig, ax = plt.subplots(figsize=(7,6))

# Colormap by log10(Pc)
logPc = np.log10(Pc)
norm  = plt.Normalize(vmin=logPc.min(), vmax=logPc.max())
cmap  = cm.plasma

# (A) Plot ALL curves very faintly so the background isn't “filled”
for i in range(len(Pc)):
    ax.plot(xcols[i], pcols[i], color=cmap(norm(logPc[i])),
            linewidth=0.6, alpha=0.12)

# (B) Overlay a small, evenly spaced subset of representative curves
n_rep = 10
idx   = np.unique(np.linspace(0, len(Pc)-1, n_rep).astype(int))
for i in idx:
    ax.plot(xcols[i], pcols[i], color=cmap(norm(logPc[i])),
            linewidth=1.8, alpha=0.95)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r"$\log_{10}(P_c/\mathrm{Pa})$")

ax.set_xlabel(r"$r/R$")
ax.set_ylabel(r"$P(r)/P(0)$")
ax.set_title("Pressure profiles for ideal neutron stars")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.2)
plt.show()
