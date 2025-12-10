import os
import numpy as np
import matplotlib.pyplot as plt

def plot_shoot_relative(filename, save_as="pdf"):
    """
    Plot shooting-method results from numerics/data/<filename>.

    Behavior:
    - Reads columns: omega2 nn r U0 U1 U2 ...
    - Ignores omega2 and nn
    - Normalizes x = r / r[-1]  (so x in [0,1])
    - Normalizes all U* by max|U0|
    - y-axis fixed to [-2, 2]
    - Saves figure (no GUI)

    Parameters
    ----------
    filename : str
        Name of the data file (e.g. 'shoot.dat')
    save_as : str
        Output format for saved figure (e.g. 'pdf', 'png')
    """
    path = os.path.join("numerics", "data", filename)
    data = np.genfromtxt(path, names=True, comments="#")

    # --- prepare x-axis ---
    r = np.asarray(data["r"], dtype=float)
    if r.size == 0 or r[-1] == 0:
        raise ValueError("r column is empty or final r=0; cannot normalize.")
    x = r / r[-1]

    # --- collect U-columns ---
    Ucols = [c for c in data.dtype.names if c.startswith("U")]
    if not Ucols:
        raise ValueError("No U* columns found.")
    Ucols.sort(key=lambda s: int(s[1:]) if s[1:].isdigit() else 0)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in Ucols:
        y = np.asarray(data[col], dtype=float) / 0.0003
        ax.plot(x, y, lw=1, label=col)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$r / R$")
    ax.set_ylabel(r"$U_n(r) / \max|U_0|$")
    ax.set_title("Shooting method convergence: $U_n(r)$ vs $r/R$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    outname = os.path.splitext(filename)[0] + "_shoot_relative." + save_as
    outpath = os.path.join("numerics", "data", outname)
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved: {outpath}")

# ---- self-contained call ----
plot_shoot_relative("shoot.dat", save_as="pdf")
