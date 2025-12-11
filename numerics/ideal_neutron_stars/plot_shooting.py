import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14


def plot_shoot_relative(filename, save_file=False):
    """
    Plot shooting-method results from numerics/data/<filename>.
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

    # --- colors from viridis ---
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, len(Ucols)))

    # --- plot ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, color in zip(Ucols, colors):
        y = np.asarray(data[col], dtype=float) / 0.0003
        ax.plot(x, y, lw=1.5, color=color)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-2, 2)

    ax.set_xlabel(r"$r/R$", fontsize=18)
    ax.set_ylabel(r"$U_n(r) / \max|U_0|$", fontsize=18)
    ax.set_title(r"Shooting Method Convergence: $U_n(r)$ vs $r/R$", fontsize=20)

    ax.grid(True, alpha=0.3, linestyle=":")

    fig.tight_layout()

    if save_file:
        outname = os.path.splitext(filename)[0] + "_shoot_relative.pdf"
        outpath = os.path.join("numerics", "data", outname)
        fig.savefig(outpath, dpi=300)
        print(f"Saved: {outpath}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    plot_shoot_relative("shoot.dat", save_file=False)
