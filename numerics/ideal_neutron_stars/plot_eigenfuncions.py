import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

def read_table(path):
    try:
        df = pd.read_csv(path, delim_whitespace=True, comment='#')
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', engine='python', comment='#')
    df.columns = [c.strip() for c in df.columns]
    return df

def unique_omega2(df):
    if 'omega2' not in df.columns:
        return None
    vals = df['omega2'].to_numpy()
    vals = vals[~np.isnan(vals)]
    return np.unique(np.round(vals, 12))

def main(path="numerics/data/nmodes_norm.dat", savefig=False):
    df = read_table(path)
    if 'r' not in df.columns:
        raise RuntimeError("Could not find column 'r' in file.")
    r = df['r'].to_numpy()

    # eigenfunctions
    Ucols = [c for c in df.columns if c.startswith('U')]
    if not Ucols:
        raise RuntimeError("No eigenfunction columns found (expected U0, U1, ...).")

    # colormap
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.2, 0.9, len(Ucols)))

    # plot
    plt.figure(figsize=(8, 5.5))
    for col, col_color in zip(Ucols, colors):
        u = df[col].to_numpy()
        mask = ~np.isnan(u)
        plt.plot(r[mask], u[mask], lw=2.0, color=col_color, label=col)

    plt.xlabel(r"$r$", fontsize=18)
    plt.ylabel(r"$U_n(r)$", fontsize=18)
    plt.title("Normalized Radial Eigenfunctions", fontsize=20)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(fontsize=14, ncol=2)
    plt.tight_layout()

    if savefig:
        out = Path(path).with_suffix(".pdf")
        plt.savefig(out, dpi=300)
        print(f"Saved {out}")

    plt.show()

if __name__ == "__main__":
    main()
