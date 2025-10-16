# plots/plot_eigenfunctions.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def read_table(path):
    # Robust read: handles header line written by your writecols
    try:
        df = pd.read_csv(path, delim_whitespace=True, comment='#')
    except Exception:
        df = pd.read_csv(path, sep=r'\s+', engine='python', comment='#')
    # Clean column names (strip stray chars)
    df.columns = [c.strip() for c in df.columns]
    return df

def unique_omega2(df):
    if 'omega2' not in df.columns:
        return None
    vals = df['omega2'].to_numpy()
    vals = vals[~np.isnan(vals)]
    # Round to reduce duplicates from repeated rows
    uniq = np.unique(np.round(vals, 12))
    return uniq

def main(path="numerics/data/nmodes_norm.dat", savefig=True):
    df = read_table(path)
    if 'r' not in df.columns:
        raise RuntimeError("Could not find column 'r' in file.")
    r = df['r'].to_numpy()
    # U-columns: all that start with 'U'
    Ucols = [c for c in df.columns if c.startswith('U')]
    if not Ucols:
        raise RuntimeError("No eigenfunction columns found (expected columns named like 'U0', 'U1', ...).")

    # Try to list omega2 values (if present)
    omegas = unique_omega2(df)
    if omegas is not None:
        print("Detected omega^2 values (unique):")
        for i, w2 in enumerate(omegas):
            print(f"  ω^2[{i}] = {w2}")

    # Plot eigenfunctions
    plt.figure(figsize=(7,5))
    for c in Ucols:
        u = df[c].to_numpy()
        # Ignore NaNs (outer cut) for plotting
        mask = ~np.isnan(u)
        plt.plot(r[mask], u[mask], label=c)
    plt.xlabel(r"$r$")
    plt.ylabel(r"$U_n(r)$ (normalized)")
    plt.title("Normalized radial eigenfunctions")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    if savefig:
        out = Path(path).with_suffix('.pdf')
        plt.savefig(out)
        print(f"Saved {out}")
    plt.show()

if __name__ == "__main__":
    main()
