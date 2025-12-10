# plot_style.py
import matplotlib.pyplot as plt
from cycler import cycler

def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "figure.figsize": (7, 5),
    })

    # nice consistent color cycle
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
