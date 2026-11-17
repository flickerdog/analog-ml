import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter


def setup_style():
    plt.rcParams.update({
        "savefig.dpi": 150, "font.size": 12, "axes.labelsize": 13,
        "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 11,
    })


def log_axis(ax, which, ticks):
    """Log axis labelled only at `ticks`, as plain numbers (no crowded 10^x labels)."""
    getattr(ax, f"set_{which}scale")("log")
    axis = getattr(ax, f"{which}axis")
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())