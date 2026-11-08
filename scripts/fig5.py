"""Recompute Fig. 5 of Li et al. 2016 and overlay the digitized experiments."""

import matplotlib.pyplot as plt
import numpy as np
from common import FITTED_BETA_EPS, ROOT, protein_potential, read_digitized
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from wertheim import Grid, Patch, Potential, hard_sphere, sweep

DARK_GREEN = "#008B00"
PHI_MAX = 0.158  # right edge of the published axis
PHIS = np.geomspace(0.001, PHI_MAX, 50)
# Colours sampled from the published figure, for side-by-side comparison.
TOP_CURVES = {25: "#000000", 15: "#FF0000", 14: "#00FF00", 12.5: "#0000FF", 10.5: "#FFA500"}


def plain_number(value: float, _: object) -> str:
    return f"{value:g}"


def save_figure(fig: Figure, stem: str) -> None:
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=300)


def s0_curve(beta_eps: float, potential: Potential, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    solutions = sweep(PHIS, Patch(beta_eps), potential, grid)
    return np.array([s.s0 for s in solutions]), np.array([s.monomer_fraction for s in solutions])


def plot_experiments(ax: Axes, data: np.ndarray) -> None:
    style = {"ms": 4.2, "ls": "none", "zorder": 5}
    low = data[data["salt_mM"] == 35]
    high = data[data["salt_mM"] == 200]
    ax.plot(low["phi"], low["S0"], "o", mfc="white", mec="black", mew=0.8, label="35 mM", **style)
    ax.plot(
        high["phi"],
        high["S0"],
        "o",
        mfc=DARK_GREEN,
        mec=DARK_GREEN,
        mew=0.8,
        label="200 mM",
        **style,
    )


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "mathtext.default": "regular",
            "legend.frameon": False,
        }
    )


def main() -> None:
    apply_style()
    data = read_digitized("li2016_fig5_digitized.csv")
    grid = Grid()
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, sharex=True, figsize=(3.4, 6.6), layout="constrained"
    )

    for beta_eps, colour in TOP_CURVES.items():
        s0, _ = s0_curve(beta_eps, hard_sphere, grid)
        top.plot(PHIS, s0, color=colour, lw=1.1, label=rf"$\beta\epsilon$ = {beta_eps:g}")
    s0, _ = s0_curve(0.0, hard_sphere, grid)
    top.plot(PHIS, s0, color="black", lw=1.1, ls="--", label="Hard spheres")
    n_curves = len(top.lines)
    plot_experiments(top, data)
    top.set(ylabel=r"$\mathit{S}(0)$", ylim=(0.5, 2.0))
    curve_key = top.legend(
        handles=top.lines[:n_curves],
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.25,
        fontsize=7,
    )
    top.add_artist(curve_key)
    # Marker key goes in empty space: among the curves it reads as extra data.
    top.legend(
        handles=top.lines[n_curves:], loc="lower left", bbox_to_anchor=(0.07, 0.0), fontsize=7
    )

    s0_35, a_35 = s0_curve(FITTED_BETA_EPS[35], protein_potential(35), grid)
    s0_200, _ = s0_curve(FITTED_BETA_EPS[200], protein_potential(200), grid)
    middle.plot(PHIS, s0_35, color="black", lw=1.3)
    middle.plot(PHIS, s0_200, color=DARK_GREEN, lw=1.3)
    plot_experiments(middle, data)
    middle.set(ylabel=r"$\mathit{S}(0)$", ylim=(0.5, 1.3))
    middle.text(0.0012, 1.19, rf"35 mM, $\beta\epsilon$ = {FITTED_BETA_EPS[35]:g}", fontsize=7)
    middle.text(
        0.0012,
        0.91,
        rf"200 mM, $\beta\epsilon$ = {FITTED_BETA_EPS[200]:g}",
        color=DARK_GREEN,
        fontsize=7,
    )

    bottom.plot(PHIS, a_35, color="black", lw=1.3)
    bottom.set(
        ylabel=r"$\rho_0/\rho$", ylim=(0.34, 1.0), xlabel=r"Volume fraction, $\mathit{\phi}$"
    )
    bottom.text(0.0012, 0.72, r"35 mM model", fontsize=7)

    bottom.set_xscale("log")
    bottom.set_xlim(0.001, PHI_MAX)
    bottom.xaxis.set_major_formatter(plain_number)
    for ax, letter, step in zip((top, middle, bottom), "abc", (0.25, 0.2, 0.1), strict=True):
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(plain_number)
        ax.text(0.02, 0.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)

    save_figure(fig, "fig5")

    for phi in (0.01, 0.021, 0.05, 0.1):
        i = np.argmin(abs(PHIS - phi))
        print(f"phi={PHIS[i]:.4f}  S0(35)={s0_35[i]:.3f}  S0(200)={s0_200[i]:.3f}  a={a_35[i]:.3f}")


if __name__ == "__main__":
    main()
