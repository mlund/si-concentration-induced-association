"""Recompute Fig. 5 of Li et al. 2016 and overlay the digitized experiments.

The paper gives Z_eff = 2, D = 6 nm, kappa_a = 20/D and K = 1 kT, but not the
Bjerrum length or Debye-length formula. Water at 25 C is assumed for both.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator

from wertheim import Grid, Patch, double_yukawa, hard_sphere, sweep

ROOT = Path(__file__).resolve().parents[1]
DIAMETER_NM = 6.0
BJERRUM_NM = 0.7
DARK_GREEN = "#008B00"
PHI_MAX = 0.158  # right edge of the published axis
PHIS = np.geomspace(0.001, PHI_MAX, 50)
# Colours sampled from the published figure, for side-by-side comparison.
TOP_CURVES = {25: "#000000", 15: "#FF0000", 14: "#00FF00", 12.5: "#0000FF", 10.5: "#FFA500"}


def debye_kappa(salt_molar: float) -> float:
    return DIAMETER_NM * np.sqrt(salt_molar) / 0.304


def protein_potential(salt_molar: float):
    return double_yukawa(
        well=1.0,
        kappa_attr=20.0,
        bjerrum=BJERRUM_NM / DIAMETER_NM,
        charge=2.0,
        kappa_debye=debye_kappa(salt_molar),
    )


def s0_curve(beta_eps: float, potential, grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    solutions = sweep(PHIS, Patch(beta_eps), potential, grid)
    return np.array([s.s0 for s in solutions]), np.array([s.monomer_fraction for s in solutions])


def plot_experiments(ax, data) -> None:
    style = {"ms": 4.2, "ls": "none", "zorder": 5}
    low = data[data["salt_mM"] == 35]
    high = data[data["salt_mM"] == 200]
    ax.plot(low["phi"], low["S0"], "o", mfc="white", mec="black", mew=0.8, label="35 mM", **style)
    ax.plot(high["phi"], high["S0"], "o", mfc=DARK_GREEN, mec=DARK_GREEN, mew=0.8, label="200 mM", **style)


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
    # skip_header: genfromtxt would otherwise take column names from the first comment.
    data = np.genfromtxt(
        ROOT / "data" / "li2016_fig5_digitized.csv", delimiter=",", names=True, skip_header=3
    )
    grid = Grid()
    fig, (top, middle, bottom) = plt.subplots(
        3, 1, sharex=True, figsize=(3.4, 6.6), layout="constrained"
    )

    for beta_eps, colour in TOP_CURVES.items():
        s0, _ = s0_curve(beta_eps, hard_sphere, grid)
        top.plot(PHIS, s0, color=colour, lw=1.1, label=rf"$\beta\epsilon$ = {beta_eps:g}")
    s0, _ = s0_curve(0.0, hard_sphere, grid)
    top.plot(PHIS, s0, color="black", lw=1.1, ls="--", label="Hard spheres")
    plot_experiments(top, data)
    top.set(ylabel=r"$\mathit{S}(0)$", ylim=(0.5, 2.0))
    # Marker key goes in empty space: among the curves it reads as extra data.
    curve_key = top.legend(
        handles=top.lines[:6],
        loc="upper right",
        handlelength=1.2,
        handletextpad=0.4,
        borderaxespad=0.2,
        labelspacing=0.25,
        fontsize=7,
    )
    top.add_artist(curve_key)
    top.legend(handles=top.lines[6:], loc="lower left", bbox_to_anchor=(0.07, 0.0), fontsize=7)

    s0_35, a_35 = s0_curve(11.2, protein_potential(0.035), grid)
    s0_200, _ = s0_curve(0.0, protein_potential(0.2), grid)
    middle.plot(PHIS, s0_35, color="black", lw=1.3)
    middle.plot(PHIS, s0_200, color=DARK_GREEN, lw=1.3)
    plot_experiments(middle, data)
    middle.set(ylabel=r"$\mathit{S}(0)$", ylim=(0.5, 1.3))
    middle.text(0.0012, 1.19, r"35 mM, $\beta\epsilon$ = 11.2", fontsize=7)
    middle.text(0.0012, 0.91, r"200 mM, $\beta\epsilon$ = 0", color=DARK_GREEN, fontsize=7)

    bottom.plot(PHIS, a_35, color="black", lw=1.3)
    bottom.set(
        ylabel=r"$\rho_0/\rho$", ylim=(0.34, 1.0), xlabel=r"Volume fraction, $\mathit{\phi}$"
    )
    bottom.text(0.0012, 0.72, r"35 mM model", fontsize=7)

    bottom.set_xscale("log")
    bottom.set_xlim(0.001, PHI_MAX)
    plain_number = FuncFormatter(lambda value, _: f"{value:g}")
    bottom.xaxis.set_major_formatter(plain_number)
    for ax, letter, step in zip((top, middle, bottom), "abc", (0.25, 0.2, 0.1)):
        ax.yaxis.set_major_locator(MultipleLocator(step))
        ax.yaxis.set_major_formatter(plain_number)
        ax.text(0.02, 0.05, letter, transform=ax.transAxes, fontweight="bold", fontsize=9)

    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig5.pdf")
    fig.savefig(out / "fig5.png", dpi=300)

    for phi in (0.01, 0.021, 0.05, 0.1):
        i = np.argmin(abs(PHIS - phi))
        print(f"phi={PHIS[i]:.4f}  S0(35)={s0_35[i]:.3f}  S0(200)={s0_200[i]:.3f}  a={a_35[i]:.3f}")


if __name__ == "__main__":
    main()
