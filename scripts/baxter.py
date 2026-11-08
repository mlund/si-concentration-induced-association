"""Recompute the Baxter lines of Fig. 3 of Li et al. 2016 from the digitized Fig. 4 stickiness.

The fits themselves cannot be redone because the measured S(q) is not available.
"""

import matplotlib.pyplot as plt
import numpy as np
from common import DIAMETER_NM, read_digitized, write_results
from fig5 import apply_style, plain_number, save_figure
from matplotlib.ticker import MultipleLocator

from wertheim import baxter_s0, baxter_s_of_q

DIAMETER_ANGSTROM = 10 * DIAMETER_NM
Q_SAMPLE = 0.00498  # between the SLS and SAXS windows, where Fig. 3 lines are unobstructed
Q_RANGE = np.geomspace(0.001, 0.399, 800)
PHI_MATCH = 0.004  # above digitization and rounding differences, below the spacing of states
COLOURS = {
    (35, 15): "#FF0000",
    (35, 25): "#00FF00",
    (35, 45): "#0000FF",
    (35, 69): "#BC8F8F",
    (35, 95): "#FF00FF",
    (35, 127): "#008B00",
    (35, 161): "#670748",
    (200, 25): "#00FF00",
    (200, 45): "#0000FF",
    (200, 65): "#BC8F8F",
    (200, 75): "#00FFFF",
    (200, 90): "#FF00FF",
    (200, 130): "#008B00",
}
# These labels move aside: their lines nearly coincide with a neighbour's.
STAGGERED = {(35, 25), (35, 15)}


def nearest_row(table: np.ndarray, phi: float, salt: int) -> np.void | None:
    """Closest row in phi, or None: the figures give phi to different precision."""
    rows = table[table["salt_mM"] == salt]
    row = rows[np.argmin(abs(rows["phi"] - phi))]
    return row if abs(row["phi"] - phi) <= PHI_MATCH else None


def write_table(taus: np.ndarray, lines: np.ndarray, experiments: np.ndarray) -> None:
    rows = ["salt_mM,phi,tau,S_baxter_at_q_sample,S_published_line,S0_baxter,S0_measured"]
    for t in taus:
        salt = int(t["salt_mM"])
        at_q = baxter_s_of_q(Q_SAMPLE * DIAMETER_ANGSTROM, t["phi"], t["tau"])[0]
        line = nearest_row(lines, t["phi"], salt)
        published = "" if line is None else f"{line['S_line']:.3f}"
        experiment = nearest_row(experiments, t["phi"], salt)
        measured = "" if experiment is None else f"{experiment['S0']:.3f}"
        s0 = baxter_s0(t["phi"], t["tau"])
        rows.append(
            f"{salt},{t['phi']:.4f},{t['tau']:.3f},{at_q:.3f},{published},{s0:.3f},{measured}"
        )
    write_results("baxter_comparison.csv", rows)
    print("\n".join(rows))


def main() -> None:
    apply_style()
    taus = read_digitized("li2016_fig4_tau_digitized.csv")
    lines = read_digitized("li2016_fig3_baxter_lines_digitized.csv")
    write_table(taus, lines, read_digitized("li2016_fig5_digitized.csv"))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 4.6), layout="constrained")
    for ax, salt, ylim in zip(axes, (35, 200), ((0.5, 1.3), (0.5, 1.2)), strict=True):
        selected = lines[lines["salt_mM"] == salt]
        for line in selected:
            conc = int(line["conc_mg_mL"])
            row = nearest_row(taus, line["phi"], salt)
            if row is None:
                raise ValueError(f"no Fig. 4 stickiness near phi={line['phi']}, {salt} mM")
            s = baxter_s_of_q(Q_RANGE * DIAMETER_ANGSTROM, row["phi"], row["tau"])
            ax.plot(Q_RANGE, s, color=COLOURS[salt, conc], lw=1.0)
            q_label, offset = (0.0032, -0.04) if (salt, conc) in STAGGERED else (0.0016, 0.02)
            s_label = baxter_s_of_q(q_label * DIAMETER_ANGSTROM, row["phi"], row["tau"])[0]
            ax.text(q_label, s_label + offset, str(conc), color=COLOURS[salt, conc], fontsize=6)
        ax.plot(
            np.full(selected.size, Q_SAMPLE),
            selected["S_line"],
            "+",
            color="black",
            ms=4,
            mew=0.7,
            ls="none",
            zorder=5,
            label="Published line" if salt == 35 else None,
        )
        ax.set(ylabel=r"$\mathit{S}(q)$", ylim=ylim)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_formatter(plain_number)
        ax.text(0.97, 0.91, rf"$c_s$ = {salt} mM", transform=ax.transAxes, ha="right", fontsize=8)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.001, 0.399)
    axes[1].xaxis.set_major_formatter(plain_number)
    axes[1].set_xlabel(r"$\mathit{q}$ ($\mathrm{\AA}^{-1}$)")
    save_figure(fig, "fig3_baxter")


if __name__ == "__main__":
    main()
