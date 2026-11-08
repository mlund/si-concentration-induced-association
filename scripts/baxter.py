"""Recompute the Baxter lines of Fig. 3 of Li et al. 2016 from the digitized Fig. 4 stickiness.

The fits themselves cannot be redone because the measured S(q) is not available.
"""

import matplotlib.pyplot as plt
import numpy as np
from fig5 import ROOT, apply_style
from matplotlib.ticker import FuncFormatter, MultipleLocator

from wertheim import baxter_s0, baxter_s_of_q

DIAMETER_ANGSTROM = 60.0
Q_SAMPLE = 0.00498  # between the SLS and SAXS windows, where Fig. 3 lines are unobstructed
Q_RANGE = np.geomspace(0.001, 0.399, 800)
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
# Label position (q, S offset), staggered where neighbouring lines nearly coincide.
LABELS = {(35, 45): (0.0016, 0.02), (35, 25): (0.0032, -0.04), (35, 69): (0.0016, 0.02),
          (35, 15): (0.0032, -0.04)}  # fmt: skip


def read(name: str) -> np.ndarray:
    return np.genfromtxt(ROOT / "data" / name, delimiter=",", names=True, skip_header=3)


def tau_for(taus: np.ndarray, phi: float, salt: int):
    """Fig. 4 row nearest in phi; Fig. 3 and 4 print phi to different precision."""
    rows = taus[taus["salt_mM"] == salt]
    row = rows[np.argmin(abs(rows["phi"] - phi))]
    return None if abs(row["phi"] - phi) > 0.004 else row


def write_table(taus, lines, experiments) -> None:
    rows = ["salt_mM,phi,tau,S_baxter_at_q_sample,S_published_line,S0_baxter,S0_measured"]
    for t in taus:
        salt = int(t["salt_mM"])
        at_q = baxter_s_of_q(np.array([Q_SAMPLE * DIAMETER_ANGSTROM]), t["phi"], t["tau"])[0]
        candidates = lines[lines["salt_mM"] == salt]
        nearest = candidates[np.argmin(abs(candidates["phi"] - t["phi"]))]
        published = f"{nearest['S_line']:.3f}" if abs(nearest["phi"] - t["phi"]) < 0.004 else ""
        measured = experiments[experiments["salt_mM"] == salt]
        s0_measured = measured[np.argmin(abs(measured["phi"] - t["phi"]))]["S0"]
        s0 = baxter_s0(t["phi"], t["tau"])
        rows.append(
            f"{salt},{t['phi']:.4f},{t['tau']:.3f},{at_q:.3f},{published},{s0:.3f},{s0_measured:.3f}"
        )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "baxter_comparison.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


def main() -> None:
    apply_style()
    taus = read("li2016_fig4_tau_digitized.csv")
    lines = read("li2016_fig3_baxter_lines_digitized.csv")
    write_table(taus, lines, read("li2016_fig5_digitized.csv"))

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(3.4, 4.6), layout="constrained")
    for ax, salt, ylim in zip(axes, (35, 200), ((0.5, 1.3), (0.5, 1.2))):
        for line in lines[lines["salt_mM"] == salt]:
            conc = int(line["conc_mg_mL"])
            row = tau_for(taus, line["phi"], salt)
            s = baxter_s_of_q(Q_RANGE * DIAMETER_ANGSTROM, row["phi"], row["tau"])
            ax.plot(Q_RANGE, s, color=COLOURS[salt, conc], lw=1.0)
            q_label, offset = LABELS.get((salt, conc), (0.0016, 0.02))
            s_label = baxter_s_of_q(np.array([q_label * DIAMETER_ANGSTROM]), row["phi"], row["tau"])[0]
            ax.text(q_label, s_label + offset, str(conc), color=COLOURS[salt, conc], fontsize=6)
        selected = lines[lines["salt_mM"] == salt]
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
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.text(0.97, 0.91, rf"$c_s$ = {salt} mM", transform=ax.transAxes, ha="right", fontsize=8)
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.92), fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.001, 0.399)
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axes[1].set_xlabel(r"$\mathit{q}$ ($\mathrm{\AA}^{-1}$)")
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "fig3_baxter.pdf")
    fig.savefig(out / "fig3_baxter.png", dpi=300)


if __name__ == "__main__":
    main()
