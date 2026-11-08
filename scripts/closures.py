"""Test the claim in Li et al. 2016 that both Wertheim closures agree for phi < 0.15."""

import numpy as np
from fig5 import ROOT, protein_potential

from wertheim import Grid, Patch, hard_sphere, sweep

PHIS = np.array([0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.1, 0.15])
CASES = [
    ("hard spheres", beta_eps, hard_sphere) for beta_eps in (25.0, 15.0, 14.0, 12.5, 10.5)
] + [
    ("35 mM", 11.2, protein_potential(0.035)),
    ("200 mM", 0.0, protein_potential(0.2)),
]


def main() -> None:
    grid = Grid()
    rows = ["model,beta_eps,phi,S0_percus_yevick,S0_reference_hnc,a_percus_yevick,a_reference_hnc"]
    for name, beta_eps, potential in CASES:
        patch = Patch(beta_eps)
        py = sweep(PHIS, patch, potential, grid)
        rhnc = sweep(PHIS, patch, potential, grid, closure="reference-hnc")
        for phi, p, h in zip(PHIS, py, rhnc):
            rows.append(
                f"{name},{beta_eps},{phi},{p.s0:.4f},{h.s0:.4f},"
                f"{p.monomer_fraction:.4f},{h.monomer_fraction:.4f}"
            )
        differences = [abs(p.s0 - h.s0) for p, h in zip(py, rhnc)]
        worst = int(np.argmax(differences))
        print(
            f"{name:13s} be={beta_eps:<5g} max|dS0|={differences[worst]:.4f} "
            f"at phi={PHIS[worst]:g}  peak S0: PY {max(p.s0 for p in py):.3f}, "
            f"RHNC {max(h.s0 for h in rhnc):.3f}"
        )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "closure_comparison.csv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
