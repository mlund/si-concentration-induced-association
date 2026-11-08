# Wertheim theory for one-patch spheres

Supporting code and data for:

> W. Li, B. A. Persson, M. Lund, J. Bergenholtz, M. Zackrisson Oskolkova.
> *Concentration-Induced Association in a Protein System Caused by a Highly
> Directional Patch Attraction.* J. Phys. Chem. B **2016**, 120, 8953–8959.
> <https://doi.org/10.1021/acs.jpcb.6b06873>

This is a retrospective reimplementation written in 2026, not the 2016 code. It
was not part of the peer-reviewed article. Experimental values here are digitized
from the published figures, because the original data are not included.
[NOTES.md](NOTES.md) lists what reproduces, what does not, and two misprints in the
article's equations.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uv run python scripts/fig5.py    # figures/fig5.{pdf,png}
uv run python scripts/b2.py      # results/b2_comparison.csv
uv run python scripts/baxter.py  # figures/fig3_baxter.{pdf,png}, results/baxter_comparison.csv
uv run python scripts/closures.py  # results/closure_comparison.csv
```

## Use

```python
from wertheim import Patch, double_yukawa, solve

solution = solve(phi=0.02, patch=Patch(beta_eps=14.0))
solution.s0, solution.monomer_fraction
q, s = solution.structure_factor()  # q in units of 1/D
solve(phi=0.02, patch=Patch(14.0), closure="reference-hnc")  # default: "percus-yevick"
```

## Contents

| Path | Purpose |
|---|---|
| `src/wertheim/solver.py` | Wertheim integral-equation solver and exact B2 |
| `src/wertheim/baxter.py` | Baxter adhesive-sphere structure factor |
| `scripts/` | One script per comparison with the article |
| `data/` | Values digitized from the article's figures |
| `results/`, `figures/` | Output of the scripts |
| `tests/` | Analytic limits, and regression against the article |

## Figures

**`figures/fig5`.** S(0) and monomer fraction ρ0/ρ against volume fraction φ for
one-patch spheres (d = D/2, A = 0.1D), from Wertheim theory with the
Percus–Yevick-like closure. (a) Hard spheres with a patch, at the strengths shown.
(b) Patch plus double-Yukawa isotropic potential at 35 mM (black) and 200 mM
(green). (c) Monomer fraction for the 35 mM model. Symbols are lactoferrin S(0)
digitized from Fig. 5 of the article, without error bars because none are published.

**`figures/fig3_baxter`.** Baxter adhesive-sphere S(q) with D = 6 nm, using the
stickiness digitized from Fig. 4 of the article. Labels give protein concentration
in mg/mL. Crosses mark the published model lines digitized at q = 0.00498 Å⁻¹.

## Licenses

Code is under the [MIT license](LICENSE). Data, results and figures in this
repository are under [CC BY 4.0](LICENSE-DATA). These licenses cover this
repository only and grant no rights to the published article or its figures.
