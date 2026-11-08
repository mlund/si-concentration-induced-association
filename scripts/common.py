"""Model inputs and file handling shared by the comparison scripts.

The paper gives Z_eff = 2, D = 6 nm, kappa_a = 20/D and K = 1 kT, but not the
Bjerrum length or Debye-length formula. Water at 25 C is assumed for both.

No matplotlib here, so numeric-only scripts don't load it.
"""

from pathlib import Path

import numpy as np

from wertheim import Potential, double_yukawa

ROOT = Path(__file__).resolve().parents[1]
DIAMETER_NM = 6.0
BJERRUM_NM = 0.7
# Patch strength the paper chose for each salt concentration (mM).
FITTED_BETA_EPS = {35: 11.2, 200: 0.0}


def read_digitized(name: str) -> np.ndarray:
    # skip_header: genfromtxt would otherwise take column names from the first comment.
    return np.genfromtxt(ROOT / "data" / name, delimiter=",", names=True, skip_header=3)


def write_results(name: str, rows: list[str]) -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / name).write_text("\n".join(rows) + "\n")


def protein_potential(salt_mM: int) -> Potential:
    return double_yukawa(
        well=1.0,
        kappa_attr=20.0,
        bjerrum=BJERRUM_NM / DIAMETER_NM,
        charge=2.0,
        kappa_debye=DIAMETER_NM * np.sqrt(salt_mM / 1000) / 0.304,
    )
