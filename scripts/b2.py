"""Compare the model's exact B2 with the measurements in Fig. 2 of Li et al. 2016."""

import numpy as np
from fig5 import DIAMETER_NM, ROOT, protein_potential

from wertheim import Patch, second_virial

CONDITIONS = {35: 11.2, 200: 0.0}


def main() -> None:
    measured = np.genfromtxt(
        ROOT / "data" / "li2016_fig2_b2_digitized.csv", delimiter=",", names=True, skip_header=3
    )
    rows = ["salt_mM,beta_eps,B2_model_nm3,B2_measured_nm3,err_measured_nm3"]
    for salt_mM, beta_eps in CONDITIONS.items():
        reduced = second_virial(Patch(beta_eps), protein_potential(salt_mM / 1000))
        row = measured[measured["salt_mM"] == salt_mM][0]
        model = reduced * DIAMETER_NM**3
        rows.append(f"{salt_mM},{beta_eps},{model:.0f},{row['B2_nm3']:.0f},{row['err_nm3']:.0f}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "b2_comparison.csv").write_text("\n".join(rows) + "\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
