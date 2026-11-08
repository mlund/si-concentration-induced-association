"""Compare the model's exact B2 with the measurements in Fig. 2 of Li et al. 2016."""

from common import DIAMETER_NM, FITTED_BETA_EPS, protein_potential, read_digitized, write_results

from wertheim import Patch, second_virial


def main() -> None:
    measured = read_digitized("li2016_fig2_b2_digitized.csv")
    rows = ["salt_mM,beta_eps,B2_model_nm3,B2_measured_nm3,err_measured_nm3"]
    for salt_mM, beta_eps in FITTED_BETA_EPS.items():
        reduced = second_virial(Patch(beta_eps), protein_potential(salt_mM))
        row = measured[measured["salt_mM"] == salt_mM][0]
        model = reduced * DIAMETER_NM**3
        rows.append(f"{salt_mM},{beta_eps},{model:.0f},{row['B2_nm3']:.0f},{row['err_nm3']:.0f}")
    write_results("b2_comparison.csv", rows)
    print("\n".join(rows))


if __name__ == "__main__":
    main()
