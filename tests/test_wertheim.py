import numpy as np
import pytest

from wertheim import (
    Grid,
    Patch,
    baxter_s0,
    baxter_s_of_q,
    double_yukawa,
    hard_sphere,
    second_virial,
    solve,
)

PATCH_FREE = Patch(0.0)


def percus_yevick_s0(phi: float) -> float:
    return (1 - phi) ** 4 / (1 + 2 * phi) ** 2


@pytest.mark.parametrize("phi", [0.1, 0.3])
def test_no_patch_gives_percus_yevick(phi):
    solution = solve(phi, PATCH_FREE)
    assert solution.monomer_fraction == 1.0
    assert solution.s0 == pytest.approx(percus_yevick_s0(phi), abs=1e-4)


def test_strong_patch_at_high_dilution_is_ideal_dimer_gas():
    solution = solve(1e-5, Patch(30.0))
    assert solution.monomer_fraction < 0.01
    assert solution.s0 == pytest.approx(2 - solution.monomer_fraction, abs=0.01)


@pytest.mark.parametrize(
    "patch, potential",
    [
        (Patch(0.0), hard_sphere),
        (Patch(11.2), double_yukawa(1.0, 20.0, 0.7 / 6, 2.0, 3.69)),
        (Patch(12.5), hard_sphere),
    ],
)
def test_dilute_s0_follows_exact_virial_coefficient(patch, potential):
    phi = 1e-4
    rho = 6 * phi / np.pi
    expected = 1 - 2 * second_virial(patch, potential) * rho
    assert solve(phi, patch, potential).s0 == pytest.approx(expected, abs=2e-4)


def test_s0_converges_with_grid():
    patch = Patch(14.0)
    coarse = solve(0.01, patch, grid=Grid(2**14, 0.002)).s0
    fine = solve(0.01, patch, grid=Grid(2**15, 0.001)).s0
    assert coarse == pytest.approx(fine, abs=5e-4)


def test_small_q_structure_factor_matches_s0():
    solution = solve(0.05, Patch(14.0))
    _, s = solution.structure_factor()
    assert s[0] == pytest.approx(solution.s0, abs=1e-3)


def test_eq1_correction_matches_monte_carlo():
    """Guards the printed typo: (2A + 2d + r) is about 11x too large at contact."""
    rng = np.random.default_rng(1)
    n = 4_000_000
    patch = Patch(1.0)
    u1, u2 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    u1 /= np.linalg.norm(u1, axis=1, keepdims=True)
    u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
    separation = np.array([0.0, 0.0, 1.0]) + patch.d * (u2 - u1)
    hits = np.mean(np.linalg.norm(separation, axis=1) < patch.reach)
    expected = float(patch.overlap_probability(np.array([1.0]))[0])
    assert hits == pytest.approx(expected, rel=0.1)


def test_patch_that_allows_trimers_is_rejected():
    with pytest.raises(ValueError):
        Patch(10.0, d=0.5, reach=0.2)


def test_figure_5_top_panel_value():
    assert solve(0.01, Patch(14.0)).s0 == pytest.approx(1.451, abs=0.003)


def test_nonconvergence_raises_instead_of_returning():
    with pytest.raises(RuntimeError):
        solve(0.1, Patch(14.0), max_iter=1)


def test_reference_hnc_without_patch_recovers_its_reference():
    potential = double_yukawa(1.0, 20.0, 0.7 / 6, 2.0, 3.69)
    reference = solve(0.1, PATCH_FREE, potential)
    solution = solve(0.1, PATCH_FREE, potential, closure="reference-hnc")
    assert solution.s0 == pytest.approx(reference.s0, abs=1e-6)
    assert solution.monomer_fraction == 1.0


@pytest.mark.parametrize("beta_eps", [11.2, 14.0])
def test_reference_hnc_dilute_limit_follows_virial_coefficient(beta_eps):
    # Large |B2|: the linear regime needs far lower density than for the free-patch case.
    phi = 1e-6
    patch = Patch(beta_eps)
    expected_excess = -2 * second_virial(patch) * 6 * phi / np.pi
    excess = solve(phi, patch, closure="reference-hnc").s0 - 1
    assert excess == pytest.approx(expected_excess, rel=2e-3)


def test_closures_agree_at_low_volume_fraction():
    patch = Patch(14.0)
    percus_yevick = solve(0.01, patch).s0
    assert solve(0.01, patch, closure="reference-hnc").s0 == pytest.approx(percus_yevick, abs=2e-3)


def test_unknown_closure_is_rejected():
    with pytest.raises(ValueError):
        solve(0.01, PATCH_FREE, closure="hnc")


def sticky_limit(phi: float, bond_volume: float) -> tuple[float, float]:
    """Monomer fraction and d(beta p)/d(rho): Wertheim, JCP 1986, 85, 2929, eqs. 90 and 96."""

    def monomer_fraction(rho):
        eta = np.pi * rho / 6
        return 2 / (1 + np.sqrt(1 + 4 * rho * bond_volume * (1 + eta / 2) / (1 - eta) ** 2))

    def pressure(rho):
        eta = np.pi * rho / 6
        hard_sphere_term = (1 + eta + eta**2) / (1 - eta) ** 3
        bonding_term = (1 + eta / 2) * (1 - monomer_fraction(rho)) / (2 * (1 - eta) ** 2)
        return rho * (hard_sphere_term - bonding_term)

    rho, step = 6 * phi / np.pi, 1e-5
    return monomer_fraction(rho), (pressure(rho * (1 + step)) - pressure(rho * (1 - step))) / (
        2 * rho * step
    )


def finite_range_state(phi: float, bond_volume: float, reach: float, dr: float, n: int):
    """Monomer fraction and compressibility from 1984 eq. 18, using only c(r) and rho0(rho)."""
    shell = np.arange(1.0, 1.0 + reach, dr / 20)
    shape = Patch(1.0, 0.5, reach).overlap_probability(shell) * shell**2
    patch = Patch(np.log1p(bond_volume / (4 * np.pi * np.trapezoid(shape, shell))), 0.5, reach)
    grid = Grid(n, dr)
    state = solve(phi, patch, grid=grid)
    above = solve(phi * 1.001, patch, grid=grid, guess=state.gamma)
    below = solve(phi * 0.999, patch, grid=grid, guess=state.gamma)
    slope = (above.monomer_fraction * above.rho - below.monomer_fraction * below.rho) / (
        above.rho - below.rho
    )
    c00, c01, c11 = (grid.zero_moment(c) for c in state.c)
    a = state.monomer_fraction
    return a, 1 - state.rho * (c00 + c01) - state.rho / a * (c01 + c11) * slope


@pytest.mark.parametrize("phi, bond_volume", [(0.1, 20.0), (0.1, 200.0), (0.3, 10.0)])
def test_sticky_limit_matches_wertheim_analytic_equation_of_state(phi, bond_volume):
    coarse = finite_range_state(phi, bond_volume, 0.02, 0.0005, 2**16)
    fine = finite_range_state(phi, bond_volume, 0.01, 0.00025, 2**17)
    extrapolated = [2 * f - c for c, f in zip(coarse, fine)]  # error is linear in range
    expected = sticky_limit(phi, bond_volume)
    assert extrapolated[0] == pytest.approx(expected[0], abs=2e-4)
    assert extrapolated[1] == pytest.approx(expected[1], rel=2e-4)


def test_baxter_reduces_to_percus_yevick_without_stickiness():
    assert baxter_s0(0.2, 1e9) == pytest.approx(percus_yevick_s0(0.2), abs=1e-6)


def test_baxter_dilute_limit_follows_virial_coefficient():
    phi, tau = 1e-5, 0.5
    # B2 / (2 pi D^3 / 3) = 1 - 1/(4 tau)
    assert baxter_s0(phi, tau) == pytest.approx(1 - 8 * phi * (1 - 1 / (4 * tau)), abs=1e-8)


def test_baxter_s_of_q_approaches_s0():
    assert baxter_s_of_q(np.array([1e-5]), 0.05, 0.3)[0] == pytest.approx(
        baxter_s0(0.05, 0.3), rel=1e-8
    )


def test_baxter_rejects_state_without_real_solution():
    with pytest.raises(ValueError):
        baxter_s0(0.1, 0.0)
