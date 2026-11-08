"""Wertheim integral-equation theory for spheres with one attractive site.

Follows Li et al., J. Phys. Chem. B 2016, 120, 8953. Both closures are from
Wertheim, J. Stat. Phys. 1984, 35, 35: Percus-Yevick-like (eq. 15) and
reference-HNC (eqs. 38-40). Li et al. misprint two equations, corrected here
(A = reach, X = bonding): eq. 1 needs (2A - 2d + r), from the site-overlap
geometry, and eq. 4 needs +rho/X, from mass action.

Lengths are in units of the hard-sphere diameter D.

Correlations are scaled by the monomer fraction a = rho0/rho, so that
c01 -> a c01 and c11 -> a^2 c11 (same for gamma). This takes rho0 out of the
Ornstein-Zernike matrix and keeps a^2 exp(beta eps) finite for strong patches.
"""

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from scipy.fft import dst

Potential = Callable[[np.ndarray], np.ndarray]
Closure = Literal["percus-yevick", "reference-hnc"]


def hard_sphere(r: np.ndarray) -> np.ndarray:
    return np.zeros_like(r)


def double_yukawa(
    well: float, kappa_attr: float, bjerrum: float, charge: float, kappa_debye: float
) -> Potential:
    """Eq. 7 outside the core, as beta*u(r)."""

    def beta_u(r: np.ndarray) -> np.ndarray:
        attraction = -well * np.exp(-kappa_attr * (r - 1)) / r
        repulsion = (
            bjerrum * charge**2 * np.exp(-kappa_debye * (r - 1)) / ((1 + kappa_debye / 2) ** 2 * r)
        )
        return attraction + repulsion

    return beta_u


@dataclass(frozen=True)
class Patch:
    """Square-well site at distance d from the centre, with range `reach`."""

    beta_eps: float
    d: float = 0.5
    reach: float = 0.1

    def __post_init__(self) -> None:
        # Eq. 2: beyond this a site can bind two particles, and dimer-only theory fails.
        overlap = self.reach + 2 * self.d - 1
        if not (0 < self.d <= 0.5 and 0 < overlap < (2 - np.sqrt(3)) * self.d):
            raise ValueError(f"patch can form more than dimers, or cannot bind: {self}")

    def overlap_probability(self, r: np.ndarray) -> np.ndarray:
        """Eq. 1 without its Boltzmann prefactor."""
        d, reach = self.d, self.reach
        end = 2 * d + reach
        shape = (end - r) ** 2 * (2 * reach - 2 * d + r) / (24 * d**2 * r)
        return np.where((r >= 1) & (r < end), shape, 0.0)


class Grid:
    """Uniform radial grid with r = D on a node, so the contact jump is exact."""

    def __init__(self, n: int = 2**14, dr: float = 0.002) -> None:
        contact = round(1 / dr)
        if not np.isclose(contact * dr, 1.0):
            raise ValueError("1/dr must be an integer to place r = D on a node")
        if contact >= n:
            raise ValueError(f"grid ends at r = {(n - 1) * dr}, before contact at r = 1")
        self.dr = 1 / contact
        # Divide: multiplying by dr can land just below 1 and lose the contact node.
        self.r = np.arange(1, n) / contact
        self.contact = contact - 1
        self.dk = np.pi / (n * self.dr)
        self.k = np.arange(1, n) * self.dk

    def forward(self, f: np.ndarray) -> np.ndarray:
        return (2 * np.pi * self.dr / self.k) * dst(self.r * f, type=1)

    def inverse(self, f: np.ndarray) -> np.ndarray:
        return (self.dk / (4 * np.pi**2 * self.r)) * dst(self.k * f, type=1)

    def zero_moment(self, f: np.ndarray) -> float:
        return float(4 * np.pi * self.dr * np.sum(self.r**2 * f))


def _s_of_q(rho: float, c00, c01, c11):
    """Analytic inverse of the 2x2 problem for S = 1 + rho(h00 + 2 h01 + h11)."""
    bonded = 1 + rho * c11
    return bonded / ((1 - rho * c01) ** 2 - rho * c00 * bonded)


@dataclass
class Solution:
    phi: float
    monomer_fraction: float
    bonding_integral: float
    gamma: np.ndarray
    c: tuple[np.ndarray, np.ndarray, np.ndarray]
    grid: Grid
    iterations: int

    @property
    def rho(self) -> float:
        return 6 * self.phi / np.pi

    @property
    def s0(self) -> float:
        c00, c01, c11 = (self.grid.zero_moment(c) for c in self.c)
        return float(_s_of_q(self.rho, c00, c01, c11))

    def structure_factor(self) -> tuple[np.ndarray, np.ndarray]:
        c00, c01, c11 = (self.grid.forward(c) for c in self.c)
        return self.grid.k, _s_of_q(self.rho, c00, c01, c11)


class _Anderson:
    """Anderson acceleration; a plain damped iteration needs far more passes."""

    def __init__(self, mixing: float = 0.5, history: int = 8) -> None:
        self.mixing = mixing
        self.xs: deque[np.ndarray] = deque(maxlen=history)
        self.residuals: deque[np.ndarray] = deque(maxlen=history)

    def step(self, x: np.ndarray, residual: np.ndarray) -> np.ndarray:
        self.xs.append(x)
        self.residuals.append(residual)
        if len(self.xs) < 2:
            return x + self.mixing * residual
        dx = np.diff(np.array(self.xs), axis=0).T
        dr = np.diff(np.array(self.residuals), axis=0).T
        weights = np.linalg.lstsq(dr, residual, rcond=None)[0]
        return x + self.mixing * residual - (dx + self.mixing * dr) @ weights


def solve(
    phi: float,
    patch: Patch,
    potential: Potential = hard_sphere,
    grid: Grid | None = None,
    guess: np.ndarray | None = None,
    tol: float = 1e-9,
    max_iter: int = 3000,
    closure: Closure = "percus-yevick",
) -> Solution:
    """Solve at volume fraction phi; `guess` is a previous `Solution.gamma`."""
    if closure not in get_args(Closure):
        raise ValueError(f"unknown closure: {closure}")
    grid = grid or Grid()
    rho = 6 * phi / np.pi
    r, contact = grid.r, grid.contact
    mayer = np.expm1(patch.beta_eps)
    overlap = patch.overlap_probability(r)

    boltzmann = np.zeros_like(r)
    boltzmann[contact:] = np.exp(-potential(r[contact:]))

    gamma_reference = None
    if closure == "reference-hnc":
        # Reference: same isotropic potential, no patch, so g00 -> g_R as the patch vanishes.
        free = Patch(0.0, patch.d, patch.reach)
        gamma_reference = solve(phi, free, potential, grid, tol=tol, max_iter=max_iter).gamma[
            : r.size
        ]

    def iterate(x: np.ndarray):
        gamma00, gamma01, gamma11 = np.split(x, 3)
        # The closures differ only in these three terms.
        if gamma_reference is None:
            g00 = boltzmann * (1 + gamma00)
            dressing = boltzmann
            chain = 0.0
        else:
            g00 = boltzmann * (1 + gamma_reference) * np.exp(gamma00 - gamma_reference)
            dressing = g00
            chain = gamma01**2
        integrand = g00 * overlap * r**2
        bonding = mayer * 4 * np.pi * np.trapezoid(integrand[contact:], r[contact:])
        a = 2 / (1 + np.sqrt(1 + 4 * bonding * rho))  # eq. 4, stable as bonding -> 0

        bond = a**2 * mayer * overlap
        c00 = g00 - 1 - gamma00
        c01 = (dressing - 1) * gamma01
        c11 = (dressing - 1) * gamma11 + g00 * (bond + chain)

        # Midpoint value at the jump keeps the sine transform second order.
        c00[contact] = 0.5 * (c00[contact] - 1 - gamma00[contact])
        c01[contact] = 0.5 * (c01[contact] - gamma01[contact])
        c11[contact] = 0.5 * (c11[contact] - gamma11[contact])

        C00, C01, C11 = grid.forward(c00), grid.forward(c01), grid.forward(c11)
        m00 = 1 - rho * (C00 + C01)
        m01 = -rho * C00
        m10 = -rho * (C01 + C11)
        m11 = 1 - rho * C01
        det = m00 * m11 - m01 * m10
        H00 = (m11 * C00 - m01 * C01) / det
        H01 = (m11 * C01 - m01 * C11) / det
        H11 = (m00 * C11 - m10 * C01) / det
        updated = np.concatenate(
            [grid.inverse(H00 - C00), grid.inverse(H01 - C01), grid.inverse(H11 - C11)]
        )
        return updated, (c00, c01, c11), a, bonding

    x = np.zeros(3 * r.size) if guess is None else guess
    mixer = _Anderson()
    error = np.inf
    for iteration in range(max_iter):
        updated, c, a, bonding = iterate(x)
        residual = updated - x
        error = np.sqrt(np.mean(residual**2))
        if not np.isfinite(error):
            raise RuntimeError(f"iteration diverged at phi={phi}, {patch}")
        if error < tol:
            return Solution(phi, float(a), float(bonding), x, c, grid, iteration)
        x = mixer.step(x, residual)
    raise RuntimeError(f"no convergence at phi={phi}, {patch}: residual {error:.2e}")


def second_virial(patch: Patch, potential: Potential = hard_sphere, r_max: float = 60.0) -> float:
    """B2 in units of D^3, exact for the orientation-averaged model."""
    r = np.linspace(1.0, r_max, 1_180_001)
    boltzmann = np.exp(-potential(r)) * (
        1 + np.expm1(patch.beta_eps) * patch.overlap_probability(r)
    )
    return float(2 * np.pi / 3 - 2 * np.pi * np.trapezoid((boltzmann - 1) * r**2, r))


def sweep(
    phis: Iterable[float],
    patch: Patch,
    potential: Potential = hard_sphere,
    grid: Grid | None = None,
    closure: Closure = "percus-yevick",
) -> list[Solution]:
    """Solve along phi, starting each state from the previous one."""
    grid = grid or Grid()
    solutions: list[Solution] = []
    guess = None
    for phi in phis:
        solution = solve(phi, patch, potential, grid, guess, closure=closure)
        solutions.append(solution)
        guess = solution.gamma
    return solutions
