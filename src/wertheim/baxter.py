"""Baxter adhesive hard spheres in the Percus-Yevick approximation.

Baxter, J. Chem. Phys. 1968, 49, 2770. Wavevectors are in units of 1/D.
"""

from __future__ import annotations

import numpy as np

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(400)
_T = 0.5 * (_NODES + 1)
_W = 0.5 * _WEIGHTS


def _coefficients(phi: float, tau: float) -> tuple[float, float, float]:
    epsilon = tau + phi / (1 - phi)
    discriminant = epsilon**2 - phi * (1 + phi / 2) / (3 * (1 - phi) ** 2)
    if discriminant < 0:
        raise ValueError(f"no real Percus-Yevick solution at phi={phi}, tau={tau}")
    # Smaller root: the one that connects to hard spheres as tau grows.
    stickiness = 6 / phi * (epsilon - np.sqrt(discriminant))
    mu = stickiness * phi * (1 - phi)
    a = (1 + 2 * phi - mu) / (1 - phi) ** 2
    b = (mu - 3 * phi) / (2 * (1 - phi) ** 2)
    return stickiness, a, b


def baxter_s_of_q(q: np.ndarray, phi: float, tau: float) -> np.ndarray:
    """S(q) from Baxter's factorization, 1/S = |Q(q)|^2."""
    stickiness, a, b = _coefficients(phi, tau)
    kernel = 0.5 * a * (_T**2 - 1) + b * (_T - 1) + stickiness / 12
    # Quadrature avoids the 0/0 of the closed form as q -> 0.
    phase = np.exp(1j * np.outer(np.atleast_1d(q), _T))
    factor = 1 - 12 * phi * phase @ (_W * kernel)
    return 1 / np.abs(factor) ** 2


def baxter_s0(phi: float, tau: float) -> float:
    stickiness, _, _ = _coefficients(phi, tau)
    mu = stickiness * phi * (1 - phi)
    return float((1 - phi) ** 4 / (1 + 2 * phi - mu) ** 2)
