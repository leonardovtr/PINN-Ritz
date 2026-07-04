"""Analytical spectra and modes for ideal quantum dots.

All energies are dimensionless eigenvalues epsilon of

    -nabla^2 psi = epsilon psi,

with infinite-wall Dirichlet boundary conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import jn_zeros, jv


@dataclass(frozen=True)
class BoxState:
    nx: int
    ny: int
    epsilon: float


@dataclass(frozen=True)
class DiskState:
    m: int
    radial_index: int
    zero: float
    epsilon: float
    degeneracy: int


def infinite_well_1d_energy(n: int | np.ndarray, length: float = 1.0) -> float | np.ndarray:
    """Dimensionless 1D infinite-well energy."""
    n_array = np.asarray(n)
    epsilon = (np.pi * n_array / length) ** 2
    return float(epsilon) if epsilon.ndim == 0 else epsilon


def infinite_well_1d_wavefunction(
    x: np.ndarray,
    n: int,
    length: float = 1.0,
) -> np.ndarray:
    """Normalized 1D infinite-well eigenfunction on [0, L]."""
    return np.sqrt(2.0 / length) * np.sin(n * np.pi * x / length)


def box2d_energy(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> float:
    """Dimensionless 2D box energy."""
    return float(np.pi**2 * ((nx / lx) ** 2 + (ny / ly) ** 2))


def box2d_wavefunction(
    x: np.ndarray,
    y: np.ndarray,
    nx: int,
    ny: int,
    lx: float = 1.0,
    ly: float = 1.0,
) -> np.ndarray:
    """Normalized 2D box eigenfunction on [0, Lx] x [0, Ly]."""
    return (
        2.0
        / np.sqrt(lx * ly)
        * np.sin(nx * np.pi * x / lx)
        * np.sin(ny * np.pi * y / ly)
    )


def box2d_spectrum(
    max_n: int,
    lx: float = 1.0,
    ly: float = 1.0,
    sort: bool = True,
) -> list[BoxState]:
    """Return box states up to max_n in each direction."""
    states = [
        BoxState(nx, ny, box2d_energy(nx, ny, lx=lx, ly=ly))
        for nx in range(1, max_n + 1)
        for ny in range(1, max_n + 1)
    ]
    if sort:
        states.sort(key=lambda state: (state.epsilon, state.nx, state.ny))
    return states


def disk_spectrum(max_m: int, max_radial: int, radius: float = 1.0) -> list[DiskState]:
    """Return disk eigenvalues from Bessel zeros.

    The real disk modes are nondegenerate for m = 0 and twofold degenerate for
    m > 0, corresponding to cos(m theta) and sin(m theta), or equivalently
    complex modes with angular momenta +m and -m.
    """
    states: list[DiskState] = []
    for m in range(max_m + 1):
        zeros = jn_zeros(m, max_radial)
        for radial_index, zero in enumerate(zeros, start=1):
            states.append(
                DiskState(
                    m=m,
                    radial_index=radial_index,
                    zero=float(zero),
                    epsilon=float((zero / radius) ** 2),
                    degeneracy=1 if m == 0 else 2,
                )
            )
    states.sort(key=lambda state: (state.epsilon, state.m, state.radial_index))
    return states


def disk_real_mode(
    x: np.ndarray,
    y: np.ndarray,
    m: int,
    radial_index: int,
    angular: str = "cos",
    radius: float = 1.0,
) -> np.ndarray:
    """Unnormalized real disk mode, masked outside radius."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    zero = jn_zeros(m, radial_index)[-1]
    radial = jv(m, zero * r / radius)
    if m == 0:
        angular_part = np.ones_like(theta)
    elif angular == "sin":
        angular_part = np.sin(m * theta)
    else:
        angular_part = np.cos(m * theta)
    mode = radial * angular_part
    return np.where(r <= radius, mode, np.nan)


def find_degenerate_groups(
    states: list[BoxState] | list[DiskState],
    tolerance: float = 1e-10,
) -> list[list[BoxState] | list[DiskState]]:
    """Group sorted states whose energies are equal within a tolerance."""
    if not states:
        return []

    sorted_states = sorted(states, key=lambda state: state.epsilon)
    groups: list[list[BoxState] | list[DiskState]] = []
    current = [sorted_states[0]]
    reference = sorted_states[0].epsilon

    for state in sorted_states[1:]:
        if abs(state.epsilon - reference) <= tolerance:
            current.append(state)
        else:
            groups.append(current)
            current = [state]
            reference = state.epsilon
    groups.append(current)
    return groups
