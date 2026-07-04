"""
Sphere -> prolate spheroid Rayleigh--Ritz diagnostic
=====================================================

Purpose
-------
This script prepares a compact 3D extension for the manuscript:

    SO(3) sphere  ->  SO(2) prolate spheroid
    l = 1 triplet ->  longitudinal singlet + transverse doublet

It is intentionally designed as a reviewer-response / optional-figure code.
The calculation uses a small symmetry-adapted Rayleigh--Ritz basis on the
unit ball mapped to a prolate spheroid,

    x = a u,    y = a v,    z = c w,    u^2 + v^2 + w^2 <= 1,

with Dirichlet boundary enforced by a polynomial boundary factor

    B(u,v,w) = 1 - u^2 - v^2 - w^2.

The physical gradient in the ellipsoid is

    |grad_{x,y,z} psi|^2 = (1/a^2)(psi_u^2 + psi_v^2) + (1/c^2) psi_w^2.

The script builds Ritz matrices for three symmetry sectors:

    px-like sector: odd in x, transverse
    py-like sector: odd in y, transverse
    pz-like sector: odd in z, longitudinal

For a = c, the three branches are degenerate. For c > a, the pz-like
branch separates from the px/py doublet, producing the expected
3 -> 1 + 2 splitting.

Notes
-----
1. This is not a full neural training script. It is a deterministic
   symmetry-adapted Rayleigh--Ritz diagnostic that can be used to validate
   the expected symmetry-breaking pattern before investing in the full PINN.
2. The basis is deliberately simple and robust. The same sector
   ansatz can later be replaced by a neural amplitude A_theta(r^2, z^2).
3. The absolute energies are variational approximations, but the degeneracy
   and splitting pattern are the main diagnostic.

Dependencies
------------
    numpy
    scipy
    matplotlib

Run
---
    python sphere_to_prolate_spheroid_ritz.py

Outputs
-------
    fig_sphere_to_prolate_spheroid.png
    sphere_to_prolate_spheroid_results.csv
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh


@dataclass
class QuadratureBall:
    """Tensor-product quadrature mapped to the unit ball.

    Coordinates:
        rho in [0, 1]
        mu = cos(theta) in [-1, 1]
        phi in [0, 2pi]

    Volume element:
        dV = rho^2 d rho d mu d phi
    """

    u: np.ndarray
    v: np.ndarray
    w: np.ndarray
    weight: np.ndarray


def build_unit_ball_quadrature(
    n_r: int = 34,
    n_mu: int = 34,
    n_phi: int = 72,
) -> QuadratureBall:
    """Create deterministic Gauss/trapezoidal quadrature on the unit ball."""
    # Gauss-Legendre for radial variable rho in [0, 1]
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    rho = 0.5 * (xr + 1.0)
    wrho = 0.5 * wr

    # Gauss-Legendre for mu = cos(theta) in [-1, 1]
    mu, wmu = np.polynomial.legendre.leggauss(n_mu)

    # Periodic trapezoidal quadrature for phi
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    wphi = np.full(n_phi, 2.0 * np.pi / n_phi)

    R, M, P = np.meshgrid(rho, mu, phi, indexing="ij")
    WR, WM, WP = np.meshgrid(wrho, wmu, wphi, indexing="ij")

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - M**2))
    u = R * sin_theta * np.cos(P)
    v = R * sin_theta * np.sin(P)
    w = R * M

    weight = WR * WM * WP * R**2

    return QuadratureBall(
        u=u.ravel(),
        v=v.ravel(),
        w=w.ravel(),
        weight=weight.ravel(),
    )


def monomial_powers(max_order: int) -> list[tuple[int, int, int]]:
    """Return powers (i, j, k) for monomials s^i t^j q^k.

    Here s = u^2, t = v^2, q = w^2. These monomials are even in all
    coordinates. Multiplication by u, v, or w creates px, py, or pz parity.
    """
    powers: list[tuple[int, int, int]] = []
    for total in range(max_order + 1):
        for i in range(total + 1):
            for j in range(total - i + 1):
                k = total - i - j
                powers.append((i, j, k))
    return powers


def sector_basis_and_gradients(
    quad: QuadratureBall,
    sector: str,
    max_order: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build basis functions and reference-coordinate gradients.

    Basis form:
        phi_i = B(u,v,w) * carrier * A_i(u^2, v^2, w^2)

    where carrier is u, v, or w for px, py, or pz sectors.
    """
    if sector not in {"px", "py", "pz"}:
        raise ValueError("sector must be 'px', 'py', or 'pz'")

    u, v, w = quad.u, quad.v, quad.w
    s, t, q = u**2, v**2, w**2
    B = 1.0 - s - t - q

    if sector == "px":
        carrier = u
        dc_du, dc_dv, dc_dw = np.ones_like(u), np.zeros_like(u), np.zeros_like(u)
    elif sector == "py":
        carrier = v
        dc_du, dc_dv, dc_dw = np.zeros_like(u), np.ones_like(u), np.zeros_like(u)
    else:
        carrier = w
        dc_du, dc_dv, dc_dw = np.zeros_like(u), np.zeros_like(u), np.ones_like(u)

    powers = monomial_powers(max_order)
    n_basis = len(powers)
    n_pts = u.size

    Phi = np.empty((n_basis, n_pts), dtype=float)
    Du = np.empty_like(Phi)
    Dv = np.empty_like(Phi)
    Dw = np.empty_like(Phi)

    dB_du = -2.0 * u
    dB_dv = -2.0 * v
    dB_dw = -2.0 * w

    for row, (i, j, k) in enumerate(powers):
        A = (s**i) * (t**j) * (q**k)

        # Derivatives of A(u^2, v^2, w^2)
        dA_du = np.zeros_like(u) if i == 0 else 2.0 * i * u * (s ** (i - 1)) * (t**j) * (q**k)
        dA_dv = np.zeros_like(u) if j == 0 else 2.0 * j * v * (s**i) * (t ** (j - 1)) * (q**k)
        dA_dw = np.zeros_like(u) if k == 0 else 2.0 * k * w * (s**i) * (t**j) * (q ** (k - 1))

        Phi[row] = B * carrier * A
        Du[row] = dB_du * carrier * A + B * dc_du * A + B * carrier * dA_du
        Dv[row] = dB_dv * carrier * A + B * dc_dv * A + B * carrier * dA_dv
        Dw[row] = dB_dw * carrier * A + B * dc_dw * A + B * carrier * dA_dw

    return Phi, Du, Dv, Dw


def ritz_lowest_energy(
    quad: QuadratureBall,
    aspect: float,
    sector: str,
    max_order: int = 3,
    jitter: float = 1.0e-12,
) -> float:
    """Compute the lowest Ritz eigenvalue in one symmetry sector.

    The prolate spheroid has a = 1 and c = aspect.
    The Jacobian factor a^2 c cancels in the Rayleigh quotient and in the
    generalized eigenvalue problem, so it is omitted consistently.
    """
    a = 1.0
    c = float(aspect)

    Phi, Du, Dv, Dw = sector_basis_and_gradients(quad, sector, max_order=max_order)
    wt = quad.weight

    # Mass matrix M_ij = int phi_i phi_j dV
    M = (Phi * wt) @ Phi.T

    # Stiffness matrix with ellipsoid metric
    K = ((Du / a**2) * wt) @ Du.T
    K += ((Dv / a**2) * wt) @ Dv.T
    K += ((Dw / c**2) * wt) @ Dw.T

    # Symmetrize to suppress roundoff noise
    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)

    # Tiny jitter only for numerical safety in the generalized problem
    evals = eigh(K, M + jitter * np.eye(M.shape[0]), eigvals_only=True)
    return float(np.min(evals))


def run_scan(
    aspects: np.ndarray,
    max_order: int = 3,
    n_r: int = 34,
    n_mu: int = 34,
    n_phi: int = 72,
) -> list[dict[str, float]]:
    """Scan aspect ratios and compute px, py, pz branches."""
    quad = build_unit_ball_quadrature(n_r=n_r, n_mu=n_mu, n_phi=n_phi)
    rows: list[dict[str, float]] = []

    for eta in aspects:
        e_px = ritz_lowest_energy(quad, eta, "px", max_order=max_order)
        e_py = ritz_lowest_energy(quad, eta, "py", max_order=max_order)
        e_pz = ritz_lowest_energy(quad, eta, "pz", max_order=max_order)
        e_t = 0.5 * (e_px + e_py)
        rows.append(
            {
                "aspect_c_over_a": float(eta),
                "E_px": e_px,
                "E_py": e_py,
                "E_pz": e_pz,
                "E_transverse_mean": e_t,
                "transverse_split_abs": abs(e_px - e_py),
                "longitudinal_minus_transverse": e_pz - e_t,
            }
        )
        print(
            f"eta={eta:.3f}  "
            f"E_px={e_px:.6f}  E_py={e_py:.6f}  E_pz={e_pz:.6f}  "
            f"E_z-E_t={e_pz - e_t:+.6f}"
        )

    return rows


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    """Save scan results to CSV."""
    if not rows:
        raise ValueError("No rows to save")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, float]], path: Path) -> None:
    """Create a compact optional manuscript/reviewer-response figure."""
    eta = np.array([r["aspect_c_over_a"] for r in rows])
    e_px = np.array([r["E_px"] for r in rows])
    e_py = np.array([r["E_py"] for r in rows])
    e_pz = np.array([r["E_pz"] for r in rows])
    e_t = np.array([r["E_transverse_mean"] for r in rows])
    dz = np.array([r["longitudinal_minus_transverse"] for r in rows])
    dxy = np.array([r["transverse_split_abs"] for r in rows])

    fig = plt.figure(figsize=(10.5, 4.2))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(eta, e_px, "o-", label=r"$p_x$")
    ax1.plot(eta, e_py, "s--", label=r"$p_y$")
    ax1.plot(eta, e_pz, "^-", label=r"$p_z$")
    ax1.axvline(1.0, linewidth=1.0, linestyle=":")
    ax1.set_xlabel(r"aspect ratio $\eta=c/a$")
    ax1.set_ylabel(r"Ritz energy $\epsilon$")
    ax1.set_title(r"$SO(3) \rightarrow SO(2)$ splitting")
    ax1.legend(frameon=False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(eta, dz, "o-", label=r"$E_z - (E_x+E_y)/2$")
    ax2.plot(eta, dxy, "s--", label=r"$|E_x-E_y|$")
    ax2.axhline(0.0, linewidth=1.0, linestyle=":")
    ax2.axvline(1.0, linewidth=1.0, linestyle=":")
    ax2.set_xlabel(r"aspect ratio $\eta=c/a$")
    ax2.set_ylabel(r"splitting")
    ax2.set_title(r"triplet $\rightarrow$ singlet + doublet")
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    aspects = np.linspace(0.75, 1.35, 13)

    # Increase max_order to 4 for a more accurate variational basis.
    # max_order=3 is already enough for the symmetry-breaking diagnostic.
    rows = run_scan(aspects, max_order=3, n_r=34, n_mu=34, n_phi=72)

    csv_path = Path("sphere_to_prolate_spheroid_results.csv")
    fig_path = Path("fig_sphere_to_prolate_spheroid.png")

    save_csv(rows, csv_path)
    make_figure(rows, fig_path)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {fig_path}")

    # Quick sanity check at the spherical point eta = 1.
    row_sphere = min(rows, key=lambda r: abs(r["aspect_c_over_a"] - 1.0))
    print("\nSpherical-limit check:")
    print(f"  eta = {row_sphere['aspect_c_over_a']:.6f}")
    print(f"  |E_px - E_py| = {abs(row_sphere['E_px'] - row_sphere['E_py']):.3e}")
    print(f"  |E_pz - E_transverse| = {abs(row_sphere['longitudinal_minus_transverse']):.3e}")


if __name__ == "__main__":
    main()
