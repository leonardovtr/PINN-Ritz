#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sphere-to-prolate-spheroid Rayleigh--Ritz diagnostic.

Purpose
-------
Prepare a 3D "reviewer-response" extension for the PINN/Rayleigh--Ritz paper.

The script demonstrates the spectral-object principle in 3D:
    sphere:      SO(3) symmetry, l=1 triplet
    prolate:     SO(2) residual symmetry, triplet -> singlet + doublet

We use a compact Rayleigh--Ritz calculation in the l=1 spherical subspace of
the unit sphere. The prolate spheroid is represented by the volume-preserving
map

    x = a u,  y = a v,  z = c w,
    a = eta^{-1/3},  c = eta^{2/3},

so that eta = c/a is the aspect ratio and a^2 c = 1.

The trial functions are the three real l=1 modes of the unit sphere,

    phi_x = j_1(alpha r) x/r,
    phi_y = j_1(alpha r) y/r,
    phi_z = j_1(alpha r) z/r,

where alpha is the first zero of j_1. At eta = 1, the three modes are
degenerate. Under prolate deformation, phi_z separates from the x/y doublet.

Outputs
-------
    data/sphere_prolate_splitting.csv
    figures/sphere_prolate_splitting.pdf
    figures/sphere_prolate_splitting.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

try:
    from scipy.special import spherical_jn
    from scipy.linalg import eigh
except ImportError as exc:
    raise SystemExit(
        "This script requires scipy. Install with: pip install scipy"
    ) from exc


ALPHA_L1_FIRST_ZERO = 4.493409457909064  # first zero of spherical_jn(1, x)


def setup_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def quadrature_unit_ball(nr: int, nmu: int, nphi: int):
    """
    Deterministic tensor-product quadrature on the unit ball.

    Coordinates:
        r in [0,1]
        mu = cos(theta) in [-1,1]
        phi in [0,2pi)

    Volume element:
        r^2 dr dmu dphi
    """
    gr, wr = np.polynomial.legendre.leggauss(nr)
    r = 0.5 * (gr + 1.0)
    wr = 0.5 * wr

    mu, wmu = np.polynomial.legendre.leggauss(nmu)

    # Uniform trapezoidal rule in phi is spectrally accurate for periodic smooth functions.
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    wphi = np.full_like(phi, 2.0 * np.pi / nphi)

    R, MU, PHI = np.meshgrid(r, mu, phi, indexing="ij")
    WR, WMU, WPHI = np.meshgrid(wr, wmu, wphi, indexing="ij")

    sintheta = np.sqrt(np.maximum(0.0, 1.0 - MU**2))
    u = R * sintheta * np.cos(PHI)
    v = R * sintheta * np.sin(PHI)
    w = R * MU

    weight = (WR * WMU * WPHI) * R**2

    pts = np.column_stack([u.ravel(), v.ravel(), w.ravel()])
    weights = weight.ravel()

    return pts, weights


def l1_basis_and_gradients(pts: np.ndarray):
    """
    Compute phi_x, phi_y, phi_z and their gradients on the unit ball.

    phi_i = f(r) * coord_i, where f(r) = j_1(alpha r)/r.

    Gradient:
        grad phi_i = f e_i + coord_i f'(r) r_vec/r

    Near r=0:
        j_1(alpha r)/r -> alpha/3
        f'(0) -> 0
    """
    alpha = ALPHA_L1_FIRST_ZERO

    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    r = np.sqrt(x*x + y*y + z*z)

    small = r < 1e-10
    ar = alpha * r

    j1 = spherical_jn(1, ar)
    j1p = spherical_jn(1, ar, derivative=True)

    f = np.empty_like(r)
    fp = np.empty_like(r)

    f[~small] = j1[~small] / r[~small]
    fp[~small] = (alpha * j1p[~small] * r[~small] - j1[~small]) / (r[~small] ** 2)

    f[small] = alpha / 3.0
    fp[small] = 0.0

    coords = np.column_stack([x, y, z])
    Phi = f[:, None] * coords

    Grad = np.zeros((pts.shape[0], 3, 3), dtype=float)
    # Grad[:, i, k] = partial phi_i / partial coord_k
    for i in range(3):
        for k in range(3):
            delta = 1.0 if i == k else 0.0
            radial_factor = np.zeros_like(r)
            radial_factor[~small] = coords[~small, i] * fp[~small] * coords[~small, k] / r[~small]
            Grad[:, i, k] = f * delta + radial_factor

    return Phi, Grad


def ritz_l1_for_eta(eta: float, pts: np.ndarray, weights: np.ndarray):
    """
    Compute 3x3 Ritz problem for volume-preserving prolate spheroid.

    Physical map:
        X = a u, Y = a v, Z = c w

    Gradient metric:
        |grad_XYZ phi|^2 =
            (1/a^2)(phi_u^2 + phi_v^2) + (1/c^2) phi_w^2

    Jacobian:
        J = a^2 c = 1 by construction.
    """
    a = eta ** (-1.0 / 3.0)
    c = eta ** (2.0 / 3.0)
    jac = a * a * c

    Phi, Grad = l1_basis_and_gradients(pts)

    M = Phi.T @ (weights[:, None] * jac * Phi)

    metric = np.array([1.0 / (a*a), 1.0 / (a*a), 1.0 / (c*c)])
    K = np.zeros((3, 3), dtype=float)

    for i in range(3):
        for j in range(3):
            integrand = (
                metric[0] * Grad[:, i, 0] * Grad[:, j, 0]
                + metric[1] * Grad[:, i, 1] * Grad[:, j, 1]
                + metric[2] * Grad[:, i, 2] * Grad[:, j, 2]
            )
            K[i, j] = np.sum(weights * jac * integrand)

    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)

    evals, evecs = eigh(K, M)

    # Character of each Ritz vector in the original {x,y,z} l=1 basis.
    # Since M is not exactly identity, use squared components normalized.
    chars = []
    for k in range(3):
        vec = evecs[:, k]
        comp2 = vec**2 / np.sum(vec**2)
        chars.append(comp2)

    return evals, np.array(chars), a, c, K, M


def make_figure(df: pd.DataFrame, fig_dir: Path):
    setup_publication_style()

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax1, ax2 = axs

    # Energies.
    ax1.plot(df["eta"], df["E_xy_1"], "o-", ms=4.2, lw=1.3, label=r"$x/y$ doublet")
    ax1.plot(df["eta"], df["E_xy_2"], "o-", ms=4.2, lw=1.3, color="C0", alpha=0.45)
    ax1.plot(df["eta"], df["E_z"], "s-", ms=4.0, lw=1.3, label=r"$z$-like singlet")
    ax1.axvline(1.0, color="0.55", ls=":", lw=1.0)
    ax1.set_xlabel(r"aspect ratio $\eta=c/a$")
    ax1.set_ylabel(r"Ritz eigenvalue $\epsilon$")
    ax1.set_title(r"$SO(3)\rightarrow SO(2)$ splitting")
    ax1.legend(frameon=False, loc="best")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.text(0.04, 0.93, "(a)", transform=ax1.transAxes, fontweight="bold", va="top")

    # Signed splitting.
    ax2.plot(df["eta"], df["Delta_z_minus_xy"], "o-", ms=4.2, lw=1.3)
    ax2.axhline(0.0, color="0.35", lw=0.9)
    ax2.axvline(1.0, color="0.55", ls=":", lw=1.0)
    ax2.set_xlabel(r"aspect ratio $\eta=c/a$")
    ax2.set_ylabel(r"$\Delta=E_z-\langle E_{x,y}\rangle$")
    ax2.set_title(r"$l=1$ triplet $\rightarrow 1+2$")
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.text(0.04, 0.93, "(b)", transform=ax2.transAxes, fontweight="bold", va="top")

    # Annotation.
    ax2.annotate(
        "sphere",
        xy=(1.0, 0.0),
        xytext=(1.04, 0.08 * max(abs(df["Delta_z_minus_xy"]))),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.25"),
        fontsize=8,
    )

    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.88, wspace=0.28)

    pdf = fig_dir / "sphere_prolate_splitting.pdf"
    png = fig_dir / "sphere_prolate_splitting.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)

    return pdf, png


def parse_args():
    p = argparse.ArgumentParser(description="Sphere-to-prolate-spheroid Rayleigh--Ritz splitting.")
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--eta-min", type=float, default=0.80)
    p.add_argument("--eta-max", type=float, default=1.40)
    p.add_argument("--n-eta", type=int, default=13)
    p.add_argument("--nr", type=int, default=26)
    p.add_argument("--nmu", type=int, default=34)
    p.add_argument("--nphi", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()

    root = Path(args.root).resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    pts, weights = quadrature_unit_ball(args.nr, args.nmu, args.nphi)

    etas = np.linspace(args.eta_min, args.eta_max, args.n_eta)

    rows = []
    for eta in etas:
        evals, chars, a, c, K, M = ritz_l1_for_eta(eta, pts, weights)

        # Classify modes by largest z character.
        z_char = chars[:, 2]
        z_idx = int(np.argmax(z_char))
        xy_idx = [i for i in range(3) if i != z_idx]

        E_z = float(evals[z_idx])
        E_xy_1 = float(evals[xy_idx[0]])
        E_xy_2 = float(evals[xy_idx[1]])
        E_xy_mean = 0.5 * (E_xy_1 + E_xy_2)

        rows.append({
            "eta": eta,
            "a": a,
            "c": c,
            "E_1": float(evals[0]),
            "E_2": float(evals[1]),
            "E_3": float(evals[2]),
            "E_z": E_z,
            "E_xy_1": E_xy_1,
            "E_xy_2": E_xy_2,
            "E_xy_mean": E_xy_mean,
            "Delta_z_minus_xy": E_z - E_xy_mean,
            "z_character_selected": float(z_char[z_idx]),
            "xy_doublet_splitting": abs(E_xy_1 - E_xy_2),
            "alpha_l1_squared_exact_sphere": ALPHA_L1_FIRST_ZERO**2,
        })

        print(
            f"eta={eta:.3f}  a={a:.4f} c={c:.4f}  "
            f"E_xy=({E_xy_1:.6f},{E_xy_2:.6f})  "
            f"E_z={E_z:.6f}  Delta={E_z - E_xy_mean:.6f}"
        )

    df = pd.DataFrame(rows)
    csv_path = data_dir / "sphere_prolate_splitting.csv"
    df.to_csv(csv_path, index=False)

    pdf, png = make_figure(df, fig_dir)

    print("\nSaved outputs:")
    print(f"  {csv_path.relative_to(root)}")
    print(f"  {pdf.relative_to(root)}")
    print(f"  {png.relative_to(root)}")

    print("\nCompact summary:")
    print(df[["eta", "E_xy_mean", "E_z", "Delta_z_minus_xy", "xy_doublet_splitting", "z_character_selected"]].to_string(index=False))


if __name__ == "__main__":
    main()
