#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pinn_ellipse_2d_splitting.py

PINN/Rayleigh--Ritz subspace solver for the 2D infinite elliptical well.

Purpose
-------
This script extends the circular-disk result to the ellipse.  The physical
question is the rotational analogue of square -> rectangle:

    disk (a=b)     : rotational symmetry, m=1 doublet is degenerate
    ellipse (a!=b) : rotational symmetry is broken, m=1 doublet splits

The network is NOT trained against exact eigenfunctions or exact eigenvalues.
It minimizes a Rayleigh--Ritz subspace objective with a boundary-satisfying
ansatz.  Exact disk/Bessel modes and finite-difference eigenvalues are used
only for validation and interpretation.

Recommended location
--------------------
    PINN/scripts/train_pinn_ellipse_2d_splitting.py

Typical command from PINN/scripts:
    python train_pinn_ellipse_2d_splitting.py

Typical command from the project root:
    python scripts/train_pinn_ellipse_2d_splitting.py

A more robust but slower command:
    python train_pinn_ellipse_2d_splitting.py \
        --etas 0.80,0.90,1.00,1.10,1.20 \
        --states 3 --restarts 3 --epochs 1600 \
        --n-r 24 --n-theta 96 --hidden-width 80 \
        --hidden-layers 3 --fourier-features 8 --dtype float64

Outputs
-------
    data/pinn_ellipse_2d_splitting.csv
    data/pinn_ellipse_2d_ranked_spectrum.csv
    data/pinn_ellipse_2d_m1_overlaps.csv
    data/pinn_ellipse_2d_modes_grid.csv
    figures/pinn_ellipse_2d_splitting.png/pdf
    figures/pinn_ellipse_2d_splitting_delta.png/pdf
    figures/pinn_ellipse_2d_modes_examples.png/pdf
    figures/pinn_ellipse_2d_m1_weights.png/pdf

Notes
-----
The ellipse is parameterized as

    x = a u,  y = b v,   u^2 + v^2 <= 1,

with a=1 and b=eta by default.  The boundary condition psi=0 on the
ellipse is imposed strongly through

    psi(u,v) = (1 - u^2 - v^2) * NN(u,v).

The Rayleigh quotient in physical coordinates is

    epsilon[psi] =
        integral_E (|dpsi/dx|^2 + |dpsi/dy|^2) dA
        ------------------------------------------------
        integral_E |psi|^2 dA

After mapping from the unit disk,

    dpsi/dx = (1/a) dpsi/du,
    dpsi/dy = (1/b) dpsi/dv.

The common Jacobian factor a*b cancels in the Rayleigh quotient, but the
metric factors 1/a^2 and 1/b^2 remain and generate the elliptical splitting.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from scipy.special import jn_zeros, jv
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

def setup_publication_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.75,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
        "image.cmap": "RdBu_r",
    })


def save_figure(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=600)
    fig.savefig(path_no_ext.with_suffix(".pdf"))
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = 0.025, y: float = 0.965) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        zorder=10,
    )


# -----------------------------------------------------------------------------
# Paths and arguments
# -----------------------------------------------------------------------------

def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    if here.parent.name == "scripts":
        return here.parent.parent
    return Path.cwd().resolve()


def parse_etas(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in ["float64", "double"]:
        return torch.float64
    if name in ["float32", "single"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


# -----------------------------------------------------------------------------
# Quadrature on the unit disk
# -----------------------------------------------------------------------------

@dataclass
class DiskQuadrature:
    uv: torch.Tensor      # (N, 2), unit disk coordinates
    w: torch.Tensor       # (N,), weights for integral over unit disk
    r: torch.Tensor       # (N,)
    theta: torch.Tensor   # (N,)


def make_disk_quadrature(
    n_r: int,
    n_theta: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> DiskQuadrature:
    """Tensor-product Gauss-Legendre in r and uniform theta.

    Integral over unit disk:
        int_0^{2pi} int_0^1 f(r,theta) r dr dtheta

    We use Gauss-Legendre on r in [0,1] and trapezoidal/uniform theta.
    """
    xr, wr = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * (xr + 1.0)
    wr = 0.5 * wr

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    wtheta = (2.0 * np.pi) / n_theta

    rr, tt = np.meshgrid(r, theta, indexing="ij")
    ww = (wr[:, None] * wtheta) * rr

    u = rr * np.cos(tt)
    v = rr * np.sin(tt)

    uv = np.column_stack([u.ravel(), v.ravel()])
    weights = ww.ravel()
    radii = rr.ravel()
    angles = tt.ravel()

    uv_t = torch.tensor(uv, device=device, dtype=dtype)
    uv_t.requires_grad_(requires_grad)

    return DiskQuadrature(
        uv=uv_t,
        w=torch.tensor(weights, device=device, dtype=dtype),
        r=torch.tensor(radii, device=device, dtype=dtype),
        theta=torch.tensor(angles, device=device, dtype=dtype),
    )


# -----------------------------------------------------------------------------
# Neural basis
# -----------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = int(n_features)

    @property
    def out_dim(self) -> int:
        # raw u,v plus sin/cos for each k and each coordinate
        return 2 + 4 * self.n_features

    def forward(self, uv: torch.Tensor) -> torch.Tensor:
        if self.n_features <= 0:
            return uv
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        feats = [uv]
        for k in range(1, self.n_features + 1):
            kk = math.pi * k
            feats.extend([
                torch.sin(kk * u), torch.cos(kk * u),
                torch.sin(kk * v), torch.cos(kk * v),
            ])
        return torch.cat(feats, dim=1)


class SubspaceNet(nn.Module):
    def __init__(
        self,
        n_outputs: int,
        hidden_width: int = 64,
        hidden_layers: int = 3,
        fourier_features: int = 6,
    ):
        super().__init__()
        self.features = FourierFeatures(fourier_features)
        layers: List[nn.Module] = []
        in_dim = self.features.out_dim
        layers.append(nn.Linear(in_dim, hidden_width))
        layers.append(nn.Tanh())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_width, hidden_width))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_width, n_outputs))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_raw(self, uv: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(uv))

    def forward_basis(self, uv: torch.Tensor) -> torch.Tensor:
        """Boundary-satisfying basis functions on the unit disk."""
        r2 = uv[:, 0:1] ** 2 + uv[:, 1:2] ** 2
        envelope = 1.0 - r2
        return envelope * self.forward_raw(uv)


# -----------------------------------------------------------------------------
# Rayleigh--Ritz matrices
# -----------------------------------------------------------------------------

@dataclass
class RitzResult:
    eigvals: torch.Tensor       # (K,)
    coeffs: torch.Tensor        # raw basis -> Ritz coefficients, (K,K)
    basis: torch.Tensor         # raw basis values, (N,K)
    modes: torch.Tensor         # Ritz mode values, (N,K)
    mass: torch.Tensor          # raw mass matrix
    stiffness: torch.Tensor     # raw stiffness matrix


def gradients_of_basis(basis: torch.Tensor, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return dphi/du and dphi/dv for all basis columns."""
    grads_u = []
    grads_v = []
    for j in range(basis.shape[1]):
        grad = torch.autograd.grad(
            basis[:, j].sum(),
            uv,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grads_u.append(grad[:, 0])
        grads_v.append(grad[:, 1])
    return torch.stack(grads_u, dim=1), torch.stack(grads_v, dim=1)


def weighted_gram(A: torch.Tensor, B: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return A.T @ (w[:, None] * B)


def compute_ritz(
    model: SubspaceNet,
    quad: DiskQuadrature,
    a: float,
    b: float,
    jitter: float = 1e-9,
) -> RitzResult:
    """Compute generalized Ritz eigenvalues K c = eps M c."""
    uv = quad.uv
    w = quad.w

    basis = model.forward_basis(uv)
    du, dv = gradients_of_basis(basis, uv)

    # Physical gradients after x=a*u, y=b*v.
    gx = du / float(a)
    gy = dv / float(b)

    M = weighted_gram(basis, basis, w)
    K = weighted_gram(gx, gx, w) + weighted_gram(gy, gy, w)

    M = 0.5 * (M + M.T)
    K = 0.5 * (K + K.T)

    n = M.shape[0]
    I = torch.eye(n, dtype=M.dtype, device=M.device)

    # Stabilize the Cholesky if the basis is temporarily close to singular.
    Mj = M + jitter * I
    try:
        L = torch.linalg.cholesky(Mj)
    except RuntimeError:
        Mj = M + (100.0 * jitter) * I
        L = torch.linalg.cholesky(Mj)

    invL = torch.linalg.solve_triangular(L, I, upper=False)
    A = invL @ K @ invL.T
    A = 0.5 * (A + A.T)

    eigvals, V = torch.linalg.eigh(A)
    coeffs = torch.linalg.solve_triangular(L.T, V, upper=True)
    modes = basis @ coeffs

    return RitzResult(eigvals=eigvals, coeffs=coeffs, basis=basis, modes=modes, mass=M, stiffness=K)


def orthonormality_error(modes: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    G = weighted_gram(modes, modes, w)
    I = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return torch.mean((G - I) ** 2)


# -----------------------------------------------------------------------------
# Reference disk modes for interpretation
# -----------------------------------------------------------------------------

def disk_reference_modes_numpy(u: np.ndarray, v: np.ndarray) -> Dict[str, np.ndarray]:
    """Canonical disk modes on the mapped unit disk.

    These are used only for interpretation.  In the ellipse, they are not exact
    eigenfunctions, but the m=1 pair remains a useful diagnostic of how the
    circular doublet splits into x-like and y-like branches.
    """
    rho = np.sqrt(u * u + v * v)
    theta = np.arctan2(v, u)
    rho_safe = np.maximum(rho, 1e-14)

    a01 = jn_zeros(0, 1)[0]
    a11 = jn_zeros(1, 1)[0]
    a21 = jn_zeros(2, 1)[0]

    out = {}
    out["m0_1"] = jv(0, a01 * rho)
    out["m1_c"] = jv(1, a11 * rho) * np.cos(theta)
    out["m1_s"] = jv(1, a11 * rho) * np.sin(theta)
    out["m2_c"] = jv(2, a21 * rho) * np.cos(2.0 * theta)
    out["m2_s"] = jv(2, a21 * rho) * np.sin(2.0 * theta)

    # Avoid meaningless angular values at rho=0; radial factor already zero for m>0.
    return out


def normalize_reference(ref: np.ndarray, w: np.ndarray) -> np.ndarray:
    norm = math.sqrt(float(np.sum(w * ref * ref)))
    return ref / max(norm, 1e-30)


def m1_weights_and_overlaps(
    modes: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    refs = disk_reference_modes_numpy(u, v)
    c = normalize_reference(refs["m1_c"], w)
    s = normalize_reference(refs["m1_s"], w)

    # Normalize modes under the same quadrature for safety.
    modes_n = modes.copy()
    for j in range(modes_n.shape[1]):
        norm = math.sqrt(float(np.sum(w * modes_n[:, j] ** 2)))
        modes_n[:, j] /= max(norm, 1e-30)

    ov_c = np.array([np.sum(w * modes_n[:, j] * c) for j in range(modes_n.shape[1])])
    ov_s = np.array([np.sum(w * modes_n[:, j] * s) for j in range(modes_n.shape[1])])
    weights = ov_c ** 2 + ov_s ** 2
    overlaps = np.column_stack([ov_c, ov_s])
    return weights, overlaps


# -----------------------------------------------------------------------------
# Finite-difference reference on the ellipse
# -----------------------------------------------------------------------------

def finite_difference_ellipse_eigs(
    a: float,
    b: float,
    n_grid: int = 101,
    k: int = 5,
) -> np.ndarray:
    """Simple 5-point finite-difference Dirichlet reference on an ellipse.

    This is not used in training.  It gives an independent numerical reference
    when eta != 1, where no elementary analytic formula is available.
    """
    if n_grid % 2 == 0:
        n_grid += 1

    xs = np.linspace(-a, a, n_grid)
    ys = np.linspace(-b, b, n_grid)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]

    index = {}
    pts = []
    count = 0
    # Exclude points too close to the boundary; Dirichlet boundary is outside.
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            inside = (x / a) ** 2 + (y / b) ** 2 < 1.0
            if inside:
                index[(ix, iy)] = count
                pts.append((ix, iy))
                count += 1

    if count <= k + 2:
        raise RuntimeError("Too few grid points inside ellipse for FD reference")

    A = lil_matrix((count, count), dtype=float)
    diag = 2.0 / hx**2 + 2.0 / hy**2

    for row, (ix, iy) in enumerate(pts):
        A[row, row] = diag
        for dix, diy, coeff in [
            (-1, 0, -1.0 / hx**2),
            (1, 0, -1.0 / hx**2),
            (0, -1, -1.0 / hy**2),
            (0, 1, -1.0 / hy**2),
        ]:
            nb = (ix + dix, iy + diy)
            if nb in index:
                A[row, index[nb]] = coeff
            # If neighbor is outside, boundary value is zero: no offdiag term.

    A = A.tocsr()
    vals = eigsh(A, k=k, which="SM", return_eigenvectors=False, tol=1e-8)
    vals.sort()
    return vals


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

@dataclass
class EtaResult:
    eta: float
    a: float
    b: float
    eigvals: np.ndarray
    fd_eigvals: np.ndarray | None
    m1_weights: np.ndarray
    m1_overlaps: np.ndarray
    best_restart: int
    best_score: float
    grid_data: pd.DataFrame | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_eta(
    eta: float,
    args: argparse.Namespace,
    root: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> EtaResult:
    a = float(args.a)
    b = float(args.a * eta)

    train_quad = make_disk_quadrature(args.n_r, args.n_theta, device, dtype, requires_grad=True)
    val_quad = make_disk_quadrature(args.n_validate_r, args.n_validate_theta, device, dtype, requires_grad=True)

    best_state = None
    best_score = float("inf")
    best_restart = -1
    best_val: RitzResult | None = None

    print(f"\nTraining ellipse eta={eta:.4f}  (a={a:.4f}, b={b:.4f})")
    print("  physical meaning: eta=1 disk; eta != 1 breaks rotational symmetry")

    for r in range(args.restarts):
        seed = args.seed + 1000 * int(round(1000 * eta)) + r
        set_seed(seed)

        model = SubspaceNet(
            n_outputs=args.states,
            hidden_width=args.hidden_width,
            hidden_layers=args.hidden_layers,
            fourier_features=args.fourier_features,
        ).to(device=device, dtype=dtype)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_local = float("inf")
        best_local_state = None

        print(f"  restart {r+1}/{args.restarts}")
        for epoch in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)
            rr = compute_ritz(model, train_quad, a=a, b=b, jitter=args.jitter)

            # Sum of the first K Ritz values is the subspace objective.
            eig_loss = torch.sum(rr.eigvals[: args.states])

            # Light orthonormality diagnostic/regularizer on the Ritz modes.
            ortho = orthonormality_error(rr.modes[:, : args.states], train_quad.w)

            loss = eig_loss + args.weight_ortho * ortho
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if epoch % args.print_every == 0 or epoch == 1 or epoch == args.epochs:
                with torch.no_grad():
                    vals = rr.eigvals[: args.states].detach().cpu().numpy()
                    print(
                        f"    epoch {epoch:6d}/{args.epochs}: "
                        f"sumE={vals.sum():.8f}, "
                        f"E=[" + ", ".join(f"{x:.5f}" for x in vals) + "]"
                    )

            # checkpoint based on validation, not training, to avoid quadrature overfit
            if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
                val_score, _ = evaluate_model_score(model, val_quad, a, b, args)
                if val_score < best_local:
                    best_local = val_score
                    best_local_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Optional LBFGS from the best Adam state
        if best_local_state is not None:
            model.load_state_dict(best_local_state)

        if args.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_steps,
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                rr = compute_ritz(model, train_quad, a=a, b=b, jitter=args.jitter)
                eig_loss = torch.sum(rr.eigvals[: args.states])
                ortho = orthonormality_error(rr.modes[:, : args.states], train_quad.w)
                loss = eig_loss + args.weight_ortho * ortho
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except RuntimeError as exc:
                print(f"    [warn] LBFGS failed for restart {r+1}: {exc}")

        val_score, val_rr = evaluate_model_score(model, val_quad, a, b, args)
        vals = val_rr.eigvals[: args.states].detach().cpu().numpy()

        print(
            f"    restart result: score={val_score:.8f}, "
            f"E=[" + ", ".join(f"{x:.8f}" for x in vals) + "]"
        )

        if val_score < best_score:
            best_score = val_score
            best_restart = r + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val = val_rr

    if best_state is None:
        raise RuntimeError(f"No valid model found for eta={eta}")

    final_model = SubspaceNet(
        n_outputs=args.states,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_features=args.fourier_features,
    ).to(device=device, dtype=dtype)
    final_model.load_state_dict(best_state)

    # Final validation with independent quadrature.
    final_quad = make_disk_quadrature(args.n_final_r, args.n_final_theta, device, dtype, requires_grad=True)
    final_rr = compute_ritz(final_model, final_quad, a=a, b=b, jitter=args.jitter)

    eigvals = final_rr.eigvals[: args.states].detach().cpu().numpy()

    u = final_quad.uv[:, 0].detach().cpu().numpy()
    v = final_quad.uv[:, 1].detach().cpu().numpy()
    w = final_quad.w.detach().cpu().numpy()
    modes = final_rr.modes[:, : args.states].detach().cpu().numpy()
    m1_weights, m1_overlaps = m1_weights_and_overlaps(modes, u, v, w)

    fd = None
    if args.fd_reference:
        try:
            fd = finite_difference_ellipse_eigs(a=a, b=b, n_grid=args.fd_grid, k=max(args.states + 2, 5))
        except Exception as exc:
            print(f"  [warn] finite-difference reference failed for eta={eta}: {exc}")
            fd = None

    print(f"  selected restart: {best_restart}")
    print("  final PINN/Ritz energies:")
    for i, ev in enumerate(eigvals, start=1):
        print(f"    {i:2d}: epsilon={ev:.8f}, w(m=1)={m1_weights[i-1]:.4f}")

    grid_df = evaluate_modes_on_grid(final_model, a, b, eta, args, device, dtype)

    return EtaResult(
        eta=eta,
        a=a,
        b=b,
        eigvals=eigvals,
        fd_eigvals=fd,
        m1_weights=m1_weights,
        m1_overlaps=m1_overlaps,
        best_restart=best_restart,
        best_score=best_score,
        grid_data=grid_df,
    )


def evaluate_model_score(
    model: SubspaceNet,
    quad: DiskQuadrature,
    a: float,
    b: float,
    args: argparse.Namespace,
) -> Tuple[float, RitzResult]:
    rr = compute_ritz(model, quad, a=a, b=b, jitter=args.jitter)
    vals = rr.eigvals[: args.states]
    score = float(torch.sum(vals).detach().cpu())
    # Small penalty for non-orthonormal Ritz modes on validation grid.
    ortho = float(orthonormality_error(rr.modes[:, : args.states], quad.w).detach().cpu())
    score += args.selection_ortho_weight * ortho
    return score, rr


# -----------------------------------------------------------------------------
# Evaluation on rectangular plotting grid
# -----------------------------------------------------------------------------

def evaluate_modes_on_grid(
    model: SubspaceNet,
    a: float,
    b: float,
    eta: float,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    n = args.n_plot_grid
    xs = np.linspace(-a, a, n)
    ys = np.linspace(-b, b, n)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    U = X / a
    V = Y / b
    mask = U**2 + V**2 <= 1.0

    uv = np.column_stack([U.ravel(), V.ravel()])
    uv_t = torch.tensor(uv, device=device, dtype=dtype)
    uv_t.requires_grad_(True)

    # Need Ritz coefficients from a quadrature grid.
    quad = make_disk_quadrature(args.n_final_r, args.n_final_theta, device, dtype, requires_grad=True)
    rr = compute_ritz(model, quad, a=a, b=b, jitter=args.jitter)

    with torch.no_grad():
        basis_grid = model.forward_basis(uv_t)
        modes_grid = basis_grid @ rr.coeffs[:, : args.states]
        modes_np = modes_grid.detach().cpu().numpy()

    rows = {
        "eta": np.full(X.size, eta),
        "x": X.ravel(),
        "y": Y.ravel(),
        "u": U.ravel(),
        "v": V.ravel(),
        "inside": mask.ravel().astype(int),
    }
    for j in range(args.states):
        values = modes_np[:, j]
        values[~mask.ravel()] = np.nan
        rows[f"mode_{j+1}"] = values

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Save outputs and figures
# -----------------------------------------------------------------------------

def save_results(results: List[EtaResult], root: Path, args: argparse.Namespace) -> None:
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    # Summary rows
    rows = []
    for res in results:
        fd_vals = res.fd_eigvals if res.fd_eigvals is not None else []
        e1 = res.eigvals[0] if len(res.eigvals) > 0 else np.nan
        e2 = res.eigvals[1] if len(res.eigvals) > 1 else np.nan
        e3 = res.eigvals[2] if len(res.eigvals) > 2 else np.nan
        fd1 = fd_vals[0] if len(fd_vals) > 0 else np.nan
        fd2 = fd_vals[1] if len(fd_vals) > 1 else np.nan
        fd3 = fd_vals[2] if len(fd_vals) > 2 else np.nan
        rows.append({
            "eta": res.eta,
            "a": res.a,
            "b": res.b,
            "E1_PINN": e1,
            "E2_PINN": e2,
            "E3_PINN": e3,
            "Delta_m1_PINN": e3 - e2,
            "E1_FD": fd1,
            "E2_FD": fd2,
            "E3_FD": fd3,
            "Delta_m1_FD": fd3 - fd2 if np.isfinite(fd2) and np.isfinite(fd3) else np.nan,
            "m1_weight_mode1": res.m1_weights[0] if len(res.m1_weights) > 0 else np.nan,
            "m1_weight_mode2": res.m1_weights[1] if len(res.m1_weights) > 1 else np.nan,
            "m1_weight_mode3": res.m1_weights[2] if len(res.m1_weights) > 2 else np.nan,
            "best_restart": res.best_restart,
            "best_score": res.best_score,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(data_dir / "pinn_ellipse_2d_splitting.csv", index=False)

    # Ranked spectrum long table
    long_rows = []
    for res in results:
        for i, ev in enumerate(res.eigvals, start=1):
            fd = res.fd_eigvals[i - 1] if res.fd_eigvals is not None and len(res.fd_eigvals) >= i else np.nan
            long_rows.append({
                "eta": res.eta,
                "state": i,
                "epsilon_PINN": ev,
                "epsilon_FD": fd,
                "rel_error_vs_FD_percent": 100.0 * abs(ev - fd) / abs(fd) if np.isfinite(fd) and fd != 0 else np.nan,
                "m1_weight": res.m1_weights[i - 1] if len(res.m1_weights) >= i else np.nan,
                "overlap_m1_cos": res.m1_overlaps[i - 1, 0] if res.m1_overlaps.shape[0] >= i else np.nan,
                "overlap_m1_sin": res.m1_overlaps[i - 1, 1] if res.m1_overlaps.shape[0] >= i else np.nan,
            })
    pd.DataFrame(long_rows).to_csv(data_dir / "pinn_ellipse_2d_ranked_spectrum.csv", index=False)

    # m=1 overlap table
    overlap_rows = []
    for res in results:
        for i in range(min(args.states, res.m1_overlaps.shape[0])):
            overlap_rows.append({
                "eta": res.eta,
                "state": i + 1,
                "overlap_m1_cos": res.m1_overlaps[i, 0],
                "overlap_m1_sin": res.m1_overlaps[i, 1],
                "m1_weight": res.m1_weights[i],
            })
    pd.DataFrame(overlap_rows).to_csv(data_dir / "pinn_ellipse_2d_m1_overlaps.csv", index=False)

    # Modes grid
    grid_frames = [r.grid_data for r in results if r.grid_data is not None]
    if grid_frames:
        pd.concat(grid_frames, ignore_index=True).to_csv(data_dir / "pinn_ellipse_2d_modes_grid.csv", index=False)

    make_splitting_figure(summary, fig_dir)
    make_delta_figure(summary, fig_dir)
    make_m1_weight_figure(pd.DataFrame(overlap_rows), fig_dir)
    if grid_frames:
        make_modes_examples(pd.concat(grid_frames, ignore_index=True), fig_dir, args)

    print("\nSaved outputs:")
    for p in [
        data_dir / "pinn_ellipse_2d_splitting.csv",
        data_dir / "pinn_ellipse_2d_ranked_spectrum.csv",
        data_dir / "pinn_ellipse_2d_m1_overlaps.csv",
        data_dir / "pinn_ellipse_2d_modes_grid.csv",
        fig_dir / "pinn_ellipse_2d_splitting.png",
        fig_dir / "pinn_ellipse_2d_splitting_delta.png",
        fig_dir / "pinn_ellipse_2d_modes_examples.png",
        fig_dir / "pinn_ellipse_2d_m1_weights.png",
    ]:
        if p.exists():
            print(f"  {p.relative_to(root)}")

    print("\nSummary")
    print("  eta      E2_PINN       E3_PINN     Delta_PINN      E2_FD        E3_FD      Delta_FD")
    for _, row in summary.iterrows():
        print(
            f"  {row['eta']:5.2f}  "
            f"{row['E2_PINN']:12.6f} {row['E3_PINN']:12.6f} {row['Delta_m1_PINN']:12.6f}  "
            f"{row['E2_FD']:10.6f} {row['E3_FD']:10.6f} {row['Delta_m1_FD']:10.6f}"
        )

    print("\nInterpretation note:")
    print("  eta=1 is the circular disk, where the m=1 pair is degenerate.")
    print("  eta != 1 breaks rotational symmetry and splits the m=1 doublet.")
    print("  Finite-difference values are independent validation only; they are not training targets.")


def make_splitting_figure(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    x = df["eta"].to_numpy()

    ax.plot(x, df["E2_PINN"], "o-", label=r"PINN/Ritz, lower $m=1$")
    ax.plot(x, df["E3_PINN"], "s-", label=r"PINN/Ritz, upper $m=1$")

    if df["E2_FD"].notna().any():
        ax.plot(x, df["E2_FD"], "--", color="0.35", lw=1.0, label="finite difference")
        ax.plot(x, df["E3_FD"], "--", color="0.35", lw=1.0)

    ax.axvline(1.0, color="0.2", ls=":", lw=0.9)
    ax.set_xlabel(r"aspect ratio $\eta=b/a$")
    ax.set_ylabel(r"eigenvalue $\epsilon$")
    ax.set_title(r"elliptic deformation splits the $m=1$ doublet")
    ax.legend(loc="best", fontsize=7)
    ax.minorticks_on()
    save_figure(fig, fig_dir / "pinn_ellipse_2d_splitting")


def make_delta_figure(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = df["eta"].to_numpy()

    ax.plot(x, df["Delta_m1_PINN"], "o-", label="PINN/Ritz")
    if df["Delta_m1_FD"].notna().any():
        ax.plot(x, df["Delta_m1_FD"], "s--", color="0.35", lw=1.0, label="finite difference")

    ax.axhline(0.0, color="0.2", lw=0.8)
    ax.axvline(1.0, color="0.2", ls=":", lw=0.9)
    ax.set_xlabel(r"aspect ratio $\eta=b/a$")
    ax.set_ylabel(r"$\Delta\epsilon_{m=1}=\epsilon_3-\epsilon_2$")
    ax.set_title("degeneracy lifting")
    ax.legend(loc="best", fontsize=7)
    ax.minorticks_on()
    save_figure(fig, fig_dir / "pinn_ellipse_2d_splitting_delta")


def make_m1_weight_figure(df: pd.DataFrame, fig_dir: Path) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(3.45, 2.45))
    for state, marker in [(1, "o"), (2, "s"), (3, "^")]:
        sub = df[df["state"] == state]
        if not sub.empty:
            ax.plot(sub["eta"], sub["m1_weight"], marker + "-", label=fr"state {state}")
    ax.axvline(1.0, color="0.2", ls=":", lw=0.9)
    ax.set_xlabel(r"aspect ratio $\eta=b/a$")
    ax.set_ylabel(r"weight in circular $m=1$ subspace")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="best", fontsize=7)
    ax.minorticks_on()
    save_figure(fig, fig_dir / "pinn_ellipse_2d_m1_weights")


def make_modes_examples(grid: pd.DataFrame, fig_dir: Path, args: argparse.Namespace) -> None:
    # Pick eta closest to 0.8, 1.0, 1.2 if available.
    etas_available = sorted(grid["eta"].unique())
    target_etas = [min(etas_available, key=lambda e: abs(e - t)) for t in [0.8, 1.0, 1.2]]
    # Keep unique preserving order.
    target_etas = list(dict.fromkeys(target_etas))

    nrows = len(target_etas)
    ncols = min(args.states, 3)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.15 * ncols, 2.05 * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for i, eta in enumerate(target_etas):
        sub = grid[np.isclose(grid["eta"], eta)]
        xs = np.sort(sub["x"].unique())
        ys = np.sort(sub["y"].unique())
        nx, ny = len(xs), len(ys)

        for j in range(ncols):
            ax = axes[i, j]
            Z = sub[f"mode_{j+1}"].to_numpy().reshape(ny, nx)
            vmax = np.nanmax(np.abs(Z))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            im = ax.imshow(
                Z,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="bilinear",
                aspect="equal",
            )
            # ellipse boundary
            a = float(args.a)
            b = float(args.a * eta)
            th = np.linspace(0, 2*np.pi, 300)
            ax.plot(a*np.cos(th), b*np.sin(th), color="0.1", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(fr"mode {j+1}")
            if j == 0:
                ax.set_ylabel(fr"$\eta={eta:.2f}$")
            panel_label(ax, f"({chr(ord('a') + i*ncols + j)})", x=0.03, y=0.94)

    save_figure(fig, fig_dir / "pinn_ellipse_2d_modes_examples")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PINN/Rayleigh-Ritz splitting in a 2D ellipse")
    p.add_argument("--root", type=str, default=None, help="Project root. Default inferred from script location.")
    p.add_argument("--etas", type=str, default="0.80,0.90,1.00,1.10,1.20", help="Comma-separated aspect ratios eta=b/a")
    p.add_argument("--a", type=float, default=1.0, help="Semi-axis a. Semi-axis b is a*eta.")
    p.add_argument("--states", type=int, default=3, help="Number of low-energy Ritz states to train")

    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lbfgs-steps", type=int, default=120)
    p.add_argument("--lbfgs-lr", type=float, default=0.7)

    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--fourier-features", type=int, default=6)

    p.add_argument("--n-r", type=int, default=20)
    p.add_argument("--n-theta", type=int, default=72)
    p.add_argument("--n-validate-r", type=int, default=22)
    p.add_argument("--n-validate-theta", type=int, default=84)
    p.add_argument("--n-final-r", type=int, default=28)
    p.add_argument("--n-final-theta", type=int, default=112)
    p.add_argument("--n-plot-grid", type=int, default=151)

    p.add_argument("--weight-ortho", type=float, default=1e-4)
    p.add_argument("--selection-ortho-weight", type=float, default=1e-2)
    p.add_argument("--jitter", type=float, default=1e-9)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--print-every", type=int, default=200)

    p.add_argument("--fd-reference", action="store_true", default=True, help="Compute independent finite-difference reference")
    p.add_argument("--no-fd-reference", dest="fd_reference", action="store_false", help="Disable finite-difference reference")
    p.add_argument("--fd-grid", type=int, default=101)

    p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    p.add_argument("--device", type=str, default="cpu")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    setup_publication_style()

    root = Path(args.root).resolve() if args.root else project_root_from_script()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    etas = parse_etas(args.etas)

    print("2D elliptical-well PINN/Rayleigh--Ritz subspace")
    print(f"  root: {root}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")
    print(f"  states: {args.states}")
    print(f"  etas: {', '.join(f'{e:.2f}' for e in etas)}")
    print("  exact disk/Bessel modes are used only for validation/interpretation")
    print("  finite-difference reference is independent and not used during training")

    results: List[EtaResult] = []
    for eta in etas:
        res = train_one_eta(eta, args, root, device, dtype)
        results.append(res)

    save_results(results, root, args)


if __name__ == "__main__":
    main()
