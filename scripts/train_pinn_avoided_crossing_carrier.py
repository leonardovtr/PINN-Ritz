#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pinn_avoided_crossing_carrier.py

Carrier-mode Rayleigh--Ritz/PINN test for an avoided crossing.

Purpose
-------
This script uses a local carrier ansatz for branch tracking near a known
near-degenerate band:

    phi_1(u,v) = psi_12(u,v) + alpha B(u,v) N_1(u,v)
    phi_2(u,v) = psi_21(u,v) + alpha B(u,v) N_2(u,v)

where B(u,v)=u(1-u)v(1-v) enforces homogeneous Dirichlet corrections and
psi_12, psi_21 are the carrier modes of the crossing band.

Then a 2x2 Rayleigh--Ritz problem is solved inside span{phi_1, phi_2}.  This is
not a black-box eigensolver; it is a symmetry/spectral-object tracking ansatz for
a known near-degenerate subspace.  It is exactly the kind of construction one
would use to follow physical modal identity across an avoided crossing.

Physics
-------
The unperturbed rectangular branches (1,2) and (2,1) cross at eta=1.
A controlled mixing potential

    V(u,v) = lambda_mix (u - 1/2)(v - 1/2)

opens an avoided crossing and causes exchange of modal character.

Outputs
-------
data/avoided_crossing_carrier_summary.csv
data/avoided_crossing_carrier_character.csv
data/avoided_crossing_carrier_overlap_tracking.csv

figures/avoided_crossing_carrier_summary.pdf/png
figures/avoided_crossing_carrier_energies.pdf/png
figures/avoided_crossing_carrier_character.pdf/png

Run
---
cd PINN

Quick:
python scripts/train_pinn_avoided_crossing_carrier.py --root . --quick

Stronger:
python scripts/train_pinn_avoided_crossing_carrier.py --root . --epochs 2000 --nq 64 --nmax_ref 10 --restarts 4

A control run with no neural correction is also possible:
python scripts/train_pinn_avoided_crossing_carrier.py --root . --alpha 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


# -----------------------------------------------------------------------------
# Network
# -----------------------------------------------------------------------------

class FourierMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        width: int = 64,
        depth: int = 3,
        n_fourier: int = 8,
        sigma_fourier: float = 2.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.n_fourier = n_fourier
        if n_fourier > 0:
            self.register_buffer("Bmat", sigma_fourier * torch.randn(in_dim, n_fourier, dtype=dtype))
            feat_dim = in_dim + 2 * n_fourier
        else:
            self.register_buffer("Bmat", torch.empty(in_dim, 0, dtype=dtype))
            feat_dim = in_dim

        layers: List[nn.Module] = []
        last = feat_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width, dtype=dtype))
            layers.append(nn.Tanh())
            last = width
        layers.append(nn.Linear(last, out_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

        # Start corrections very small.
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.25)
                    nn.init.zeros_(m.bias)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_fourier <= 0:
            return x
        z = 2.0 * math.pi * x @ self.Bmat
        return torch.cat([x, torch.sin(z), torch.cos(z)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(x))


# -----------------------------------------------------------------------------
# Analytical/reference helpers
# -----------------------------------------------------------------------------

def make_grid(nq: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 1.0, nq)
    v = np.linspace(0.0, 1.0, nq)
    U, V = np.meshgrid(u, v, indexing="ij")
    pts = np.stack([U.ravel(), V.ravel()], axis=1)
    return pts, U, V


def sine_basis_np(pts: np.ndarray, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = pts[:, 0]
    v = pts[:, 1]
    phi = 2.0 * np.sin(nx * math.pi * u) * np.sin(ny * math.pi * v)
    du = 2.0 * nx * math.pi * np.cos(nx * math.pi * u) * np.sin(ny * math.pi * v)
    dv = 2.0 * ny * math.pi * np.sin(nx * math.pi * u) * np.cos(ny * math.pi * v)
    return phi, du, dv


def potential_np(pts: np.ndarray, lambda_mix: float) -> np.ndarray:
    u = pts[:, 0]
    v = pts[:, 1]
    return lambda_mix * (u - 0.5) * (v - 0.5)


def uncoupled_energies(eta: float) -> Tuple[float, float]:
    e12 = math.pi**2 * (1.0 + 4.0 / (eta * eta))
    e21 = math.pi**2 * (4.0 + 1.0 / (eta * eta))
    return e12, e21


def generalized_eigh_np(K: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    M = M + 1e-12 * np.eye(M.shape[0])
    L = np.linalg.cholesky(M)
    Y = np.linalg.solve(L, K)
    S = np.linalg.solve(L, Y.T).T
    S = 0.5 * (S + S.T)
    evals, Z = np.linalg.eigh(S)
    C = np.linalg.solve(L.T, Z)
    return evals, C


def reference_spectrum(
    eta: float,
    pts: np.ndarray,
    w: np.ndarray,
    nmax: int,
    lambda_mix: float,
) -> Dict[str, object]:
    phis, dus, dvs = [], [], []
    labels = []
    for nx in range(1, nmax + 1):
        for ny in range(1, nmax + 1):
            phi, du, dv = sine_basis_np(pts, nx, ny)
            phis.append(phi)
            dus.append(du)
            dvs.append(dv)
            labels.append((nx, ny))

    Phi = np.stack(phis, axis=1)
    Du = np.stack(dus, axis=1)
    Dv = np.stack(dvs, axis=1)
    V = potential_np(pts, lambda_mix)

    nb = Phi.shape[1]
    K = np.zeros((nb, nb))
    M = np.zeros((nb, nb))

    for i in range(nb):
        for j in range(i, nb):
            kij = np.sum(
                w * (
                    Du[:, i] * Du[:, j]
                    + (Dv[:, i] * Dv[:, j]) / (eta * eta)
                    + V * Phi[:, i] * Phi[:, j]
                )
            )
            mij = np.sum(w * Phi[:, i] * Phi[:, j])
            K[i, j] = K[j, i] = kij
            M[i, j] = M[j, i] = mij

    evals, C = generalized_eigh_np(K, M)
    modes = Phi @ C

    for k in range(modes.shape[1]):
        modes[:, k] /= math.sqrt(float(np.sum(w * modes[:, k] ** 2))) + 1e-14

    psi12, _, _ = sine_basis_np(pts, 1, 2)
    psi21, _, _ = sine_basis_np(pts, 2, 1)
    psi12 /= math.sqrt(float(np.sum(w * psi12**2))) + 1e-14
    psi21 /= math.sqrt(float(np.sum(w * psi21**2))) + 1e-14

    char = []
    for k in range(5):
        w12 = float(np.sum(w * modes[:, k] * psi12) ** 2)
        w21 = float(np.sum(w * modes[:, k] * psi21) ** 2)
        char.append((w12, w21))

    return {"evals": evals, "modes": modes, "char": char, "labels": labels}


# -----------------------------------------------------------------------------
# Torch carrier ansatz
# -----------------------------------------------------------------------------

def boundary_square_torch(pts: torch.Tensor) -> torch.Tensor:
    u = pts[:, 0]
    v = pts[:, 1]
    return u * (1.0 - u) * v * (1.0 - v)


def potential_torch(pts: torch.Tensor, lambda_mix: float) -> torch.Tensor:
    u = pts[:, 0]
    v = pts[:, 1]
    return lambda_mix * (u - 0.5) * (v - 0.5)


def psi_carrier_torch(pts: torch.Tensor, nx: int, ny: int) -> torch.Tensor:
    u = pts[:, 0]
    v = pts[:, 1]
    return 2.0 * torch.sin(nx * math.pi * u) * torch.sin(ny * math.pi * v)


def first_derivatives(y: torch.Tensor, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    grad = torch.autograd.grad(
        y, pts, torch.ones_like(y), create_graph=True, retain_graph=True
    )[0]
    return grad[:, 0], grad[:, 1]


def build_carrier_basis(
    model: nn.Module,
    pts: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Return N x 2 carrier-corrected basis."""
    if alpha == 0.0:
        corr = torch.zeros((pts.shape[0], 2), dtype=pts.dtype, device=pts.device)
    else:
        corr = model(pts)

    B = boundary_square_torch(pts)
    psi12 = psi_carrier_torch(pts, 1, 2)
    psi21 = psi_carrier_torch(pts, 2, 1)

    phi1 = psi12 + alpha * B * corr[:, 0]
    phi2 = psi21 + alpha * B * corr[:, 1]
    Phi = torch.stack([phi1, phi2], dim=1)
    return Phi


def ritz_from_basis(
    Phi: torch.Tensor,
    pts: torch.Tensor,
    eta: float,
    lambda_mix: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build 2x2 weak Hamiltonian and mass matrices and solve generalized Ritz."""
    V = potential_torch(pts, lambda_mix)
    grads = []
    for k in range(Phi.shape[1]):
        du, dv = first_derivatives(Phi[:, k], pts)
        grads.append((du, dv))

    K = torch.zeros((2, 2), dtype=pts.dtype, device=pts.device)
    M = torch.zeros((2, 2), dtype=pts.dtype, device=pts.device)
    for i in range(2):
        for j in range(i, 2):
            kij = torch.mean(
                grads[i][0] * grads[j][0]
                + (grads[i][1] * grads[j][1]) / (eta * eta)
                + V * Phi[:, i] * Phi[:, j]
            )
            mij = torch.mean(Phi[:, i] * Phi[:, j])
            K[i, j] = K[j, i] = kij
            M[i, j] = M[j, i] = mij

    Mj = M + 1e-12 * torch.eye(2, dtype=pts.dtype, device=pts.device)
    L = torch.linalg.cholesky(Mj)
    Y = torch.linalg.solve_triangular(L, K, upper=False)
    S = torch.linalg.solve_triangular(L, Y.T, upper=False).T
    S = 0.5 * (S + S.T)
    evals, Z = torch.linalg.eigh(S)
    C = torch.linalg.solve_triangular(L.T, Z, upper=True)
    return evals, C, K, M


def band_tracking_penalties(
    Phi: torch.Tensor,
    C: torch.Tensor,
    pts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Penalize leakage out of the intended (1,2)/(2,1) tracking band.

    For alpha > 0 the neural corrections are variationally free enough to
    reduce the Ritz energy by contaminating the target band with lower modes.
    These penalties keep the learned Ritz modes close to the carrier subspace
    and away from the ground-state component.
    """
    modes = Phi @ C
    modes = modes / torch.sqrt(torch.mean(modes**2, dim=0, keepdim=True) + 1e-14)

    psi11 = psi_carrier_torch(pts, 1, 1)
    psi12 = psi_carrier_torch(pts, 1, 2)
    psi21 = psi_carrier_torch(pts, 2, 1)
    psi11 = psi11 / torch.sqrt(torch.mean(psi11**2) + 1e-14)
    psi12 = psi12 / torch.sqrt(torch.mean(psi12**2) + 1e-14)
    psi21 = psi21 / torch.sqrt(torch.mean(psi21**2) + 1e-14)

    low_leak = torch.tensor(0.0, dtype=pts.dtype, device=pts.device)
    anchor_loss = torch.tensor(0.0, dtype=pts.dtype, device=pts.device)
    for k in range(modes.shape[1]):
        m = modes[:, k]
        w11 = torch.mean(m * psi11) ** 2
        w12 = torch.mean(m * psi12) ** 2
        w21 = torch.mean(m * psi21) ** 2
        carrier_weight = w12 + w21
        low_leak = low_leak + w11
        anchor_loss = anchor_loss + (1.0 - carrier_weight) ** 2
    return low_leak / modes.shape[1], anchor_loss / modes.shape[1]


def correction_h1_penalty(
    model: nn.Module,
    pts: torch.Tensor,
    eta: float,
    alpha: float,
) -> torch.Tensor:
    """Penalize the energy-size of the neural carrier correction.

    L2 overlap anchoring can remain small even when the correction changes the
    derivatives enough to lower the Rayleigh quotient. This H1-like term keeps
    the correction small in the same metric used by the kinetic energy.
    """
    if alpha == 0.0:
        return torch.tensor(0.0, dtype=pts.dtype, device=pts.device)

    corr = model(pts)
    B = boundary_square_torch(pts)
    penalty = torch.tensor(0.0, dtype=pts.dtype, device=pts.device)
    for k in range(corr.shape[1]):
        delta = alpha * B * corr[:, k]
        du, dv = first_derivatives(delta, pts)
        penalty = penalty + torch.mean(delta**2 + du**2 + (dv**2) / (eta * eta))
    return penalty / corr.shape[1]


def train_eta(
    eta: float,
    pts_np: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    pts_base = torch.tensor(pts_np, dtype=dtype, device=device)

    best_score = None
    best_state = None
    best_info = None

    for restart in range(args.restarts):
        torch.manual_seed(seed + 7919 * restart)
        np.random.seed(seed + 7919 * restart)

        model = FourierMLP(
            in_dim=2,
            out_dim=2,
            width=args.width,
            depth=args.depth,
            n_fourier=args.n_fourier,
            sigma_fourier=args.fourier_sigma,
            dtype=dtype,
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        for ep in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)

            pts = pts_base.detach().clone().requires_grad_(True)
            Phi = build_carrier_basis(model, pts, args.alpha)
            evals, C, K, M = ritz_from_basis(Phi, pts, eta, args.lambda_mix)

            # The variational 2x2 Ritz result is already the target band.
            # For alpha>0, train corrections only to reduce the total target-band energy.
            # Add small regularization to avoid unnecessary correction growth.
            energy_loss = torch.sum(evals)
            corr_loss = torch.tensor(0.0, dtype=dtype, device=device)
            if args.alpha != 0.0:
                raw_corr = model(pts)
                corr_loss = torch.mean(raw_corr**2)

            # Keep the carrier subspace well conditioned.
            I = torch.eye(2, dtype=dtype, device=device)
            M_scaled = M / torch.sqrt(torch.diag(M)[:, None] * torch.diag(M)[None, :] + 1e-14)
            ortho_loss = torch.mean((M_scaled - I) ** 2)

            low_loss = torch.tensor(0.0, dtype=dtype, device=device)
            anchor_loss = torch.tensor(0.0, dtype=dtype, device=device)
            if args.alpha != 0.0 and (args.lambda_low > 0.0 or args.lambda_anchor > 0.0):
                low_loss, anchor_loss = band_tracking_penalties(Phi, C, pts)
            h1_corr_loss = torch.tensor(0.0, dtype=dtype, device=device)
            if args.alpha != 0.0 and args.lambda_h1corr > 0.0:
                h1_corr_loss = correction_h1_penalty(model, pts, eta, args.alpha)

            loss = (
                energy_loss
                + args.lambda_corr * corr_loss
                + args.lambda_ortho * ortho_loss
                + args.lambda_low * low_loss
                + args.lambda_anchor * anchor_loss
                + args.lambda_h1corr * h1_corr_loss
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            with torch.no_grad():
                score = float((
                    energy_loss
                    + args.lambda_corr * corr_loss
                    + args.lambda_ortho * ortho_loss
                    + args.lambda_low * low_loss
                    + args.lambda_anchor * anchor_loss
                    + args.lambda_h1corr * h1_corr_loss
                ).detach().cpu())

            if best_score is None or score < best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_info = {
                    "restart": restart,
                    "score": score,
                    "energy_loss": float(energy_loss.detach().cpu()),
                    "corr_loss": float(corr_loss.detach().cpu()),
                    "ortho_loss": float(ortho_loss.detach().cpu()),
                    "low_loss": float(low_loss.detach().cpu()),
                    "anchor_loss": float(anchor_loss.detach().cpu()),
                    "h1_corr_loss": float(h1_corr_loss.detach().cpu()),
                }

            if args.verbose and (ep == 1 or ep % args.print_every == 0 or ep == args.epochs):
                print(
                    f"    restart {restart+1}/{args.restarts}, epoch {ep:5d}/{args.epochs}: "
                    f"E=({float(evals[0]):.6f},{float(evals[1]):.6f}), "
                    f"loss={float(loss):.6e}, corr={float(corr_loss):.2e}, "
                    f"ortho={float(ortho_loss):.2e}, low={float(low_loss):.2e}, "
                    f"anchor={float(anchor_loss):.2e}, h1corr={float(h1_corr_loss):.2e}"
                )

    # Evaluate best.
    model = FourierMLP(
        in_dim=2,
        out_dim=2,
        width=args.width,
        depth=args.depth,
        n_fourier=args.n_fourier,
        sigma_fourier=args.fourier_sigma,
        dtype=dtype,
    ).to(device)
    model.load_state_dict(best_state)

    pts = pts_base.detach().clone().requires_grad_(True)
    Phi = build_carrier_basis(model, pts, args.alpha)
    evals, C, K, M = ritz_from_basis(Phi, pts, eta, args.lambda_mix)

    modes = Phi @ C
    modes = modes / torch.sqrt(torch.mean(modes**2, dim=0, keepdim=True) + 1e-14)

    evals_np = evals.detach().cpu().numpy()
    modes_np = modes.detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()

    w = np.ones(len(pts_np)) / len(pts_np)
    psi12, _, _ = sine_basis_np(pts_np, 1, 2)
    psi21, _, _ = sine_basis_np(pts_np, 2, 1)
    psi12 /= math.sqrt(float(np.sum(w * psi12**2))) + 1e-14
    psi21 /= math.sqrt(float(np.sum(w * psi21**2))) + 1e-14

    char = []
    for k in range(2):
        m = modes_np[:, k]
        m /= math.sqrt(float(np.sum(w * m**2))) + 1e-14
        w12 = float(np.sum(w * m * psi12) ** 2)
        w21 = float(np.sum(w * m * psi21) ** 2)
        char.append((w12, w21))

    return {
        "evals": evals_np,
        "modes": modes_np,
        "char": char,
        "M_cond": float(np.linalg.cond(M_np)),
        "M_trace": float(np.trace(M_np)),
        **best_info,
    }


def overlap_matrix(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
    n = min(modes_a.shape[1], modes_b.shape[1])
    w = np.ones(modes_a.shape[0]) / modes_a.shape[0]
    A = modes_a[:, :n].copy()
    B = modes_b[:, :n].copy()
    for i in range(n):
        A[:, i] /= math.sqrt(float(np.sum(w * A[:, i] ** 2))) + 1e-14
        B[:, i] /= math.sqrt(float(np.sum(w * B[:, i] ** 2))) + 1e-14
    O = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            O[i, j] = float(np.sum(w * A[:, i] * B[:, j]) ** 2)
    return O


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def save_fig(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)


def plot_results(summary: pd.DataFrame, char_df: pd.DataFrame, fig_dir: Path) -> None:
    setup_style()
    eta = summary["eta"].to_numpy()
    lower = char_df[char_df["branch"] == "lower"].copy()

    fig = plt.figure(figsize=(7.2, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85])

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(eta, summary["E12_uncoupled"], "--", color="0.6", lw=0.9, label=r"uncoupled $(1,2)$")
    ax.plot(eta, summary["E21_uncoupled"], ":", color="0.6", lw=1.0, label=r"uncoupled $(2,1)$")
    ax.plot(eta, summary["E2_ref"], "-", color="0.1", lw=1.2, label="reference")
    ax.plot(eta, summary["E3_ref"], "-", color="0.1", lw=1.2)
    ax.plot(eta, summary["E2_pinn"], "o", mfc="white", mec="0.1", mew=0.7, ms=2.8, label="carrier Ritz")
    ax.plot(eta, summary["E3_pinn"], "s", mfc="white", mec="0.1", mew=0.7, ms=2.8)
    ax.set_xlabel(r"$\eta=L_y/L_x$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("(a) avoided crossing")
    ax.legend(fontsize=6.0, loc="best")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(lower["eta"], lower["w12_ref"], "-", color="0.15", lw=1.2, label=r"ref $w_{12}$")
    ax.plot(lower["eta"], lower["w21_ref"], "--", color="0.15", lw=1.2, label=r"ref $w_{21}$")
    ax.plot(lower["eta"], lower["w12_pinn"], "o", mfc="white", mec="0.15", mew=0.7, ms=2.7, label=r"carrier $w_{12}$")
    ax.plot(lower["eta"], lower["w21_pinn"], "s", mfc="white", mec="0.15", mew=0.7, ms=2.7, label=r"carrier $w_{21}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("lower-branch character")
    ax.set_title("(b) modal exchange")
    ax.legend(fontsize=6.0, loc="center right")

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(eta, summary["gap_ref"], "-", color="0.1", lw=1.2, label="reference")
    ax.plot(eta, summary["gap_pinn"], "o", mfc="white", mec="0.1", mew=0.7, ms=2.8, label="carrier")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$E_3-E_2$")
    ax.set_title("(c) gap")
    ax.legend(fontsize=6.0, loc="best")

    save_fig(fig, fig_dir / "avoided_crossing_carrier_summary")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.72))
    ax.plot(eta, summary["E12_uncoupled"], "--", color="0.6", lw=0.9, label=r"uncoupled $(1,2)$")
    ax.plot(eta, summary["E21_uncoupled"], ":", color="0.6", lw=1.0, label=r"uncoupled $(2,1)$")
    ax.plot(eta, summary["E2_ref"], "-", color="0.1", lw=1.2, label="reference")
    ax.plot(eta, summary["E3_ref"], "-", color="0.1", lw=1.2)
    ax.plot(eta, summary["E2_pinn"], "o", mfc="white", mec="0.1", mew=0.7, ms=2.8, label="carrier Ritz")
    ax.plot(eta, summary["E3_pinn"], "s", mfc="white", mec="0.1", mew=0.7, ms=2.8)
    ax.set_xlabel(r"$\eta=L_y/L_x$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("avoided crossing")
    ax.legend(fontsize=6.5, loc="best")
    save_fig(fig, fig_dir / "avoided_crossing_carrier_energies")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.5, 2.72))
    ax.plot(lower["eta"], lower["w12_ref"], "-", color="0.15", lw=1.2, label=r"ref $w_{12}$")
    ax.plot(lower["eta"], lower["w21_ref"], "--", color="0.15", lw=1.2, label=r"ref $w_{21}$")
    ax.plot(lower["eta"], lower["w12_pinn"], "o", mfc="white", mec="0.15", mew=0.7, ms=2.7, label=r"carrier $w_{12}$")
    ax.plot(lower["eta"], lower["w21_pinn"], "s", mfc="white", mec="0.15", mew=0.7, ms=2.7, label=r"carrier $w_{21}$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("lower-branch character")
    ax.set_title("modal character exchange")
    ax.legend(fontsize=6.5, loc="center right")
    save_fig(fig, fig_dir / "avoided_crossing_carrier_character")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--eta_min", type=float, default=0.82)
    p.add_argument("--eta_max", type=float, default=1.18)
    p.add_argument("--n_eta", type=int, default=13)
    p.add_argument("--lambda_mix", type=float, default=120.0)

    p.add_argument("--nq", type=int, default=56)
    p.add_argument("--nmax_ref", type=int, default=10)

    p.add_argument("--alpha", type=float, default=0.0,
                   help="Neural correction amplitude. alpha=0 gives pure carrier Ritz.")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n_fourier", type=int, default=8)
    p.add_argument("--fourier_sigma", type=float, default=2.0)

    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lambda_corr", type=float, default=1e-4)
    p.add_argument("--lambda_ortho", type=float, default=1e-2)
    p.add_argument("--lambda_low", type=float, default=0.0,
                   help="Penalty for leakage of Ritz modes into the lower (1,1) mode.")
    p.add_argument("--lambda_anchor", type=float, default=0.0,
                   help="Penalty for leaving the (1,2)/(2,1) carrier subspace.")
    p.add_argument("--lambda_h1corr", type=float, default=0.0,
                   help="H1-like penalty on the boundary-preserving neural correction.")
    p.add_argument("--grad_clip", type=float, default=10.0)

    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print_every", type=int, default=400)
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.quick:
        args.n_eta = 9
        args.nq = min(args.nq, 42)
        args.nmax_ref = min(args.nmax_ref, 8)
        args.width = min(args.width, 48)
        args.epochs = min(args.epochs, 500)
        args.restarts = min(args.restarts, 2)

    root = Path(args.root).resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = torch.float64

    pts_np, U, V = make_grid(args.nq)
    w_np = np.ones(len(pts_np)) / len(pts_np)
    etas = np.linspace(args.eta_min, args.eta_max, args.n_eta)

    print("2D avoided-crossing carrier-mode Rayleigh--Ritz/PINN")
    print(f"  root: {root}")
    print(f"  eta: {etas[0]:.3f} ... {etas[-1]:.3f} ({len(etas)} points)")
    print(f"  lambda_mix: {args.lambda_mix}")
    print(f"  grid: {args.nq} x {args.nq}")
    print(f"  alpha: {args.alpha}")
    print(f"  epochs per eta: {args.epochs}")
    print(f"  restarts: {args.restarts}")

    rows = []
    char_rows = []
    overlap_rows = []

    ref_prev = None
    pinn_prev = None
    eta_prev = None

    for idx, eta in enumerate(etas):
        print(f"\neta = {eta:.4f} ({idx+1}/{len(etas)})")

        ref = reference_spectrum(
            eta=eta,
            pts=pts_np,
            w=w_np,
            nmax=args.nmax_ref,
            lambda_mix=args.lambda_mix,
        )

        t0 = time.time()
        pinn = train_eta(
            eta=eta,
            pts_np=pts_np,
            args=args,
            seed=args.seed + 97 * idx,
            device=device,
            dtype=dtype,
        )
        runtime = time.time() - t0

        E2_ref, E3_ref = float(ref["evals"][1]), float(ref["evals"][2])
        E2_pinn, E3_pinn = float(pinn["evals"][0]), float(pinn["evals"][1])
        e12, e21 = uncoupled_energies(eta)

        row = {
            "eta": eta,
            "E12_uncoupled": e12,
            "E21_uncoupled": e21,
            "E2_ref": E2_ref,
            "E3_ref": E3_ref,
            "gap_ref": E3_ref - E2_ref,
            "E2_pinn": E2_pinn,
            "E3_pinn": E3_pinn,
            "gap_pinn": E3_pinn - E2_pinn,
            "err_E2_percent": 100.0 * abs(E2_pinn - E2_ref) / abs(E2_ref),
            "err_E3_percent": 100.0 * abs(E3_pinn - E3_ref) / abs(E3_ref),
            "M_cond": pinn["M_cond"],
            "M_trace": pinn["M_trace"],
            "score": pinn["score"],
            "energy_loss": pinn["energy_loss"],
            "corr_loss": pinn["corr_loss"],
            "ortho_loss": pinn["ortho_loss"],
            "best_restart": pinn["restart"],
            "runtime_s": runtime,
        }
        rows.append(row)

        for branch, k_ref, k_pinn in [("lower", 1, 0), ("upper", 2, 1)]:
            w12_ref, w21_ref = ref["char"][k_ref]
            w12_pinn, w21_pinn = pinn["char"][k_pinn]
            char_rows.append({
                "eta": eta,
                "branch": branch,
                "w12_ref": w12_ref,
                "w21_ref": w21_ref,
                "w12_pinn": w12_pinn,
                "w21_pinn": w21_pinn,
            })

        print(f"  ref:  E2={E2_ref:.6f}, E3={E3_ref:.6f}, gap={E3_ref-E2_ref:.6f}")
        print(
            f"  carrier: E2={E2_pinn:.6f}, E3={E3_pinn:.6f}, gap={E3_pinn-E2_pinn:.6f}, "
            f"err=({row['err_E2_percent']:.3f}%, {row['err_E3_percent']:.3f}%), "
            f"Mcond={row['M_cond']:.2e}, trM={row['M_trace']:.3f}"
        )
        print(
            f"  lower character ref=(w12={ref['char'][1][0]:.3f}, w21={ref['char'][1][1]:.3f}) "
            f"carrier=(w12={pinn['char'][0][0]:.3f}, w21={pinn['char'][0][1]:.3f})"
        )

        if ref_prev is not None:
            Oref = overlap_matrix(ref_prev[:, 1:3], ref["modes"][:, 1:3])
            Opinn = overlap_matrix(pinn_prev[:, :2], pinn["modes"][:, :2])
            for i in range(2):
                for j in range(2):
                    overlap_rows.append({
                        "eta_left": eta_prev,
                        "eta_right": eta,
                        "i": i + 2,
                        "j": j + 2,
                        "O_ref": Oref[i, j],
                        "O_pinn": Opinn[i, j],
                    })

        ref_prev = ref["modes"][:, :3]
        pinn_prev = pinn["modes"][:, :2]
        eta_prev = eta

    summary = pd.DataFrame(rows)
    char_df = pd.DataFrame(char_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    summary_path = data_dir / "avoided_crossing_carrier_summary.csv"
    char_path = data_dir / "avoided_crossing_carrier_character.csv"
    overlap_path = data_dir / "avoided_crossing_carrier_overlap_tracking.csv"

    summary.to_csv(summary_path, index=False)
    char_df.to_csv(char_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    plot_results(summary, char_df, fig_dir)

    print("\nSaved outputs:")
    print(f"  {summary_path}")
    print(f"  {char_path}")
    print(f"  {overlap_path}")
    print(f"  {fig_dir / 'avoided_crossing_carrier_summary.pdf'}")
    print(f"  {fig_dir / 'avoided_crossing_carrier_energies.pdf'}")
    print(f"  {fig_dir / 'avoided_crossing_carrier_character.pdf'}")

    print("\nCompact summary:")
    print(summary[[
        "eta", "E2_ref", "E3_ref", "gap_ref",
        "E2_pinn", "E3_pinn", "gap_pinn",
        "err_E2_percent", "err_E3_percent",
        "M_cond", "M_trace"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
