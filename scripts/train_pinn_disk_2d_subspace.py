#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_pinn_disk_2d_subspace.py

Physics-informed Rayleigh--Ritz/PINN calculation for the circular infinite
quantum dot (unit disk by default).

Purpose
-------
This script is the natural continuation of the 1D well, square, and rectangle
experiments.  It targets a different kind of degeneracy: rotational degeneracy
in the disk.  The first mode is nondegenerate (m=0), while the next two modes
span the two-dimensional m=1 angular subspace, usually represented by
cos(theta) and sin(theta).

The script does NOT train against the analytical Bessel modes.  The exact
Bessel solutions are used only after training for validation and interpretation.

Physics
-------
Solve the dimensionless Dirichlet eigenvalue problem

    -∇² ψ = ε ψ,     ψ|_{r=R}=0,

using a Rayleigh--Ritz subspace formulation.  A neural network produces k
boundary-satisfying trial functions,

    φ_j(x,y) = (R² - x² - y²) u_j(x,y),

inside the disk.  The network outputs are used to build the mass and stiffness
matrices

    S_ij = ∫_Ω φ_i φ_j dA,
    K_ij = ∫_Ω ∇φ_i · ∇φ_j dA,

and the Ritz eigenvalues are obtained from

    K c = ε S c.

The loss minimizes the sum of the lowest k Ritz eigenvalues plus a mild
orthogonality/conditioning regularizer on S.  This trains the whole low-energy
subspace at once, which is the correct strategy in the presence of degeneracy.

Recommended command
-------------------
From the project root:

    python scripts/train_pinn_disk_2d_subspace.py --states 3

or from the scripts directory:

    python train_pinn_disk_2d_subspace.py --states 3

Outputs
-------
data/
    pinn_disk_2d_subspace_summary.csv
    pinn_disk_2d_subspace_overlap_matrix.csv
    pinn_disk_2d_subspace_modes_grid.csv

figures/
    pinn_disk_2d_subspace_modes.png
    pinn_disk_2d_subspace_energy_comparison.png
    pinn_disk_2d_subspace_degenerate_subspace.png
    pinn_disk_2d_subspace_overlap_matrix.png
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import torch
from torch import nn

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.special import jn_zeros, jv


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "quantum_dots").exists() or (parent / "scripts").exists():
            return parent
    return here.parents[1]


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "DejaVu Serif",
                "Computer Modern Roman",
            ],
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
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.025,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.frameon": False,
            "image.cmap": "RdBu_r",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.03,
        0.95,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.4),
        zorder=20,
    )


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExactDiskMode:
    label: str
    m: int
    radial_index: int
    angular: str
    zero: float
    epsilon: float
    degeneracy: int


@dataclass
class TrainedDiskSubspace:
    model_state: dict
    ritz_values: np.ndarray
    coeffs: np.ndarray
    S: np.ndarray
    K: np.ndarray
    score: float
    restart: int
    train_loss: float
    orth_error: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"float64", "double"}:
        return torch.float64
    if name in {"float32", "single"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def disk_quadrature(
    n_radial: int,
    n_theta: int,
    radius: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Polar quadrature points and weights for the disk.

    Uses t in [0,1], r = R sqrt(t), so r dr = R² dt / 2.
    The area integral becomes

        ∫_disk f dA = (R²/2) ∫_0^1 ∫_0^{2π} f(sqrt(t),theta) dtheta dt.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    t = 0.5 * (nodes + 1.0)
    wt = 0.5 * weights

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    tt, th = np.meshgrid(t, theta, indexing="ij")
    wtt, _ = np.meshgrid(wt, theta, indexing="ij")

    r = radius * np.sqrt(tt)
    x = r * np.cos(th)
    y = r * np.sin(th)

    # Weight for each point.
    # R²/2 * wt * 2π/Ntheta = π R² wt / Ntheta
    w = np.pi * radius**2 * wtt / float(n_theta)

    xy = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=1), dtype=dtype, device=device)
    w_torch = torch.tensor(w.ravel(), dtype=dtype, device=device)
    if requires_grad:
        xy.requires_grad_(True)
    return xy, w_torch, torch.tensor(np.stack([r.ravel(), th.ravel()], axis=1), dtype=dtype, device=device)


def make_plot_grid(n: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(-radius, radius, n)
    y = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    mask = X**2 + Y**2 <= radius**2
    return X, Y, mask


def weighted_inner_np(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * a * b))


def normalize_columns_np(Psi: np.ndarray, w: np.ndarray) -> np.ndarray:
    out = Psi.copy()
    for j in range(out.shape[1]):
        norm = math.sqrt(max(weighted_inner_np(out[:, j], out[:, j], w), 1e-30))
        out[:, j] /= norm
    return out


# ---------------------------------------------------------------------------
# Exact disk modes
# ---------------------------------------------------------------------------

def exact_disk_modes(
    n_modes: int,
    radius: float = 1.0,
    max_m: int = 8,
    max_radial: int = 5,
) -> list[ExactDiskMode]:
    modes: list[ExactDiskMode] = []
    for m in range(max_m + 1):
        zeros = jn_zeros(m, max_radial)
        for ridx, zero in enumerate(zeros, start=1):
            eps = float((zero / radius) ** 2)
            if m == 0:
                modes.append(
                    ExactDiskMode(
                        label=fr"$\psi_{{{m},{ridx}}}$",
                        m=m,
                        radial_index=ridx,
                        angular="radial",
                        zero=float(zero),
                        epsilon=eps,
                        degeneracy=1,
                    )
                )
            else:
                modes.append(
                    ExactDiskMode(
                        label=fr"$\psi_{{{m},{ridx}}}^c$",
                        m=m,
                        radial_index=ridx,
                        angular="cos",
                        zero=float(zero),
                        epsilon=eps,
                        degeneracy=2,
                    )
                )
                modes.append(
                    ExactDiskMode(
                        label=fr"$\psi_{{{m},{ridx}}}^s$",
                        m=m,
                        radial_index=ridx,
                        angular="sin",
                        zero=float(zero),
                        epsilon=eps,
                        degeneracy=2,
                    )
                )
    modes.sort(key=lambda mode: (mode.epsilon, mode.m, mode.radial_index, mode.angular))
    return modes[:n_modes]


def evaluate_exact_modes_np(
    xy: np.ndarray,
    modes: list[ExactDiskMode],
    radius: float,
) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    out = []
    for mode in modes:
        radial = jv(mode.m, mode.zero * r / radius)
        if mode.m == 0:
            angular = np.ones_like(theta)
        elif mode.angular == "sin":
            angular = np.sin(mode.m * theta)
        else:
            angular = np.cos(mode.m * theta)
        vals = radial * angular
        vals = np.where(r <= radius + 1e-12, vals, np.nan)
        out.append(vals)
    return np.stack(out, axis=1)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class FourierFeatures(nn.Module):
    def __init__(self, n_features: int, radius: float) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.radius = float(radius)

    @property
    def out_dim(self) -> int:
        # x, y, r² plus sin/cos for x and y at each frequency.
        return 3 + 4 * self.n_features

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1] / self.radius
        y = xy[:, 1:2] / self.radius
        r2 = x**2 + y**2
        feats = [x, y, r2]
        for k in range(1, self.n_features + 1):
            kk = float(k) * math.pi
            feats.extend(
                [
                    torch.sin(kk * x),
                    torch.cos(kk * x),
                    torch.sin(kk * y),
                    torch.cos(kk * y),
                ]
            )
        return torch.cat(feats, dim=1)


class DiskSubspaceNet(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_width: int,
        hidden_layers: int,
        fourier_features: int,
        radius: float,
    ) -> None:
        super().__init__()
        self.radius = float(radius)
        self.features = FourierFeatures(fourier_features, radius=radius)
        layers: list[nn.Module] = []
        in_dim = self.features.out_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.Tanh())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

        # Small output scale helps initial conditioning.
        with torch.no_grad():
            last = self.net[-1]
            if isinstance(last, nn.Linear):
                last.weight.mul_(0.2)
                last.bias.mul_(0.2)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        r2 = x**2 + y**2
        # Dirichlet condition at r=R is imposed exactly.
        envelope = self.radius**2 - r2
        return envelope * self.net(self.features(xy))


# ---------------------------------------------------------------------------
# Rayleigh--Ritz matrices and losses
# ---------------------------------------------------------------------------

def channel_gradients(
    values: torch.Tensor,
    xy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ∂values_j/∂x and ∂values_j/∂y for each output channel."""
    grads_x = []
    grads_y = []
    for j in range(values.shape[1]):
        grad = torch.autograd.grad(
            values[:, j].sum(),
            xy,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grads_x.append(grad[:, 0])
        grads_y.append(grad[:, 1])
    return torch.stack(grads_x, dim=1), torch.stack(grads_y, dim=1)


def mass_and_stiffness(
    model: nn.Module,
    xy: torch.Tensor,
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phi = model(xy)
    dphix, dphiy = channel_gradients(phi, xy)
    ww = w[:, None]
    S = phi.T @ (ww * phi)
    K = dphix.T @ (ww * dphix) + dphiy.T @ (ww * dphiy)
    # Symmetrize to reduce numerical asymmetry.
    S = 0.5 * (S + S.T)
    K = 0.5 * (K + K.T)
    return S, K, phi


def generalized_ritz(
    K: torch.Tensor,
    S: torch.Tensor,
    jitter: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve K c = lambda S c using symmetric whitening."""
    n = S.shape[0]
    eye = torch.eye(n, dtype=S.dtype, device=S.device)
    S_reg = S + jitter * eye
    eval_s, U = torch.linalg.eigh(S_reg)
    eval_s = torch.clamp(eval_s, min=jitter)
    Sinvhalf = U @ torch.diag(eval_s.rsqrt()) @ U.T
    A = Sinvhalf @ K @ Sinvhalf
    A = 0.5 * (A + A.T)
    vals, V = torch.linalg.eigh(A)
    C = Sinvhalf @ V
    return vals, C


def loss_fn(
    model: nn.Module,
    xy: torch.Tensor,
    w: torch.Tensor,
    states: int,
    weight_orth: float,
    weight_condition: float,
    jitter: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    S, K, _ = mass_and_stiffness(model, xy, w)
    vals, _ = generalized_ritz(K, S, jitter=jitter)
    vals = vals[:states]
    eye = torch.eye(states, dtype=S.dtype, device=S.device)
    # This regularizer is not a target eigenfunction.  It simply discourages
    # rank collapse of the learned trial subspace.
    S_block = S[:states, :states]
    S_scale = torch.trace(S_block) / states
    S_normalized = S_block / (S_scale + jitter)
    orth_error = torch.mean((S_normalized - eye) ** 2)
    s_eigs = torch.linalg.eigvalsh(S_block + jitter * eye)
    condition_penalty = torch.mean(torch.relu(1e-6 - s_eigs) ** 2)
    loss = torch.sum(vals) + weight_orth * orth_error + weight_condition * condition_penalty
    info = {
        "loss": float(loss.detach().cpu()),
        "sum_ritz": float(torch.sum(vals).detach().cpu()),
        "orth_error": float(orth_error.detach().cpu()),
        "e0": float(vals[0].detach().cpu()),
        "e_last": float(vals[-1].detach().cpu()),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_restart(
    args: argparse.Namespace,
    restart: int,
    device: torch.device,
    dtype: torch.dtype,
) -> TrainedDiskSubspace:
    set_seed(args.seed + 1000 * restart)

    xy, w, _ = disk_quadrature(
        args.n_radial,
        args.n_theta,
        args.radius,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    model = DiskSubspaceNet(
        out_dim=args.states,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_features=args.fourier_features,
        radius=args.radius,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_state = None
    best_score = float("inf")
    best_info = None

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, info = loss_fn(
            model,
            xy,
            w,
            states=args.states,
            weight_orth=args.weight_orth,
            weight_condition=args.weight_condition,
            jitter=args.jitter,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        score = info["loss"]
        if score < best_score and math.isfinite(score):
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_info = dict(info)

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"    epoch {epoch:6d}/{args.epochs}: "
                f"loss={info['loss']:.6e}, "
                f"sum_ritz={info['sum_ritz']:.6e}, "
                f"e0={info['e0']:.6e}, "
                f"e_last={info['e_last']:.6e}, "
                f"orth={info['orth_error']:.3e}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    # LBFGS refinement.  Keep the best state if LBFGS destabilizes.
    pre_lbfgs_state = copy.deepcopy(model.state_dict())
    pre_loss, pre_info = loss_fn(
        model,
        xy,
        w,
        states=args.states,
        weight_orth=args.weight_orth,
        weight_condition=args.weight_condition,
        jitter=args.jitter,
    )
    pre_score = float(pre_loss.detach().cpu())

    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.lbfgs_steps,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        lbfgs.zero_grad(set_to_none=True)
        loss, _ = loss_fn(
            model,
            xy,
            w,
            states=args.states,
            weight_orth=args.weight_orth,
            weight_condition=args.weight_condition,
            jitter=args.jitter,
        )
        loss.backward()
        return loss

    if args.lbfgs_steps > 0:
        try:
            lbfgs.step(closure)
        except RuntimeError as exc:
            print(f"    [warn] LBFGS failed: {exc}")
            model.load_state_dict(pre_lbfgs_state)

    post_loss, post_info = loss_fn(
        model,
        xy,
        w,
        states=args.states,
        weight_orth=args.weight_orth,
        weight_condition=args.weight_condition,
        jitter=args.jitter,
    )
    post_score = float(post_loss.detach().cpu())

    if not math.isfinite(post_score) or post_score > pre_score * 1.05:
        model.load_state_dict(pre_lbfgs_state)
        final_loss = pre_loss
        final_info = pre_info
    else:
        final_loss = post_loss
        final_info = post_info

    with torch.no_grad():
        # Need gradients for K, so cannot use no_grad for mass_and_stiffness.
        pass

    xy_eval, w_eval, _ = disk_quadrature(
        args.n_validate_radial,
        args.n_validate_theta,
        args.radius,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    S, K, _ = mass_and_stiffness(model, xy_eval, w_eval)
    vals, C = generalized_ritz(K, S, jitter=args.jitter)

    vals_np = vals[: args.states].detach().cpu().numpy()
    C_np = C[:, : args.states].detach().cpu().numpy()
    S_np = S.detach().cpu().numpy()
    K_np = K.detach().cpu().numpy()

    # Restart score: internal physics only.  Exact modes are not used here.
    orth = float(final_info["orth_error"])
    score = float(np.sum(vals_np) + args.selection_orth_weight * orth)

    print(
        f"    restart result: sum_e={np.sum(vals_np):.8f}, "
        f"eigs={np.array2string(vals_np, precision=5)}, "
        f"orth={orth:.3e}, score={score:.8f}",
        flush=True,
    )

    return TrainedDiskSubspace(
        model_state=copy.deepcopy(model.state_dict()),
        ritz_values=vals_np,
        coeffs=C_np,
        S=S_np,
        K=K_np,
        score=score,
        restart=restart,
        train_loss=float(final_loss.detach().cpu()),
        orth_error=orth,
    )


def train_subspace(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[DiskSubspaceNet, TrainedDiskSubspace]:
    best: TrainedDiskSubspace | None = None
    for restart in range(1, args.restarts + 1):
        print(f"  restart {restart}/{args.restarts}", flush=True)
        result = train_one_restart(args, restart, device, dtype)
        if best is None or result.score < best.score:
            best = result

    assert best is not None

    model = DiskSubspaceNet(
        out_dim=args.states,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_features=args.fourier_features,
        radius=args.radius,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(best.model_state)
    print(
        f"  selected restart {best.restart}: "
        f"eigs={np.array2string(best.ritz_values, precision=8)}, "
        f"score={best.score:.8f}",
        flush=True,
    )
    return model, best


# ---------------------------------------------------------------------------
# Evaluation and outputs
# ---------------------------------------------------------------------------

def evaluate_ritz_modes(
    model: nn.Module,
    coeffs_np: np.ndarray,
    xy_np: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype,
    radius: float,
) -> np.ndarray:
    xy = torch.tensor(xy_np, dtype=dtype, device=device)
    with torch.no_grad():
        phi = model(xy).detach().cpu().numpy()
    psi = phi @ coeffs_np
    return psi


def evaluate_on_validation(
    model: nn.Module,
    result: TrainedDiskSubspace,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    xy, w, _ = disk_quadrature(
        args.n_validate_radial,
        args.n_validate_theta,
        args.radius,
        device=device,
        dtype=dtype,
        requires_grad=False,
    )
    xy_np = xy.detach().cpu().numpy()
    w_np = w.detach().cpu().numpy()

    psi = evaluate_ritz_modes(
        model,
        result.coeffs[:, : args.states],
        xy_np,
        device=device,
        dtype=dtype,
        radius=args.radius,
    )
    psi = normalize_columns_np(psi, w_np)

    exact = exact_disk_modes(args.states, radius=args.radius)
    exact_vals = evaluate_exact_modes_np(xy_np, exact, radius=args.radius)
    exact_vals = normalize_columns_np(exact_vals, w_np)

    O = exact_vals.T @ (w_np[:, None] * psi)

    # Align signs for cleaner plots.
    for j in range(psi.shape[1]):
        imax = int(np.argmax(np.abs(O[:, j])))
        if O[imax, j] < 0:
            psi[:, j] *= -1
            O[:, j] *= -1

    return {
        "xy": xy_np,
        "w": w_np,
        "psi": psi,
        "exact_modes": exact,
        "exact_values": exact_vals,
        "overlap": O,
    }


def save_outputs(
    model: nn.Module,
    result: TrainedDiskSubspace,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    eval_data = evaluate_on_validation(model, result, args, device, dtype)
    exact = eval_data["exact_modes"]
    O = eval_data["overlap"]

    # Summary table.
    rows = []
    for i in range(args.states):
        mode = exact[i]
        eps_exact = mode.epsilon
        eps_pinn = float(result.ritz_values[i])
        rel = abs(eps_pinn - eps_exact) / eps_exact * 100.0

        # Degenerate-subspace weights for m=1,r=1 and m=2,r=1 if present.
        m1_idx = [k for k, m in enumerate(exact) if m.m == 1 and m.radial_index == 1]
        m2_idx = [k for k, m in enumerate(exact) if m.m == 2 and m.radial_index == 1]
        weight_m1 = float(np.sum(O[m1_idx, i] ** 2)) if m1_idx else 0.0
        weight_m2 = float(np.sum(O[m2_idx, i] ** 2)) if m2_idx else 0.0
        dom = int(np.argmax(np.abs(O[:, i])))
        rows.append(
            {
                "state": i + 1,
                "exact_label": mode.label,
                "m": mode.m,
                "radial_index": mode.radial_index,
                "angular": mode.angular,
                "epsilon_exact": eps_exact,
                "epsilon_PINN": eps_pinn,
                "relative_error_percent": rel,
                "weight_m1_radial1": weight_m1,
                "weight_m2_radial1": weight_m2,
                "dominant_exact_label": exact[dom].label,
                "dominant_overlap": O[dom, i],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "pinn_disk_2d_subspace_summary.csv", index=False)

    # Overlap matrix.
    overlap_df = pd.DataFrame(
        O,
        index=[m.label for m in exact],
        columns=[f"PINN_{i+1}" for i in range(args.states)],
    )
    overlap_df.to_csv(DATA_DIR / "pinn_disk_2d_subspace_overlap_matrix.csv")

    # Grid values for plotting and future reuse.
    X, Y, mask = make_plot_grid(args.n_plot_grid, args.radius)
    xy_grid = np.stack([X.ravel(), Y.ravel()], axis=1)
    psi_grid = evaluate_ritz_modes(
        model,
        result.coeffs[:, : args.states],
        xy_grid,
        device=device,
        dtype=dtype,
        radius=args.radius,
    )

    for j in range(args.states):
        # align grid sign with validation modes
        dom = int(np.argmax(np.abs(O[:, j])))
        if O[dom, j] < 0:
            psi_grid[:, j] *= -1

    grid_df = pd.DataFrame({"x": xy_grid[:, 0], "y": xy_grid[:, 1], "inside_disk": mask.ravel()})
    for j in range(args.states):
        vals = psi_grid[:, j].copy()
        vals[~mask.ravel()] = np.nan
        grid_df[f"PINN_{j+1}"] = vals
    grid_df.to_csv(DATA_DIR / "pinn_disk_2d_subspace_modes_grid.csv", index=False)

    make_figures(summary, overlap_df, X, Y, mask, psi_grid, exact, O, args)

    print("\nSaved outputs:")
    print("  data/pinn_disk_2d_subspace_summary.csv")
    print("  data/pinn_disk_2d_subspace_overlap_matrix.csv")
    print("  data/pinn_disk_2d_subspace_modes_grid.csv")
    print("  figures/pinn_disk_2d_subspace_modes.png/pdf")
    print("  figures/pinn_disk_2d_subspace_energy_comparison.png/pdf")
    print("  figures/pinn_disk_2d_subspace_degenerate_subspace.png/pdf")
    print("  figures/pinn_disk_2d_subspace_overlap_matrix.png/pdf")

    print("\nSummary")
    print("  i     exact label      epsilon_exact      epsilon_PINN       rel.err      w(m=1)")
    for _, row in summary.iterrows():
        print(
            f"{int(row['state']):3d}  {str(row['exact_label']):>14s}  "
            f"{row['epsilon_exact']:16.8f}  {row['epsilon_PINN']:16.8f}  "
            f"{row['relative_error_percent']:9.4f}%  "
            f"{row['weight_m1_radial1']:10.4f}"
        )

    print("\nInterpretation note:")
    print("  Exact Bessel modes are used only for validation/interpretation.")
    print("  For m>0, disk modes are degenerate: cos(m theta) and sin(m theta)")
    print("  span the same physical eigenspace. A PINN mode may be any orthonormal")
    print("  linear combination inside that subspace.")


def make_figures(
    summary: pd.DataFrame,
    overlap_df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    mask: np.ndarray,
    psi_grid: np.ndarray,
    exact: list[ExactDiskMode],
    O: np.ndarray,
    args: argparse.Namespace,
) -> None:
    setup_style()

    # Figure 1: learned modes.
    n = args.states
    cols = min(n, 3)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.35 * cols, 2.15 * rows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for j in range(n):
        ax = axes[j]
        Z = psi_grid[:, j].reshape(X.shape).copy()
        Z[~mask] = np.nan
        vmax = np.nanmax(np.abs(Z))
        im = ax.imshow(
            Z,
            extent=[-args.radius, args.radius, -args.radius, args.radius],
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
        )
        ax.set_aspect("equal")
        ax.set_title(fr"PINN {j+1}, $\epsilon={summary.loc[j, 'epsilon_PINN']:.3f}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        panel_label(ax, f"({chr(ord('a') + j)})")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.colorbar(im, ax=axes[:n], shrink=0.82, pad=0.02, label=r"$\psi_\theta$")
    savefig(fig, FIG_DIR / "pinn_disk_2d_subspace_modes.png")

    # Figure 2: energy comparison.
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    idx = np.arange(1, n + 1)
    ax.plot(idx, summary["epsilon_exact"], color="0.15", marker="o", label="Bessel exact")
    ax.plot(idx, summary["epsilon_PINN"], color="#c7423d", marker="s", ls="none", label="PINN/Ritz")
    ax.set_xlabel("ranked state")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_xticks(idx)
    ax.minorticks_on()
    ax.legend(loc="best")
    panel_label(ax, "(a)")
    savefig(fig, FIG_DIR / "pinn_disk_2d_subspace_energy_comparison.png")

    # Figure 3: degenerate subspace weights.
    fig, ax = plt.subplots(figsize=(3.45, 2.55), constrained_layout=True)
    ax.bar(idx - 0.15, summary["weight_m1_radial1"], width=0.3, label=r"$m=1,n=1$ subspace")
    if "weight_m2_radial1" in summary:
        ax.bar(idx + 0.15, summary["weight_m2_radial1"], width=0.3, label=r"$m=2,n=1$ subspace")
    ax.set_xlabel("PINN/Ritz state")
    ax.set_ylabel("subspace weight")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(idx)
    ax.legend(loc="best")
    panel_label(ax, "(a)")
    savefig(fig, FIG_DIR / "pinn_disk_2d_subspace_degenerate_subspace.png")

    # Figure 4: overlap matrix.
    fig, ax = plt.subplots(figsize=(3.7, 3.0), constrained_layout=True)
    mat = np.abs(O)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels([f"PINN {i}" for i in idx], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(exact)))
    ax.set_yticklabels([m.label for m in exact])
    ax.set_title(r"$|\langle \psi_{\rm exact}|\psi_{\rm PINN}\rangle|$")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    savefig(fig, FIG_DIR / "pinn_disk_2d_subspace_overlap_matrix.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PINN/Rayleigh--Ritz subspace solver for the circular quantum dot")
    parser.add_argument("--states", type=int, default=3, help="Number of low-energy Ritz states to train")
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260515)

    parser.add_argument("--n-radial", type=int, default=22, help="Radial Gauss points for training")
    parser.add_argument("--n-theta", type=int, default=72, help="Angular points for training")
    parser.add_argument("--n-validate-radial", type=int, default=30)
    parser.add_argument("--n-validate-theta", type=int, default=96)
    parser.add_argument("--n-plot-grid", type=int, default=160)

    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--fourier-features", type=int, default=5)

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=300)
    parser.add_argument("--grad-clip", type=float, default=10.0)

    parser.add_argument("--weight-orth", type=float, default=0.5)
    parser.add_argument("--weight-condition", type=float, default=1.0)
    parser.add_argument("--selection-orth-weight", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=1e-8)

    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--print-every", type=int, default=200)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    dtype = parse_dtype(args.dtype)
    device = torch.device(args.device)

    print("2D circular-disk subspace PINN/Rayleigh--Ritz")
    print(f"  root: {ROOT}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")
    print(f"  radius: {args.radius}")
    print(f"  states: {args.states}")
    print("  note: exact Bessel modes are validation references only, not training targets\n")

    exact = exact_disk_modes(args.states, radius=args.radius)
    print("Analytical references for validation:")
    for i, mode in enumerate(exact, start=1):
        print(
            f"  {i:2d}: {mode.label:>12s}  "
            f"m={mode.m}, radial={mode.radial_index}, angular={mode.angular}, "
            f"epsilon={mode.epsilon:.10f}"
        )
    print()

    model, result = train_subspace(args, device, dtype)
    save_outputs(model, result, args, device, dtype)


if __name__ == "__main__":
    main()
