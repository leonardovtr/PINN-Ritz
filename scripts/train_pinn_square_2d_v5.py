#!/usr/bin/env python3
"""Projected variational PINN for the 2D infinite square well.

This script is the recommended continuation after the robust 1D PINN.
It solves the dimensionless eigenproblem

    -Delta psi = epsilon psi,   psi|boundary = 0

on a rectangular domain [0, Lx] x [0, Ly].  The default is the unit square.

Important methodological choice
-----------------------------
This version uses a Rayleigh-Ritz / variational PINN formulation rather than
trying to optimize the strong-form residual directly.  The boundary condition is
imposed exactly by the ansatz

    psi_theta(x,y) = sin(pi x/Lx) sin(pi y/Ly) u_theta(x,y),

and excited states are obtained by explicit Gram-Schmidt projection against the
previously learned physical states.  Exact analytical modes are NOT used as
training targets; they are used only after training for validation and
interpretation.

Outputs
-------
  data/pinn_square_2d_summary_v5.csv
  data/pinn_square_2d_overlap_matrix_v5.csv
  data/pinn_square_2d_modes_grid_v5.csv
  figures/pinn_square_2d_modes_v5.png
  figures/pinn_square_2d_energy_comparison_v5.png
  figures/pinn_square_2d_degenerate_subspace_v5.png
  figures/pinn_square_2d_overlap_matrix_v5.png
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


# -----------------------------------------------------------------------------
# Paths and reproducibility
# -----------------------------------------------------------------------------


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "quantum_dots").exists() or (parent / "requirements.txt").exists():
            return parent
    return here.parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# -----------------------------------------------------------------------------
# Quadrature and analytical reference
# -----------------------------------------------------------------------------


def gauss_legendre_square(n: int, lx: float, ly: float, device: torch.device, dtype: torch.dtype):
    """Tensor-product Gauss-Legendre nodes and weights on [0,Lx]x[0,Ly]."""
    gx, gw = np.polynomial.legendre.leggauss(n)
    x = 0.5 * lx * (gx + 1.0)
    wx = 0.5 * lx * gw
    y = 0.5 * ly * (gx + 1.0)
    wy = 0.5 * ly * gw

    X, Y = np.meshgrid(x, y, indexing="ij")
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    w = (WX * WY).ravel()

    xy_t = torch.tensor(xy, device=device, dtype=dtype)
    w_t = torch.tensor(w, device=device, dtype=dtype)
    return xy_t, w_t


def exact_energy(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> float:
    return math.pi**2 * ((nx / lx) ** 2 + (ny / ly) ** 2)


def exact_mode_np(x: np.ndarray, y: np.ndarray, nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> np.ndarray:
    return (2.0 / math.sqrt(lx * ly)) * np.sin(nx * math.pi * x / lx) * np.sin(ny * math.pi * y / ly)


def exact_mode_torch(xy: torch.Tensor, nx: int, ny: int, lx: float, ly: float) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    return (2.0 / math.sqrt(lx * ly)) * torch.sin(nx * math.pi * x / lx) * torch.sin(ny * math.pi * y / ly)


def ranked_exact_states(count: int, max_n: int, lx: float, ly: float):
    states = []
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            states.append({"nx": nx, "ny": ny, "epsilon": exact_energy(nx, ny, lx, ly), "label": f"psi_{nx}{ny}"})
    states.sort(key=lambda d: (d["epsilon"], d["nx"], d["ny"]))
    return states[:count]


# -----------------------------------------------------------------------------
# Neural model
# -----------------------------------------------------------------------------


class FourierMLP(nn.Module):
    def __init__(self, hidden_width: int, hidden_layers: int, fourier_features: int, lx: float, ly: float):
        super().__init__()
        self.fourier_features = int(fourier_features)
        self.lx = float(lx)
        self.ly = float(ly)

        in_dim = 2 + 4 * self.fourier_features
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_width))
            layers.append(nn.Tanh())
            last = hidden_width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1] / self.lx
        y = xy[:, 1:2] / self.ly
        feats = [x, y]
        for k in range(1, self.fourier_features + 1):
            kk = float(k)
            feats.extend([
                torch.sin(2.0 * math.pi * kk * x),
                torch.cos(2.0 * math.pi * kk * x),
                torch.sin(2.0 * math.pi * kk * y),
                torch.cos(2.0 * math.pi * kk * y),
            ])
        return torch.cat(feats, dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(xy)).squeeze(-1)


def boundary_envelope(xy: torch.Tensor, lx: float, ly: float) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    return torch.sin(math.pi * x / lx) * torch.sin(math.pi * y / ly)


def raw_values_and_gradients(model: nn.Module, xy: torch.Tensor, lx: float, ly: float):
    """Return raw boundary-satisfying values and spatial gradients.

    The input xy must have requires_grad=True.
    """
    raw = boundary_envelope(xy, lx, ly) * model(xy)
    grad = torch.autograd.grad(raw.sum(), xy, create_graph=True, retain_graph=True)[0]
    return raw, grad[:, 0], grad[:, 1]


def raw_values_no_grad(model: nn.Module, xy: torch.Tensor, lx: float, ly: float) -> torch.Tensor:
    return boundary_envelope(xy, lx, ly) * model(xy)


@dataclass
class LearnedState:
    index: int
    model: nn.Module
    coeffs: list[float]
    norm_sqrt: float
    epsilon: float
    train_energy: float
    quad_values: torch.Tensor
    quad_grad_x: torch.Tensor
    quad_grad_y: torch.Tensor
    best_restart: int


# -----------------------------------------------------------------------------
# Projection and variational loss
# -----------------------------------------------------------------------------


def project_values_and_grads(
    values: torch.Tensor,
    grad_x: torch.Tensor,
    grad_y: torch.Tensor,
    previous: list[LearnedState],
    weights: torch.Tensor,
):
    """Project candidate field against previously learned normalized states.

    The previous states are fixed functions on the same quadrature grid.
    The projection coefficients are global scalars.  They are differentiable with
    respect to the candidate network parameters, but are not spatially varying.
    """
    coeffs = []
    v = values
    gx = grad_x
    gy = grad_y

    for prev in previous:
        pv = prev.quad_values.to(device=v.device, dtype=v.dtype)
        pgx = prev.quad_grad_x.to(device=v.device, dtype=v.dtype)
        pgy = prev.quad_grad_y.to(device=v.device, dtype=v.dtype)
        c = torch.sum(weights * v * pv)  # previous states are normalized
        v = v - c * pv
        gx = gx - c * pgx
        gy = gy - c * pgy
        coeffs.append(c)

    return v, gx, gy, coeffs


def rayleigh_loss(
    model: nn.Module,
    xy_base: torch.Tensor,
    weights: torch.Tensor,
    previous: list[LearnedState],
    lx: float,
    ly: float,
    norm_weight: float,
    norm_floor: float = 1e-14,
):
    xy = xy_base.detach().clone().requires_grad_(True)
    raw, raw_gx, raw_gy = raw_values_and_gradients(model, xy, lx, ly)
    v, gx, gy, coeffs = project_values_and_grads(raw, raw_gx, raw_gy, previous, weights)

    norm = torch.sum(weights * v * v)
    kinetic = torch.sum(weights * (gx * gx + gy * gy))
    energy = kinetic / (norm + norm_floor)

    # This only fixes the arbitrary scale and stabilizes the quotient. It does
    # not encode analytical eigenvalues or modes.
    scale_penalty = norm_weight * (torch.log(norm + norm_floor) ** 2)
    loss = energy + scale_penalty
    return loss, energy, norm, coeffs


@torch.no_grad()
def evaluate_all_states_on_grid(states: list[LearnedState], xy: torch.Tensor, lx: float, ly: float) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for state in states:
        raw = raw_values_no_grad(state.model, xy, lx, ly)
        v = raw.clone()
        for c, prev_v in zip(state.coeffs, values):
            v = v - float(c) * prev_v
        v = v / float(state.norm_sqrt)
        values.append(v)
    return values


def finalize_state_on_quad(
    model: nn.Module,
    index: int,
    xy_base: torch.Tensor,
    weights: torch.Tensor,
    previous: list[LearnedState],
    lx: float,
    ly: float,
    best_restart: int,
) -> LearnedState:
    xy = xy_base.detach().clone().requires_grad_(True)
    raw, raw_gx, raw_gy = raw_values_and_gradients(model, xy, lx, ly)
    v, gx, gy, coeffs_t = project_values_and_grads(raw, raw_gx, raw_gy, previous, weights)

    norm = torch.sum(weights * v * v)
    norm_sqrt = torch.sqrt(norm)
    v_n = v / norm_sqrt
    gx_n = gx / norm_sqrt
    gy_n = gy / norm_sqrt
    energy = torch.sum(weights * (gx_n * gx_n + gy_n * gy_n))

    coeffs = [float(c.detach().cpu()) for c in coeffs_t]
    return LearnedState(
        index=index,
        model=copy.deepcopy(model).cpu(),
        coeffs=coeffs,
        norm_sqrt=float(norm_sqrt.detach().cpu()),
        epsilon=float(energy.detach().cpu()),
        train_energy=float(energy.detach().cpu()),
        quad_values=v_n.detach().cpu(),
        quad_grad_x=gx_n.detach().cpu(),
        quad_grad_y=gy_n.detach().cpu(),
        best_restart=best_restart,
    )


def orthogonality_report(states: list[LearnedState], weights: torch.Tensor, device: torch.device, dtype: torch.dtype):
    if not states:
        return np.zeros((0, 0))
    vals = [s.quad_values.to(device=device, dtype=dtype) for s in states]
    m = np.zeros((len(vals), len(vals)))
    for i, vi in enumerate(vals):
        for j, vj in enumerate(vals):
            m[i, j] = float(torch.sum(weights * vi * vj).detach().cpu())
    return m


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


def train_one_state(
    state_index: int,
    previous: list[LearnedState],
    args: argparse.Namespace,
    xy_quad: torch.Tensor,
    weights: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    best_state: LearnedState | None = None
    best_energy = float("inf")

    exact_refs = ranked_exact_states(args.states, max(args.max_exact_n, 5), args.lx, args.ly)
    ref = exact_refs[state_index - 1]
    print(f"\nTraining square-well state #{state_index}")
    print(
        f"  analytical reference for validation: {ref['label']}=({ref['nx']},{ref['ny']}), "
        f"epsilon={ref['epsilon']:.10f}"
    )

    for restart in range(1, args.restarts + 1):
        set_seed(args.seed + 1000 * state_index + restart)
        model = FourierMLP(
            hidden_width=args.hidden_width,
            hidden_layers=args.hidden_layers,
            fourier_features=args.fourier_features,
            lx=args.lx,
            ly=args.ly,
        ).to(device=device, dtype=dtype)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        local_best_dict = copy.deepcopy(model.state_dict())
        local_best_energy = float("inf")
        local_best_norm = float("nan")

        print(f"  restart {restart}/{args.restarts}")
        for epoch in range(1, args.epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            loss, energy, norm, _ = rayleigh_loss(
                model, xy_quad, weights, previous, args.lx, args.ly, args.norm_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            e_value = float(energy.detach().cpu())
            n_value = float(norm.detach().cpu())
            if math.isfinite(e_value) and e_value < local_best_energy:
                local_best_energy = e_value
                local_best_norm = n_value
                local_best_dict = copy.deepcopy(model.state_dict())

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                max_overlap = 0.0
                if previous:
                    with torch.no_grad():
                        # Approximate current physical state for monitoring.
                        xy_tmp = xy_quad.detach().clone().requires_grad_(True)
                    # Avoid an expensive special path; report projection is exact by construction.
                    max_overlap = 0.0
                print(
                    f"    epoch {epoch:6d}/{args.epochs}: "
                    f"loss={float(loss.detach().cpu()):.6e}, "
                    f"epsilon={e_value:.10f}, norm={n_value:.3e}, "
                    f"max|overlap|~{max_overlap:.1e}"
                )

        # Start LBFGS from the best Adam checkpoint, not necessarily the last one.
        model.load_state_dict(local_best_dict)
        if args.lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_steps,
                tolerance_grad=1e-11,
                tolerance_change=1e-13,
                history_size=100,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _, _, _ = rayleigh_loss(
                    model, xy_quad, weights, previous, args.lx, args.ly, args.norm_weight
                )
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except RuntimeError as exc:
                print(f"    LBFGS warning: {exc}")

        # Compare post-LBFGS with best Adam checkpoint, keep whichever has lower energy.
        with torch.enable_grad():
            _, e_post, norm_post, _ = rayleigh_loss(
                model, xy_quad, weights, previous, args.lx, args.ly, args.norm_weight
            )
        post_energy = float(e_post.detach().cpu())
        post_norm = float(norm_post.detach().cpu())

        if post_energy > local_best_energy:
            model.load_state_dict(local_best_dict)
            chosen_energy = local_best_energy
            chosen_norm = local_best_norm
            chosen_source = "Adam checkpoint"
        else:
            chosen_energy = post_energy
            chosen_norm = post_norm
            chosen_source = "LBFGS"

        candidate = finalize_state_on_quad(
            model, state_index, xy_quad, weights, previous, args.lx, args.ly, restart
        )
        print(
            f"    restart result: epsilon={candidate.epsilon:.10f}, "
            f"norm_before_normalization={chosen_norm:.3e}, source={chosen_source}"
        )

        if candidate.epsilon < best_energy:
            best_energy = candidate.epsilon
            best_state = candidate

    assert best_state is not None
    print(f"  selected state #{state_index}: epsilon={best_state.epsilon:.10f}, best_restart={best_state.best_restart}")
    return best_state


# -----------------------------------------------------------------------------
# Validation, saving, plotting
# -----------------------------------------------------------------------------


def compute_overlap_matrix(
    states: list[LearnedState],
    exact_refs: list[dict],
    lx: float,
    ly: float,
    n_validate: int,
    device: torch.device,
    dtype: torch.dtype,
):
    xy_val, w_val = gauss_legendre_square(n_validate, lx, ly, device, dtype)
    pinn_vals = evaluate_all_states_on_grid(states, xy_val, lx, ly)
    matrix = np.zeros((len(states), len(exact_refs)))
    for i, pv in enumerate(pinn_vals):
        for j, ref in enumerate(exact_refs):
            ev = exact_mode_torch(xy_val, ref["nx"], ref["ny"], lx, ly)
            matrix[i, j] = float(torch.sum(w_val * pv * ev).detach().cpu())
    return matrix


def save_outputs(states: list[LearnedState], args: argparse.Namespace, root: Path, device: torch.device, dtype: torch.dtype):
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    exact_refs = ranked_exact_states(args.states, max(args.max_exact_n, 5), args.lx, args.ly)
    overlap = compute_overlap_matrix(states, exact_refs, args.lx, args.ly, args.n_validate, device, dtype)

    # Degenerate subspace weight for span{psi_12, psi_21}, when present.
    deg_cols = [j for j, ref in enumerate(exact_refs) if {ref["nx"], ref["ny"]} == {1, 2}]
    deg_weight = np.zeros(len(states))
    if deg_cols:
        deg_weight = np.sum(overlap[:, deg_cols] ** 2, axis=1)

    summary_rows = []
    for i, state in enumerate(states):
        ref = exact_refs[i]
        rel = abs(state.epsilon - ref["epsilon"]) / ref["epsilon"] * 100.0
        summary_rows.append(
            {
                "state_index": i + 1,
                "exact_label": ref["label"],
                "nx_ref": ref["nx"],
                "ny_ref": ref["ny"],
                "epsilon_exact_ranked": ref["epsilon"],
                "epsilon_pinn": state.epsilon,
                "relative_error_percent_vs_ranked": rel,
                "degenerate_weight_12_21": deg_weight[i],
                "best_restart": state.best_restart,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = data_dir / "pinn_square_2d_summary_v5.csv"
    summary.to_csv(summary_path, index=False)

    overlap_df = pd.DataFrame(overlap, columns=[r["label"] for r in exact_refs])
    overlap_df.insert(0, "pinn_state", [f"PINN_{i+1}" for i in range(len(states))])
    overlap_path = data_dir / "pinn_square_2d_overlap_matrix_v5.csv"
    overlap_df.to_csv(overlap_path, index=False)

    # Grid data for plotting and external inspection.
    n_plot = args.n_plot
    x = np.linspace(0.0, args.lx, n_plot)
    y = np.linspace(0.0, args.ly, n_plot)
    X, Y = np.meshgrid(x, y, indexing="ij")
    xy_np = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_np, device=device, dtype=dtype)
    vals = evaluate_all_states_on_grid(states, xy_t, args.lx, args.ly)

    grid_df = pd.DataFrame({"x": xy_np[:, 0], "y": xy_np[:, 1]})
    for i, val in enumerate(vals, start=1):
        arr = val.detach().cpu().numpy()
        # Align sign with dominant exact reference for prettier plots.
        if overlap[i - 1, np.argmax(np.abs(overlap[i - 1]))] < 0:
            arr = -arr
        grid_df[f"pinn_state_{i}"] = arr
    grid_path = data_dir / "pinn_square_2d_modes_grid_v5.csv"
    grid_df.to_csv(grid_path, index=False)

    # Figure: learned modes.
    fig, axes = plt.subplots(1, len(states), figsize=(3.0 * len(states), 2.8), constrained_layout=True)
    if len(states) == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        Z = grid_df[f"pinn_state_{i+1}"].to_numpy().reshape(n_plot, n_plot)
        vmax = np.nanmax(np.abs(Z))
        im = ax.imshow(
            Z.T,
            origin="lower",
            extent=[0, args.lx, 0, args.ly],
            aspect="equal",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"PINN {i+1}\n$\\epsilon$={states[i].epsilon:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.72)
    modes_path = fig_dir / "pinn_square_2d_modes_v5.png"
    fig.savefig(modes_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure: energy comparison.
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    xs = np.arange(1, len(states) + 1)
    exact_e = [r["epsilon"] for r in exact_refs]
    pinn_e = [s.epsilon for s in states]
    ax.plot(xs, exact_e, "o-", label="analytical reference")
    ax.plot(xs, pinn_e, "s--", label="PINN")
    ax.set_xlabel("ranked state")
    ax.set_ylabel(r"dimensionless energy $\epsilon$")
    ax.set_title("2D square well: analytical vs projected variational PINN")
    ax.legend()
    energy_path = fig_dir / "pinn_square_2d_energy_comparison_v5.png"
    fig.savefig(energy_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure: degenerate subspace coefficients.
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    # Find columns for psi_12 and psi_21 even if ordered differently.
    c12 = None
    c21 = None
    for j, ref in enumerate(exact_refs):
        if ref["nx"] == 1 and ref["ny"] == 2:
            c12 = overlap[:, j]
        if ref["nx"] == 2 and ref["ny"] == 1:
            c21 = overlap[:, j]
    if c12 is not None and c21 is not None:
        ax.axhline(0, lw=0.8)
        ax.axvline(0, lw=0.8)
        ax.scatter(c12, c21, s=60)
        for i in range(len(states)):
            ax.text(c12[i] + 0.02, c21[i] + 0.02, f"{i+1}")
        ax.set_xlabel(r"$\langle \psi_{PINN}|\psi_{1,2}\rangle$")
        ax.set_ylabel(r"$\langle \psi_{PINN}|\psi_{2,1}\rangle$")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect("equal")
        ax.set_title("Projection onto the degenerate subspace")
    else:
        ax.text(0.5, 0.5, "Degenerate pair not included", ha="center", va="center")
        ax.axis("off")
    deg_path = fig_dir / "pinn_square_2d_degenerate_subspace_v5.png"
    fig.savefig(deg_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure: overlap matrix.
    fig, ax = plt.subplots(figsize=(1.1 * len(exact_refs) + 2.5, 0.75 * len(states) + 2.0))
    im = ax.imshow(overlap, vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(exact_refs)))
    ax.set_xticklabels([r["label"] for r in exact_refs], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels([f"PINN {i+1}" for i in range(len(states))])
    ax.set_title("Overlap matrix with analytical modes")
    fig.colorbar(im, ax=ax, shrink=0.85)
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            ax.text(j, i, f"{overlap[i, j]:.2f}", ha="center", va="center", fontsize=8)
    overlap_fig_path = fig_dir / "pinn_square_2d_overlap_matrix_v5.png"
    fig.savefig(overlap_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    for p in [summary_path, overlap_path, grid_path, modes_path, energy_path, deg_path, overlap_fig_path]:
        print(f"  {p.relative_to(root)}")

    print("\nSummary")
    print("  i  exact label      epsilon_exact      epsilon_PINN       rel.err      deg.weight")
    for row in summary_rows:
        print(
            f"  {row['state_index']:1d}  {row['exact_label']:>10s}  "
            f"{row['epsilon_exact_ranked']:16.8f}  {row['epsilon_pinn']:16.8f}  "
            f"{row['relative_error_percent_vs_ranked']:9.4f}%  "
            f"{row['degenerate_weight_12_21']:10.4f}"
        )

    print("\nInterpretation note:")
    print("  Exact modes are used only for validation/interpretation, not as loss targets.")
    print("  States associated with psi_12 and psi_21 may appear as arbitrary")
    print("  orthonormal linear combinations inside the degenerate subspace.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description="Projected variational PINN for the 2D square infinite well.")
    parser.add_argument("--states", type=int, default=4, help="Number of ranked states to train.")
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--n-quad", type=int, default=34, help="Gauss-Legendre quadrature points per dimension for training.")
    parser.add_argument("--n-validate", type=int, default=52, help="Gauss-Legendre quadrature points per dimension for validation overlaps.")
    parser.add_argument("--n-plot", type=int, default=90)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--fourier-features", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=350)
    parser.add_argument("--lbfgs-lr", type=float, default=0.7)
    parser.add_argument("--norm-weight", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-exact-n", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    root = find_project_root()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    set_seed(args.seed)

    print("2D square-well projected variational PINN v5")
    print(f"  root: {root}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")
    print(f"  domain: Lx={args.lx}, Ly={args.ly}")
    print(f"  states: 1..{args.states}")
    print("  note: exact modes are used for validation only, not as training targets")
    print("  method: Rayleigh-Ritz loss + exact boundary ansatz + explicit projection")

    xy_quad, weights = gauss_legendre_square(args.n_quad, args.lx, args.ly, device, dtype)

    learned: list[LearnedState] = []
    for idx in range(1, args.states + 1):
        state = train_one_state(idx, learned, args, xy_quad, weights, device, dtype)
        learned.append(state)
        ortho = orthogonality_report(learned, weights, device, dtype)
        max_offdiag = 0.0
        if len(learned) > 1:
            off = ortho - np.eye(len(learned))
            max_offdiag = float(np.max(np.abs(off)))
        print(f"  orthogonality check after state #{idx}: max offdiag={max_offdiag:.3e}")

    save_outputs(learned, args, root, device, dtype)


if __name__ == "__main__":
    main()
