#!/usr/bin/env python3
"""Subspace Rayleigh-Ritz PINN for degeneracy lifting in the 2D rectangular well.

This script is the natural continuation after the square-well subspace script.
It trains a K-dimensional boundary-satisfying neural subspace for each aspect
ratio eta = Ly/Lx and compares the learned Ritz eigenvalues with the analytical
rectangular-well energies.

Physics target
--------------
For an infinite rectangular well,

    -laplacian psi = epsilon psi,    psi|boundary = 0,

with Lx = 1 and Ly = eta, the analytical energies are

    epsilon_{nx,ny}(eta) = pi^2 * (nx^2/Lx^2 + ny^2/Ly^2).

The square eta=1 has the degeneracy

    epsilon_{1,2} = epsilon_{2,1} = 5*pi^2.

For eta != 1, this degeneracy is lifted. The script overlays PINN/Ritz points
against the analytical splitting curves.

Important methodological note
-----------------------------
The analytical modes/energies are NOT used as training targets. Training uses:
  - a boundary-satisfying ansatz;
  - a Rayleigh-Ritz variational objective;
  - orthonormalization of the learned subspace by quadrature/Cholesky.
The analytical solution is used only after training for validation, matching,
and interpretation.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


# -----------------------------------------------------------------------------
# Project / reproducibility utilities
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
# Quadrature and exact rectangular-well references
# -----------------------------------------------------------------------------


def gauss_legendre_rect(n: int, lx: float, ly: float, device, dtype):
    """Tensor-product Gauss-Legendre quadrature on [0,Lx]x[0,Ly]."""
    g, gw = np.polynomial.legendre.leggauss(n)
    x = 0.5 * lx * (g + 1.0)
    y = 0.5 * ly * (g + 1.0)
    wx = 0.5 * lx * gw
    wy = 0.5 * ly * gw
    X, Y = np.meshgrid(x, y, indexing="ij")
    WX, WY = np.meshgrid(wx, wy, indexing="ij")
    xy = np.column_stack([X.ravel(), Y.ravel()])
    w = (WX * WY).ravel()
    return torch.tensor(xy, device=device, dtype=dtype), torch.tensor(w, device=device, dtype=dtype)


def exact_energy(nx: int, ny: int, lx: float, ly: float) -> float:
    return math.pi**2 * ((nx / lx) ** 2 + (ny / ly) ** 2)


def exact_mode_torch(xy: torch.Tensor, nx: int, ny: int, lx: float, ly: float) -> torch.Tensor:
    x = xy[:, 0]
    y = xy[:, 1]
    return (2.0 / math.sqrt(lx * ly)) * torch.sin(nx * math.pi * x / lx) * torch.sin(ny * math.pi * y / ly)


def ranked_exact_states(k: int, max_n: int, lx: float, ly: float):
    states = []
    for nx in range(1, max_n + 1):
        for ny in range(1, max_n + 1):
            states.append(
                {
                    "nx": nx,
                    "ny": ny,
                    "epsilon": exact_energy(nx, ny, lx, ly),
                    "label": f"psi_{nx}{ny}",
                }
            )
    states.sort(key=lambda d: (d["epsilon"], d["nx"], d["ny"]))
    return states[:k]


# -----------------------------------------------------------------------------
# Neural ansatz and subspace Rayleigh-Ritz machinery
# -----------------------------------------------------------------------------


class MultiOutputBoundedMLP(nn.Module):
    """K-output MLP with Fourier features.

    The network represents only the smooth multiplier u_theta(x,y). The final
    physical fields are envelope(x,y)*u_theta(x,y), so the Dirichlet boundary
    condition is imposed exactly.
    """

    def __init__(
        self,
        k: int,
        hidden_width: int,
        hidden_layers: int,
        fourier_features: int,
        output_scale: float,
        lx: float,
        ly: float,
    ):
        super().__init__()
        self.k = int(k)
        self.fourier_features = int(fourier_features)
        self.output_scale = float(output_scale)
        self.lx = float(lx)
        self.ly = float(ly)

        in_dim = 2 + 4 * self.fourier_features
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_width))
            layers.append(nn.Tanh())
            last = hidden_width
        layers.append(nn.Linear(last, self.k))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        final = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(final.bias, -0.2, 0.2)

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        # Dimensionless coordinates are important when Ly != Lx.
        x = xy[:, 0:1] / self.lx
        y = xy[:, 1:2] / self.ly
        feats = [x, y]
        for kk in range(1, self.fourier_features + 1):
            k = float(kk)
            feats.extend(
                [
                    torch.sin(2 * math.pi * k * x),
                    torch.cos(2 * math.pi * k * x),
                    torch.sin(2 * math.pi * k * y),
                    torch.cos(2 * math.pi * k * y),
                ]
            )
        return torch.cat(feats, dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.output_scale * torch.tanh(self.net(self.features(xy)))


def envelope(xy: torch.Tensor, lx: float, ly: float) -> torch.Tensor:
    """Boundary-satisfying envelope for the rectangular infinite well."""
    return torch.sin(math.pi * xy[:, 0] / lx) * torch.sin(math.pi * xy[:, 1] / ly)


def raw_fields_and_grads(model: nn.Module, xy_base: torch.Tensor, lx: float, ly: float):
    """Return raw fields F_j and spatial gradients dF_j/dx,dF_j/dy.

    No analytical eigenfunction is used here. The only embedded physics is the
    zero-boundary envelope.
    """
    xy = xy_base.detach().clone().requires_grad_(True)
    F = envelope(xy, lx, ly).unsqueeze(1) * model(xy)  # N x K

    grads_x = []
    grads_y = []
    for j in range(F.shape[1]):
        g = torch.autograd.grad(F[:, j].sum(), xy, create_graph=True, retain_graph=True)[0]
        grads_x.append(g[:, 0])
        grads_y.append(g[:, 1])
    Fx = torch.stack(grads_x, dim=1)
    Fy = torch.stack(grads_y, dim=1)
    return F, Fx, Fy


def orthonormalize(F: torch.Tensor, Fx: torch.Tensor, Fy: torch.Tensor, w: torch.Tensor, eps: float):
    """Quadrature orthonormalization and Ritz Hamiltonian.

    If S = F^T W F, then Q = F A with A = inv(L^T), where S = L L^T.
    This gives Q^T W Q = I. The Hamiltonian matrix for -laplacian with
    Dirichlet boundary conditions is H_ij = int grad Q_i . grad Q_j dA.
    """
    K = F.shape[1]
    Wf = w[:, None] * F
    S = F.T @ Wf
    S = 0.5 * (S + S.T) + eps * torch.eye(K, device=F.device, dtype=F.dtype)

    L = torch.linalg.cholesky(S)
    I = torch.eye(K, device=F.device, dtype=F.dtype)
    A = torch.linalg.solve_triangular(L.T, I, upper=True)

    Q = F @ A
    Qx = Fx @ A
    Qy = Fy @ A

    H = Qx.T @ (w[:, None] * Qx) + Qy.T @ (w[:, None] * Qy)
    H = 0.5 * (H + H.T)
    return Q, Qx, Qy, H, S


def subspace_loss(model: nn.Module, xy: torch.Tensor, w: torch.Tensor, args):
    F, Fx, Fy = raw_fields_and_grads(model, xy, args.lx, args.ly)
    _, _, _, H, S = orthonormalize(F, Fx, Fy, w, args.cholesky_eps)
    eigvals = torch.linalg.eigvalsh(H)
    loss = torch.sum(eigvals)

    # Mild protection against nearly linearly dependent raw outputs.
    sign, logabsdet = torch.linalg.slogdet(S)
    loss = loss + args.gram_penalty * torch.relu(-logabsdet)
    return loss, eigvals, H


def validation_eigs_grad(model: nn.Module, xy: torch.Tensor, w: torch.Tensor, args):
    F, Fx, Fy = raw_fields_and_grads(model, xy, args.lx, args.ly)
    Q, _, _, H, _ = orthonormalize(F, Fx, Fy, w, args.cholesky_eps)
    eigvals, eigvecs = torch.linalg.eigh(H)
    modes = Q @ eigvecs
    return eigvals, modes, H, eigvecs


@dataclass
class EtaResult:
    eta: float
    lx: float
    ly: float
    eigvals: np.ndarray
    overlap: np.ndarray
    refs: list[dict]
    best_model_state: dict


# -----------------------------------------------------------------------------
# Training for one eta
# -----------------------------------------------------------------------------


def train_one_eta(args, device, dtype, eta: float) -> nn.Module:
    args = copy.copy(args)
    args.lx = float(args.lx)
    args.ly = float(args.lx * eta)

    xy, w = gauss_legendre_rect(args.n_quad, args.lx, args.ly, device, dtype)
    xyv, wv = gauss_legendre_rect(args.n_validate, args.lx, args.ly, device, dtype)

    best_global = None
    best_score = float("inf")

    for restart in range(1, args.restarts + 1):
        set_seed(args.seed + int(round(1000 * eta)) + restart)
        model = MultiOutputBoundedMLP(
            args.states,
            args.hidden_width,
            args.hidden_layers,
            args.fourier_features,
            args.output_scale,
            args.lx,
            args.ly,
        ).to(device=device, dtype=dtype)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_state = copy.deepcopy(model.state_dict())
        best_val_sum = float("inf")
        best_vals = None

        if not args.quiet:
            print(f"    restart {restart}/{args.restarts}")

        for epoch in range(1, args.epochs + 1):
            opt.zero_grad(set_to_none=True)
            loss, eigs, _ = subspace_loss(model, xy, w, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if epoch == 1 or epoch % args.check_every == 0 or epoch == args.epochs:
                vals, _, _, _ = validation_eigs_grad(model, xyv, wv, args)
                val_sum = float(torch.sum(vals).detach().cpu())
                if math.isfinite(val_sum) and val_sum < best_val_sum:
                    best_val_sum = val_sum
                    best_vals = vals.detach().cpu().numpy()
                    best_state = copy.deepcopy(model.state_dict())

            if (not args.quiet) and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
                train_vals = eigs.detach().cpu().numpy()
                val_vals = vals.detach().cpu().numpy()
                print(
                    f"      epoch {epoch:5d}/{args.epochs}: "
                    f"train={np.array2string(train_vals, precision=3)}, "
                    f"val={np.array2string(val_vals, precision=3)}"
                )

        model.load_state_dict(best_state)

        if args.lbfgs_steps > 0:
            before_state = copy.deepcopy(model.state_dict())
            before_vals, _, _, _ = validation_eigs_grad(model, xyv, wv, args)
            before_sum = float(torch.sum(before_vals).detach().cpu())
            lbfgs = torch.optim.LBFGS(
                model.parameters(),
                lr=args.lbfgs_lr,
                max_iter=args.lbfgs_steps,
                line_search_fn="strong_wolfe",
                history_size=80,
            )

            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _, _ = subspace_loss(model, xy, w, args)
                loss.backward()
                return loss

            try:
                lbfgs.step(closure)
            except RuntimeError as exc:
                print(f"      LBFGS warning: {exc}")

            after_vals, _, _, _ = validation_eigs_grad(model, xyv, wv, args)
            after_sum = float(torch.sum(after_vals).detach().cpu())
            if after_sum > before_sum:
                model.load_state_dict(before_state)
                chosen_sum = before_sum
                chosen_vals = before_vals.detach().cpu().numpy()
                source = "Adam checkpoint"
            else:
                chosen_sum = after_sum
                chosen_vals = after_vals.detach().cpu().numpy()
                source = "LBFGS"
        else:
            chosen_sum = best_val_sum
            chosen_vals = best_vals
            source = "Adam checkpoint"

        if not args.quiet:
            print(f"      restart result: val={np.array2string(chosen_vals, precision=6)}, sum={chosen_sum:.6f}, source={source}")

        if chosen_sum < best_score:
            best_score = chosen_sum
            best_global = copy.deepcopy(model.state_dict())

    model = MultiOutputBoundedMLP(
        args.states,
        args.hidden_width,
        args.hidden_layers,
        args.fourier_features,
        args.output_scale,
        args.lx,
        args.ly,
    ).to(device=device, dtype=dtype)
    model.load_state_dict(best_global)
    return model


def evaluate_one_eta(model: nn.Module, args, device, dtype, eta: float) -> EtaResult:
    args = copy.copy(args)
    args.lx = float(args.lx)
    args.ly = float(args.lx * eta)

    xyv, wv = gauss_legendre_rect(args.n_validate, args.lx, args.ly, device, dtype)
    eigvals, modes, _, _ = validation_eigs_grad(model, xyv, wv, args)
    eig_np = eigvals.detach().cpu().numpy()

    refs = ranked_exact_states(args.max_exact_modes, max(args.max_exact_n, 6), args.lx, args.ly)
    overlap = np.zeros((args.states, len(refs)))
    for i in range(args.states):
        for j, ref in enumerate(refs):
            ex = exact_mode_torch(xyv, ref["nx"], ref["ny"], args.lx, args.ly)
            overlap[i, j] = float(torch.sum(wv * modes[:, i] * ex).detach().cpu())

    return EtaResult(
        eta=float(eta),
        lx=float(args.lx),
        ly=float(args.ly),
        eigvals=eig_np,
        overlap=overlap,
        refs=refs,
        best_model_state=copy.deepcopy(model.state_dict()),
    )


# -----------------------------------------------------------------------------
# Matching and output
# -----------------------------------------------------------------------------


def parse_etas(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("No eta values supplied.")
    return vals


def find_ref_index(refs: Sequence[dict], nx: int, ny: int) -> int | None:
    for j, ref in enumerate(refs):
        if ref["nx"] == nx and ref["ny"] == ny:
            return j
    return None


def matched_energy(result: EtaResult, nx: int, ny: int) -> tuple[float, int, float, float]:
    """Return PINN energy matched by maximum overlap to exact psi_{nx,ny}.

    Returns: (energy, pinn_state_index_1based, overlap, overlap_weight).
    """
    j = find_ref_index(result.refs, nx, ny)
    if j is None:
        return float("nan"), -1, float("nan"), float("nan")
    weights = result.overlap[:, j] ** 2
    i = int(np.argmax(weights))
    return float(result.eigvals[i]), i + 1, float(result.overlap[i, j]), float(weights[i])


def save_outputs(results: list[EtaResult], args, root: Path, device, dtype) -> None:
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    rows = []
    ranked_rows = []
    overlap_rows = []

    for result in results:
        eta = result.eta
        lx = result.lx
        ly = result.ly
        e12_exact = exact_energy(1, 2, lx, ly)
        e21_exact = exact_energy(2, 1, lx, ly)
        e12_pinn, idx12, ov12, w12 = matched_energy(result, 1, 2)
        e21_pinn, idx21, ov21, w21 = matched_energy(result, 2, 1)

        rows.append(
            {
                "eta": eta,
                "Lx": lx,
                "Ly": ly,
                "E12_exact": e12_exact,
                "E21_exact": e21_exact,
                "splitting_exact_E12_minus_E21": e12_exact - e21_exact,
                "E12_PINN_matched": e12_pinn,
                "E21_PINN_matched": e21_pinn,
                "splitting_PINN_E12_minus_E21": e12_pinn - e21_pinn,
                "PINN_state_for_psi12": idx12,
                "PINN_state_for_psi21": idx21,
                "overlap_psi12": ov12,
                "overlap_weight_psi12": w12,
                "overlap_psi21": ov21,
                "overlap_weight_psi21": w21,
                "relative_error_E12_percent": abs(e12_pinn - e12_exact) / e12_exact * 100.0,
                "relative_error_E21_percent": abs(e21_pinn - e21_exact) / e21_exact * 100.0,
            }
        )

        for i, e in enumerate(result.eigvals[: args.states], start=1):
            ref = result.refs[i - 1] if i - 1 < len(result.refs) else None
            ranked_rows.append(
                {
                    "eta": eta,
                    "pinn_rank": i,
                    "epsilon_PINN": e,
                    "ranked_exact_label": ref["label"] if ref else "",
                    "ranked_exact_nx": ref["nx"] if ref else np.nan,
                    "ranked_exact_ny": ref["ny"] if ref else np.nan,
                    "ranked_exact_epsilon": ref["epsilon"] if ref else np.nan,
                    "relative_error_percent_vs_ranked": abs(e - ref["epsilon"]) / ref["epsilon"] * 100.0 if ref else np.nan,
                }
            )

        for i in range(result.overlap.shape[0]):
            for j, ref in enumerate(result.refs):
                overlap_rows.append(
                    {
                        "eta": eta,
                        "pinn_rank": i + 1,
                        "exact_label": ref["label"],
                        "nx": ref["nx"],
                        "ny": ref["ny"],
                        "overlap": result.overlap[i, j],
                        "overlap_weight": result.overlap[i, j] ** 2,
                    }
                )

    split_df = pd.DataFrame(rows)
    ranked_df = pd.DataFrame(ranked_rows)
    overlap_df = pd.DataFrame(overlap_rows)

    split_path = data_dir / "pinn_rectangle_2d_splitting.csv"
    ranked_path = data_dir / "pinn_rectangle_2d_ranked_spectrum.csv"
    overlap_path = data_dir / "pinn_rectangle_2d_overlaps.csv"
    split_df.to_csv(split_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)
    overlap_df.to_csv(overlap_path, index=False)

    # Analytical curves for splitting.
    eta_curve = np.linspace(min(split_df["eta"]) * 0.98, max(split_df["eta"]) * 1.02, 400)
    lx0 = args.lx
    e12_curve = np.array([exact_energy(1, 2, lx0, lx0 * eta) for eta in eta_curve])
    e21_curve = np.array([exact_energy(2, 1, lx0, lx0 * eta) for eta in eta_curve])

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(eta_curve, e12_curve, lw=2, label=r"analytical $\epsilon_{1,2}$")
    ax.plot(eta_curve, e21_curve, lw=2, label=r"analytical $\epsilon_{2,1}$")
    ax.scatter(split_df["eta"], split_df["E12_PINN_matched"], marker="o", s=55, label=r"PINN/Ritz matched to $\psi_{1,2}$")
    ax.scatter(split_df["eta"], split_df["E21_PINN_matched"], marker="s", s=55, label=r"PINN/Ritz matched to $\psi_{2,1}$")
    ax.axvline(1.0, ls="--", lw=1.0)
    ax.set_xlabel(r"aspect ratio $\eta=L_y/L_x$")
    ax.set_ylabel(r"dimensionless energy $\epsilon$")
    ax.set_title("Geometry-induced degeneracy lifting: square to rectangle")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    split_fig_path = fig_dir / "pinn_rectangle_2d_splitting.png"
    fig.savefig(split_fig_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Plot the splitting itself.
    fig, ax = plt.subplots(figsize=(6.5, 3.9))
    ax.plot(eta_curve, e12_curve - e21_curve, lw=2, label=r"analytical $\Delta=\epsilon_{1,2}-\epsilon_{2,1}$")
    ax.scatter(split_df["eta"], split_df["splitting_PINN_E12_minus_E21"], s=55, label="PINN/Ritz")
    ax.axhline(0.0, ls="--", lw=1.0)
    ax.axvline(1.0, ls="--", lw=1.0)
    ax.set_xlabel(r"aspect ratio $\eta=L_y/L_x$")
    ax.set_ylabel(r"splitting $\Delta\epsilon$")
    ax.set_title("Degeneracy lifting vanishes at the square geometry")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    delta_fig_path = fig_dir / "pinn_rectangle_2d_splitting_delta.png"
    fig.savefig(delta_fig_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Ranked spectrum comparison, one mini-panel per eta.
    n_eta = len(results)
    fig, axes = plt.subplots(1, n_eta, figsize=(3.0 * n_eta, 3.3), sharey=True, constrained_layout=True)
    if n_eta == 1:
        axes = [axes]
    for ax, result in zip(axes, results):
        xs = np.arange(1, args.states + 1)
        exact_ranked = [r["epsilon"] for r in result.refs[: args.states]]
        ax.plot(xs, exact_ranked, "o-", label="exact" if ax is axes[0] else None)
        ax.plot(xs, result.eigvals[: args.states], "s--", label="PINN" if ax is axes[0] else None)
        ax.set_title(rf"$\eta={result.eta:.2f}$")
        ax.set_xlabel("rank")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel(r"$\epsilon$")
    axes[0].legend(fontsize=8)
    rank_fig_path = fig_dir / "pinn_rectangle_2d_ranked_spectra.png"
    fig.savefig(rank_fig_path, dpi=240, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    for p in [split_path, ranked_path, overlap_path, split_fig_path, delta_fig_path, rank_fig_path]:
        print(f"  {p.relative_to(root)}")

    print("\nSplitting summary")
    print("  eta      E12_exact      E12_PINN    err12%      E21_exact      E21_PINN    err21%    Delta_PINN")
    for row in rows:
        print(
            f"  {row['eta']:5.2f}  "
            f"{row['E12_exact']:13.6f}  {row['E12_PINN_matched']:12.6f}  {row['relative_error_E12_percent']:7.3f}  "
            f"{row['E21_exact']:13.6f}  {row['E21_PINN_matched']:12.6f}  {row['relative_error_E21_percent']:7.3f}  "
            f"{row['splitting_PINN_E12_minus_E21']:11.6f}"
        )

    print("\nInterpretation note:")
    print("  eta=1 is the square geometry, where psi_12 and psi_21 are degenerate.")
    print("  For eta != 1, the rectangle breaks the x/y symmetry and the two branches split.")
    print("  Exact modes are used only for validation/matching, not as training targets.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="PINN/Rayleigh-Ritz degeneracy lifting for the 2D rectangular infinite well.")
    p.add_argument("--etas", type=str, default="0.80,0.90,1.00,1.10,1.20", help="Comma-separated aspect ratios eta=Ly/Lx.")
    p.add_argument("--states", type=int, default=4, help="Subspace dimension / number of low-energy states.")
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--n-quad", type=int, default=28)
    p.add_argument("--n-validate", type=int, default=48)
    p.add_argument("--hidden-width", type=int, default=32)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--fourier-features", type=int, default=3)
    p.add_argument("--output-scale", type=float, default=4.0)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--lbfgs-lr", type=float, default=0.6)
    p.add_argument("--grad-clip", type=float, default=100.0)
    p.add_argument("--cholesky-eps", type=float, default=1e-8)
    p.add_argument("--gram-penalty", type=float, default=1e-4)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--max-exact-n", type=int, default=6)
    p.add_argument("--max-exact-modes", type=int, default=12)
    p.add_argument("--quiet", action="store_true", help="Reduce per-epoch/restart logging.")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    root = find_project_root()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    etas = parse_etas(args.etas)

    set_seed(args.seed)
    print("2D rectangular-well degeneracy-lifting subspace PINN")
    print(f"  root: {root}")
    print(f"  device: {device}, dtype={args.dtype}, threads={args.threads}")
    print(f"  Lx={args.lx}; etas={etas}; states={args.states}")
    print("  exact modes are validation/matching references only, not training targets")

    results: list[EtaResult] = []
    for eta in etas:
        print("\n" + "=" * 78)
        print(f"Training rectangle eta={eta:.4f}  (Lx={args.lx:.4f}, Ly={args.lx*eta:.4f})")
        model = train_one_eta(args, device, dtype, eta)
        result = evaluate_one_eta(model, args, device, dtype, eta)
        refs_preview = ranked_exact_states(args.states, args.max_exact_n, args.lx, args.lx * eta)
        print("  Ranked exact vs PINN validation:")
        for i, ref in enumerate(refs_preview):
            e = result.eigvals[i]
            rel = abs(e - ref["epsilon"]) / ref["epsilon"] * 100.0
            print(f"    {i+1:2d} {ref['label']:>7s}: exact={ref['epsilon']:10.5f}, PINN={e:10.5f}, err={rel:7.3f}%")
        e12_p, idx12, ov12, w12 = matched_energy(result, 1, 2)
        e21_p, idx21, ov21, w21 = matched_energy(result, 2, 1)
        print(f"  Matched branches: psi_12 -> PINN {idx12} E={e12_p:.6f} |ov|^2={w12:.3f}; psi_21 -> PINN {idx21} E={e21_p:.6f} |ov|^2={w21:.3f}")
        results.append(result)

    save_outputs(results, args, root, device, dtype)


if __name__ == "__main__":
    main()
