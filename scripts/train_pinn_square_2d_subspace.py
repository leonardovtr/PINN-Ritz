#!/usr/bin/env python3
"""Subspace Rayleigh-Ritz PINN for the 2D infinite square well.

This is the recommended 2D square-well version after the robust 1D study.
Instead of training states one-by-one, it trains K boundary-satisfying neural
functions simultaneously and orthonormalizes them by a differentiable
Rayleigh-Ritz/Cholesky step on the quadrature grid.

Why this version?
-----------------
Sequential deflation is fragile in degenerate spectra. For a square well,
psi_12 and psi_21 are degenerate, so a subspace method is more natural: it
learns the low-energy subspace, then diagonalizes the Hamiltonian inside that
subspace. This also makes arbitrary linear combinations in the degenerate
subspace physically interpretable.

Training uses only:
  - boundary condition through a bounded boundary-satisfying ansatz;
  - Rayleigh-Ritz variational energy;
  - orthonormality through quadrature/Cholesky.
Exact modes are used only after training for validation and interpretation.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "quantum_dots").exists() or (parent / "requirements.txt").exists():
            return parent
    return here.parents[1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def gauss_legendre_square(n: int, lx: float, ly: float, device, dtype):
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
            states.append({"nx": nx, "ny": ny, "epsilon": exact_energy(nx, ny, lx, ly), "label": f"psi_{nx}{ny}"})
    states.sort(key=lambda d: (d["epsilon"], d["nx"], d["ny"]))
    return states[:k]


class MultiOutputBoundedMLP(nn.Module):
    def __init__(self, k: int, hidden_width: int, hidden_layers: int, fourier_features: int, output_scale: float, lx: float, ly: float):
        super().__init__()
        self.k = int(k)
        self.fourier_features = int(fourier_features)
        self.output_scale = float(output_scale)
        self.lx = float(lx)
        self.ly = float(ly)
        in_dim = 2 + 4 * self.fourier_features
        layers = []
        last = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last, hidden_width))
            layers.append(nn.Tanh())
            last = hidden_width
        layers.append(nn.Linear(last, self.k))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        final = [m for m in self.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.uniform_(final.bias, -0.2, 0.2)

    def features(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1] / self.lx
        y = xy[:, 1:2] / self.ly
        feats = [x, y]
        for kk in range(1, self.fourier_features + 1):
            k = float(kk)
            feats.extend([
                torch.sin(2 * math.pi * k * x), torch.cos(2 * math.pi * k * x),
                torch.sin(2 * math.pi * k * y), torch.cos(2 * math.pi * k * y),
            ])
        return torch.cat(feats, dim=1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.output_scale * torch.tanh(self.net(self.features(xy)))


def envelope(xy: torch.Tensor, lx: float, ly: float) -> torch.Tensor:
    return torch.sin(math.pi * xy[:, 0] / lx) * torch.sin(math.pi * xy[:, 1] / ly)


def raw_fields_and_grads(model: nn.Module, xy_base: torch.Tensor, lx: float, ly: float):
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
    K = F.shape[1]
    Wf = w[:, None] * F
    S = F.T @ Wf
    S = 0.5 * (S + S.T) + eps * torch.eye(K, device=F.device, dtype=F.dtype)
    L = torch.linalg.cholesky(S)
    I = torch.eye(K, device=F.device, dtype=F.dtype)
    # A = inv(L.T), so (F A)^T W (F A) = I
    A = torch.linalg.solve_triangular(L.T, I, upper=True)
    Q = F @ A
    Qx = Fx @ A
    Qy = Fy @ A
    H = Qx.T @ (w[:, None] * Qx) + Qy.T @ (w[:, None] * Qy)
    H = 0.5 * (H + H.T)
    return Q, Qx, Qy, H, S


def subspace_loss(model, xy, w, args):
    F, Fx, Fy = raw_fields_and_grads(model, xy, args.lx, args.ly)
    Q, Qx, Qy, H, S = orthonormalize(F, Fx, Fy, w, args.cholesky_eps)
    eigvals = torch.linalg.eigvalsh(H)
    loss = torch.sum(eigvals)
    # Weakly penalize nearly dependent raw outputs.
    sign, logabsdet = torch.linalg.slogdet(S)
    loss = loss + args.gram_penalty * torch.relu(-logabsdet)
    return loss, eigvals, H


@torch.no_grad()
def validation_eigs(model, xy, w, args):
    # Need gradients; no_grad cannot be used for raw gradients. This function is overwritten below.
    raise RuntimeError("Use validation_eigs_grad")


def validation_eigs_grad(model, xy, w, args):
    F, Fx, Fy = raw_fields_and_grads(model, xy, args.lx, args.ly)
    Q, Qx, Qy, H, _ = orthonormalize(F, Fx, Fy, w, args.cholesky_eps)
    eigvals, eigvecs = torch.linalg.eigh(H)
    modes = Q @ eigvecs
    return eigvals, modes, H, eigvecs


def train(args, device, dtype):
    xy, w = gauss_legendre_square(args.n_quad, args.lx, args.ly, device, dtype)
    xyv, wv = gauss_legendre_square(args.n_validate, args.lx, args.ly, device, dtype)
    best_global = None
    best_score = float("inf")

    for restart in range(1, args.restarts + 1):
        set_seed(args.seed + restart)
        model = MultiOutputBoundedMLP(args.states, args.hidden_width, args.hidden_layers, args.fourier_features, args.output_scale, args.lx, args.ly).to(device=device, dtype=dtype)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_state = copy.deepcopy(model.state_dict())
        best_val_sum = float("inf")
        best_vals = None
        print(f"\nRestart {restart}/{args.restarts}")
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
            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                train_vals = eigs.detach().cpu().numpy()
                val_vals = vals.detach().cpu().numpy()
                print(f"  epoch {epoch:5d}/{args.epochs}: train={np.array2string(train_vals, precision=3)}, val={np.array2string(val_vals, precision=3)}")

        model.load_state_dict(best_state)
        if args.lbfgs_steps > 0:
            before_state = copy.deepcopy(model.state_dict())
            before_vals, _, _, _ = validation_eigs_grad(model, xyv, wv, args)
            before_sum = float(torch.sum(before_vals).detach().cpu())
            lbfgs = torch.optim.LBFGS(model.parameters(), lr=args.lbfgs_lr, max_iter=args.lbfgs_steps, line_search_fn="strong_wolfe", history_size=80)
            def closure():
                lbfgs.zero_grad(set_to_none=True)
                loss, _, _ = subspace_loss(model, xy, w, args)
                loss.backward()
                return loss
            try:
                lbfgs.step(closure)
            except RuntimeError as exc:
                print(f"  LBFGS warning: {exc}")
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
        print(f"  restart result: val={np.array2string(chosen_vals, precision=6)}, sum={chosen_sum:.6f}, source={source}")
        if chosen_sum < best_score:
            best_score = chosen_sum
            best_global = copy.deepcopy(model.state_dict())

    model = MultiOutputBoundedMLP(args.states, args.hidden_width, args.hidden_layers, args.fourier_features, args.output_scale, args.lx, args.ly).to(device=device, dtype=dtype)
    model.load_state_dict(best_global)
    return model


def save_outputs(model, args, root, device, dtype):
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    refs = ranked_exact_states(args.states, max(args.max_exact_n, 5), args.lx, args.ly)
    xyv, wv = gauss_legendre_square(args.n_validate, args.lx, args.ly, device, dtype)
    eigvals, modes, _, _ = validation_eigs_grad(model, xyv, wv, args)
    eig_np = eigvals.detach().cpu().numpy()
    overlap = np.zeros((args.states, len(refs)))
    for i in range(args.states):
        for j, ref in enumerate(refs):
            ex = exact_mode_torch(xyv, ref["nx"], ref["ny"], args.lx, args.ly)
            overlap[i, j] = float(torch.sum(wv * modes[:, i] * ex).detach().cpu())
    deg_cols = [j for j, ref in enumerate(refs) if {ref["nx"], ref["ny"]} == {1, 2}]
    deg_weight = np.sum(overlap[:, deg_cols] ** 2, axis=1) if deg_cols else np.zeros(args.states)
    rows = []
    for i, ref in enumerate(refs):
        rel = abs(eig_np[i] - ref["epsilon"]) / ref["epsilon"] * 100.0
        rows.append({
            "state_index": i + 1,
            "exact_label_ranked": ref["label"],
            "nx_ref": ref["nx"],
            "ny_ref": ref["ny"],
            "epsilon_exact_ranked": ref["epsilon"],
            "epsilon_pinn_validation": eig_np[i],
            "relative_error_percent_vs_ranked": rel,
            "degenerate_weight_12_21": deg_weight[i],
        })
    summary = pd.DataFrame(rows)
    summary_path = data_dir / "pinn_square_2d_subspace_summary.csv"
    summary.to_csv(summary_path, index=False)
    overlap_df = pd.DataFrame(overlap, columns=[r["label"] for r in refs])
    overlap_df.insert(0, "pinn_state", [f"PINN_{i+1}" for i in range(args.states)])
    overlap_path = data_dir / "pinn_square_2d_subspace_overlap_matrix.csv"
    overlap_df.to_csv(overlap_path, index=False)

    # Plot modes on uniform grid, diagonalizing the subspace on a Gauss grid, then evaluating raw subspace on the uniform grid with same Ritz vectors from validation.
    # For visual consistency, recompute modes directly on uniform grid and diagonalize with uniform weights.
    n = args.n_plot
    x = np.linspace(0, args.lx, n)
    y = np.linspace(0, args.ly, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    xy_np = np.column_stack([X.ravel(), Y.ravel()])
    xy_t = torch.tensor(xy_np, device=device, dtype=dtype)
    w_np = np.ones(n * n) * (args.lx / (n - 1)) * (args.ly / (n - 1))
    edge = (np.isclose(xy_np[:, 0], 0) | np.isclose(xy_np[:, 0], args.lx)).astype(float) + (np.isclose(xy_np[:, 1], 0) | np.isclose(xy_np[:, 1], args.ly)).astype(float)
    w_np[edge == 1] *= 0.5
    w_np[edge == 2] *= 0.25
    w_t = torch.tensor(w_np, device=device, dtype=dtype)
    _, plot_modes, _, _ = validation_eigs_grad(model, xy_t, w_t, args)
    grid_df = pd.DataFrame({"x": xy_np[:,0], "y": xy_np[:,1]})
    for i in range(args.states):
        arr = plot_modes[:, i].detach().cpu().numpy()
        if overlap[i, np.argmax(np.abs(overlap[i]))] < 0:
            arr = -arr
        grid_df[f"pinn_state_{i+1}"] = arr
    grid_path = data_dir / "pinn_square_2d_subspace_modes_grid.csv"
    grid_df.to_csv(grid_path, index=False)

    fig, axes = plt.subplots(1, args.states, figsize=(3 * args.states, 2.8), constrained_layout=True)
    if args.states == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        Z = grid_df[f"pinn_state_{i+1}"].to_numpy().reshape(n, n)
        vmax = np.max(np.abs(Z)) or 1.0
        im = ax.imshow(Z.T, origin="lower", extent=[0,args.lx,0,args.ly], aspect="equal", vmin=-vmax, vmax=vmax)
        ax.set_title(f"PINN {i+1}\n$\\epsilon$={eig_np[i]:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.72)
    modes_path = fig_dir / "pinn_square_2d_subspace_modes.png"
    fig.savefig(modes_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    xs = np.arange(1, args.states+1)
    fig, ax = plt.subplots(figsize=(6.4,3.7))
    ax.plot(xs, [r["epsilon"] for r in refs], "o-", label="analytical reference")
    ax.plot(xs, eig_np, "s--", label="PINN/Ritz")
    ax.set_xlabel("ranked state"); ax.set_ylabel(r"dimensionless energy $\epsilon$")
    ax.set_title("2D square well: subspace PINN/Rayleigh-Ritz")
    ax.legend()
    energy_path = fig_dir / "pinn_square_2d_subspace_energy_comparison.png"
    fig.savefig(energy_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6,4.2))
    c12 = c21 = None
    for j, ref in enumerate(refs):
        if ref["nx"] == 1 and ref["ny"] == 2: c12 = overlap[:, j]
        if ref["nx"] == 2 and ref["ny"] == 1: c21 = overlap[:, j]
    if c12 is not None and c21 is not None:
        ax.axhline(0, lw=0.8); ax.axvline(0, lw=0.8)
        ax.scatter(c12, c21, s=60)
        for i in range(args.states): ax.text(c12[i]+0.02, c21[i]+0.02, f"{i+1}")
        ax.set_xlabel(r"$\langle \psi_{PINN}|\psi_{1,2}\rangle$")
        ax.set_ylabel(r"$\langle \psi_{PINN}|\psi_{2,1}\rangle$")
        ax.set_xlim(-1.1,1.1); ax.set_ylim(-1.1,1.1); ax.set_aspect("equal")
        ax.set_title("Projection onto degenerate subspace")
    deg_path = fig_dir / "pinn_square_2d_subspace_degenerate_subspace.png"
    fig.savefig(deg_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.1*len(refs)+2.5, 0.75*args.states+2.0))
    im = ax.imshow(overlap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(refs))); ax.set_xticklabels([r["label"] for r in refs], rotation=45, ha="right")
    ax.set_yticks(np.arange(args.states)); ax.set_yticklabels([f"PINN {i+1}" for i in range(args.states)])
    ax.set_title("Overlap matrix with analytical modes")
    fig.colorbar(im, ax=ax, shrink=0.85)
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            ax.text(j, i, f"{overlap[i,j]:.2f}", ha="center", va="center", fontsize=8)
    overlap_fig_path = fig_dir / "pinn_square_2d_subspace_overlap_matrix.png"
    fig.savefig(overlap_fig_path, dpi=220, bbox_inches="tight"); plt.close(fig)

    print("\nSaved outputs:")
    for p in [summary_path, overlap_path, grid_path, modes_path, energy_path, deg_path, overlap_fig_path]:
        print(f"  {p.relative_to(root)}")
    print("\nSummary")
    print("  i  exact label      epsilon_exact      epsilon_PINN       rel.err      deg.weight")
    for row in rows:
        print(f"  {row['state_index']:1d}  {row['exact_label_ranked']:>10s}  {row['epsilon_exact_ranked']:16.8f}  {row['epsilon_pinn_validation']:16.8f}  {row['relative_error_percent_vs_ranked']:9.4f}%  {row['degenerate_weight_12_21']:10.4f}")


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser(description="Subspace Rayleigh-Ritz PINN for the 2D square infinite well.")
    p.add_argument("--states", type=int, default=4)
    p.add_argument("--lx", type=float, default=1.0)
    p.add_argument("--ly", type=float, default=1.0)
    p.add_argument("--n-quad", type=int, default=32)
    p.add_argument("--n-validate", type=int, default=56)
    p.add_argument("--n-plot", type=int, default=90)
    p.add_argument("--hidden-width", type=int, default=48)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--fourier-features", type=int, default=3)
    p.add_argument("--output-scale", type=float, default=4.0)
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1200)
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
    p.add_argument("--max-exact-n", type=int, default=5)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None):
    args = parse_args(argv)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    root = find_project_root()
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    set_seed(args.seed)
    print("2D square-well subspace Rayleigh-Ritz PINN")
    print(f"  root: {root}")
    print(f"  device: {device}, dtype={args.dtype}, threads={args.threads}")
    print(f"  states: {args.states}, domain: Lx={args.lx}, Ly={args.ly}")
    print("  exact modes are validation references only, not training targets")
    model = train(args, device, dtype)
    save_outputs(model, args, root, device, dtype)


if __name__ == "__main__":
    main()
