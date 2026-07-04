#!/usr/bin/env python3
"""Projected/Fourier-feature 1D PINN for the infinite quantum well.

Physical problem
----------------
Dimensionless stationary Schrödinger equation in a 1D infinite well:

    - d²ψ/dx² = ε ψ,        0 < x < L
    ψ(0) = ψ(L) = 0,
    ∫₀ᴸ |ψ(x)|² dx = 1.

For L=1, the analytical reference is

    ε_n = n²π²,
    ψ_n(x) = √2 sin(nπx).

Why this version exists
-----------------------
The first implementation used a soft orthogonality penalty to obtain excited
states. That is fine for n=1 and usually n=2, but for n>=3 the optimizer may
find a function that is nearly orthogonal to the lower states without being the
next eigenfunction. This file uses a hard Gram-Schmidt/Rayleigh-Ritz projection:

    ψ̃_n = ψθ - Σ_{j<n} <ψθ, ψ_j> ψ_j,
    ψ_n = ψ̃_n / sqrt(<ψ̃_n, ψ̃_n>).

The previous states are represented on the current training grid together with
their first and second derivatives. This keeps the projected residual

    -ψ_n'' - ε_n ψ_n

physically consistent without backpropagating through old frozen networks.

Outputs
-------
- data/pinn_1d_summary.csv
- data/pinn_1d_wavefunctions.csv
- figures/pinn_1d_wavefunctions.png
- figures/pinn_1d_energy_comparison.png

Run from the project root, for example:

    python scripts/train_pinn_1d_projected.py --states 3

or, if you are currently inside scripts/:

    python train_pinn_1d_projected.py --states 3
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required. Install it with:\n\n"
        "    python -m pip install torch\n\n"
        "Then run again from the project root:\n\n"
        "    python scripts/train_pinn_1d_projected.py --states 3\n"
    ) from exc

from quantum_dots.analytical import (
    infinite_well_1d_energy,
    infinite_well_1d_wavefunction,
)

FIGURES = ROOT / "figures"
DATA = ROOT / "data"


class WellNet(nn.Module):
    """Neural trial function with exact infinite-wall boundary conditions.

    The optional Fourier features help the MLP represent excited states with
    several internal nodes. This keeps the model physics-informed while
    reducing the spectral-bias problem of plain tanh networks.
    """

    def __init__(
        self,
        length: float,
        hidden_width: int,
        hidden_layers: int,
        fourier_features: int = 0,
    ) -> None:
        super().__init__()
        self.length = float(length)
        self.fourier_features = int(fourier_features)

        layers: list[nn.Module] = []
        in_features = 1 + 2 * self.fourier_features
        for _ in range(hidden_layers):
            layer = nn.Linear(in_features, hidden_width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.Tanh())
            in_features = hidden_width

        output = nn.Linear(in_features, 1)
        nn.init.xavier_normal_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # Map [0, L] -> [-1, 1] for better conditioning.
        z = 2.0 * x / self.length - 1.0
        if self.fourier_features <= 0:
            return z

        terms = [z]
        xi = x / self.length
        for k in range(1, self.fourier_features + 1):
            angle = k * torch.pi * xi
            terms.append(torch.sin(angle))
            terms.append(torch.cos(angle))
        return torch.cat(terms, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hard Dirichlet boundary condition: ψ(0)=ψ(L)=0.
        return x * (self.length - x) * self.net(self.features(x))


@dataclass
class BasisOnGrid:
    """Physical state already projected, normalized and detached on a grid."""

    psi: torch.Tensor
    psi_x: torch.Tensor
    psi_xx: torch.Tensor


@dataclass
class StateResult:
    n: int
    model: WellNet
    epsilon_pinn: float
    epsilon_exact: float
    relative_error: float
    norm: float
    residual_mse: float
    max_overlap: float
    node_count: int
    basis: BasisOnGrid


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_grid(n_points: int, length: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x = torch.linspace(0.0, length, n_points, device=device, dtype=dtype)
    return x.reshape(-1, 1).requires_grad_(True)


def integral(values: torch.Tensor, length: float) -> torch.Tensor:
    """Trapezoidal integral over the uniform 1D grid."""
    if values.shape[0] < 2:
        return length * torch.mean(values)
    dx = length / (values.shape[0] - 1)
    weights = torch.ones_like(values)
    weights[0] = 0.5
    weights[-1] = 0.5
    return dx * torch.sum(weights * values)


def derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def raw_quantities(model: WellNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw trial function and derivatives before projection/normalization."""
    psi = model(x)
    psi_x = derivative(psi, x)
    psi_xx = derivative(psi_x, x)
    return psi, psi_x, psi_xx


def project_and_normalize(
    psi: torch.Tensor,
    psi_x: torch.Tensor,
    psi_xx: torch.Tensor,
    previous_basis: list[BasisOnGrid],
    length: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hard-project a trial state away from lower states and normalize it.

    The same projection coefficients are applied to ψ, ψ' and ψ'', which is the
    discrete-grid version of differentiating the projected function. The
    coefficients are global scalars: they may affect parameter gradients, but
    they are not treated as local functions of x.
    """
    max_overlap_before = torch.zeros((), device=psi.device, dtype=psi.dtype)

    for prev in previous_basis:
        coeff = integral(psi * prev.psi, length)
        max_overlap_before = torch.maximum(max_overlap_before, torch.abs(coeff.detach()))
        psi = psi - coeff * prev.psi
        psi_x = psi_x - coeff * prev.psi_x
        psi_xx = psi_xx - coeff * prev.psi_xx

    norm_raw = integral(psi.pow(2), length)
    # Treat normalization as a global scalar for spatial derivatives. This is
    # consistent with the manually projected ψ', ψ'' above.
    scale = torch.sqrt(norm_raw.detach() + 1.0e-14)
    psi = psi / scale
    psi_x = psi_x / scale
    psi_xx = psi_xx / scale

    max_overlap_after = torch.zeros((), device=psi.device, dtype=psi.dtype)
    for prev in previous_basis:
        overlap = integral(psi * prev.psi, length)
        max_overlap_after = torch.maximum(max_overlap_after, torch.abs(overlap.detach()))

    return psi, psi_x, psi_xx, max_overlap_before, max_overlap_after


def normalized_quantities(
    model: WellNet,
    x: torch.Tensor,
    previous_basis: list[BasisOnGrid],
) -> dict[str, torch.Tensor]:
    length = model.length
    raw_psi, raw_psi_x, raw_psi_xx = raw_quantities(model, x)
    psi, psi_x, psi_xx, max_before, max_after = project_and_normalize(
        raw_psi,
        raw_psi_x,
        raw_psi_xx,
        previous_basis,
        length,
    )

    norm = integral(psi.pow(2), length)
    epsilon = integral(psi_x.pow(2), length) / (norm + 1.0e-14)
    residual = -psi_xx - epsilon * psi
    residual_mse = integral(residual[1:-1].pow(2), length)

    return {
        "psi": psi,
        "psi_x": psi_x,
        "psi_xx": psi_xx,
        "norm": norm,
        "epsilon": epsilon,
        "residual_mse": residual_mse,
        "max_overlap_before": max_before,
        "max_overlap_after": max_after,
    }


def loss_function(
    model: WellNet,
    x: torch.Tensor,
    previous_basis: list[BasisOnGrid],
    weight_pde: float,
    weight_norm: float,
    weight_orthogonality: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    q = normalized_quantities(model, x, previous_basis)
    epsilon = q["epsilon"]
    norm_loss = (q["norm"] - 1.0).pow(2)

    # Orthogonality should already be enforced by projection. This term is a
    # small diagnostic/safety penalty, not the main mechanism.
    orthogonality_loss = q["max_overlap_after"].pow(2)

    total = (
        epsilon
        + weight_pde * q["residual_mse"]
        + weight_norm * norm_loss
        + weight_orthogonality * orthogonality_loss
    )

    diagnostics = {
        "loss": total.detach(),
        "epsilon": epsilon.detach(),
        "norm": q["norm"].detach(),
        "residual_mse": q["residual_mse"].detach(),
        "max_overlap": q["max_overlap_after"].detach(),
        "max_overlap_before_projection": q["max_overlap_before"].detach(),
    }
    return total, diagnostics


def count_internal_nodes(x: np.ndarray, psi: np.ndarray, trim: float = 1.0e-3) -> int:
    """Count sign changes away from the two hard-wall boundaries."""
    mask = (x > x.min() + trim) & (x < x.max() - trim)
    y = psi[mask]
    if y.size < 3:
        return 0
    # Avoid artificial nodes from tiny values near zero.
    threshold = 1.0e-3 * max(1.0, np.nanmax(np.abs(y)))
    y = y.copy()
    y[np.abs(y) < threshold] = 0.0
    signs = np.sign(y)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def train_state(
    n: int,
    previous: list[StateResult],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> StateResult:
    model = WellNet(
        args.length,
        args.hidden_width,
        args.hidden_layers,
        fourier_features=args.fourier_features,
    ).to(device=device, dtype=dtype)
    previous_basis = [state.basis for state in previous]

    x = make_grid(args.n_collocation, args.length, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epsilon_exact = float(infinite_well_1d_energy(n, length=args.length))
    print(f"\nTraining n={n}")
    print(f"  analytical epsilon = {epsilon_exact:.10f}")

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, diag = loss_function(
            model,
            x,
            previous_basis,
            weight_pde=args.weight_pde,
            weight_norm=args.weight_norm,
            weight_orthogonality=args.weight_orthogonality,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"  epoch {epoch:6d}/{args.epochs}: "
                f"loss={diag['loss'].item():.4e}, "
                f"epsilon={diag['epsilon'].item():.8f}, "
                f"norm={diag['norm'].item():.6f}, "
                f"res={diag['residual_mse'].item():.2e}, "
                f"max|overlap|={diag['max_overlap'].item():.2e}"
            )

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            history_size=50,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = loss_function(
                model,
                x,
                previous_basis,
                weight_pde=args.weight_pde,
                weight_norm=args.weight_norm,
                weight_orthogonality=args.weight_orthogonality,
            )
            loss.backward()
            return loss

        lbfgs.step(closure)
        _, diag = loss_function(
            model,
            x,
            previous_basis,
            weight_pde=args.weight_pde,
            weight_norm=args.weight_norm,
            weight_orthogonality=args.weight_orthogonality,
        )
        print(
            f"  LBFGS done: epsilon={diag['epsilon'].item():.8f}, "
            f"norm={diag['norm'].item():.6f}, "
            f"res={diag['residual_mse'].item():.2e}, "
            f"max|overlap|={diag['max_overlap'].item():.2e}"
        )

    # Final physical state on the training grid. Store detached projected ψ, ψ', ψ''
    # so that higher states project against the actual lower states, not against
    # the raw neural-network outputs.
    q = normalized_quantities(model, x, previous_basis)
    epsilon_pinn = float(q["epsilon"].detach().cpu())
    rel_error = abs(epsilon_pinn - epsilon_exact) / epsilon_exact
    basis = BasisOnGrid(
        psi=q["psi"].detach(),
        psi_x=q["psi_x"].detach(),
        psi_xx=q["psi_xx"].detach(),
    )

    x_np = x.detach().cpu().numpy().reshape(-1)
    psi_np = basis.psi.detach().cpu().numpy().reshape(-1)
    psi_exact = infinite_well_1d_wavefunction(x_np, n, length=args.length)
    if np.trapz(psi_np * psi_exact, x_np) < 0:
        psi_np = -psi_np
    node_count = count_internal_nodes(x_np, psi_np, trim=args.length / args.n_collocation)

    return StateResult(
        n=n,
        model=model,
        epsilon_pinn=epsilon_pinn,
        epsilon_exact=epsilon_exact,
        relative_error=rel_error,
        norm=float(q["norm"].detach().cpu()),
        residual_mse=float(q["residual_mse"].detach().cpu()),
        max_overlap=float(q["max_overlap_after"].detach().cpu()),
        node_count=node_count,
        basis=basis,
    )


def projected_psi_for_plot(
    model: WellNet,
    x: torch.Tensor,
    previous_psis: list[torch.Tensor],
    length: float,
) -> torch.Tensor:
    """Projected normalized ψ for plotting/CSV only; derivatives are not needed."""
    with torch.no_grad():
        psi = model(x)
        for prev in previous_psis:
            coeff = integral(psi * prev, length)
            psi = psi - coeff * prev
        norm = integral(psi.pow(2), length)
        psi = psi / torch.sqrt(norm + 1.0e-14)
    return psi


def evaluate_curves(
    states: list[StateResult],
    length: float,
    device: torch.device,
    dtype: torch.dtype,
    n_grid: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[int, int]]:
    x_np = np.linspace(0.0, length, n_grid)
    x_torch = torch.tensor(x_np.reshape(-1, 1), device=device, dtype=dtype)

    pinn: dict[int, np.ndarray] = {}
    exact: dict[int, np.ndarray] = {}
    nodes: dict[int, int] = {}
    previous_psis: list[torch.Tensor] = []

    for state in states:
        state.model.eval()
        psi_t = projected_psi_for_plot(state.model, x_torch, previous_psis, length)
        previous_psis.append(psi_t.detach())

        psi = psi_t.detach().cpu().numpy().reshape(-1)
        norm = np.sqrt(np.trapz(psi**2, x_np))
        if norm > 0:
            psi = psi / norm

        psi_exact = infinite_well_1d_wavefunction(x_np, state.n, length=length)
        if np.trapz(psi * psi_exact, x_np) < 0:
            psi = -psi

        pinn[state.n] = psi
        exact[state.n] = psi_exact
        nodes[state.n] = count_internal_nodes(x_np, psi, trim=length / n_grid)

    return x_np, pinn, exact, nodes


def save_outputs(
    states: list[StateResult],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    x, pinn, exact, nodes = evaluate_curves(states, args.length, device, dtype, args.n_plot_grid)

    summary_path = DATA / "pinn_1d_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n",
                "epsilon_exact",
                "epsilon_pinn",
                "relative_error",
                "norm_integral",
                "residual_mse",
                "max_abs_overlap_with_lower_states",
                "expected_internal_nodes",
                "pinn_internal_nodes",
            ],
        )
        writer.writeheader()
        for s in states:
            writer.writerow(
                {
                    "n": s.n,
                    "epsilon_exact": f"{s.epsilon_exact:.12f}",
                    "epsilon_pinn": f"{s.epsilon_pinn:.12f}",
                    "relative_error": f"{s.relative_error:.12e}",
                    "norm_integral": f"{s.norm:.12f}",
                    "residual_mse": f"{s.residual_mse:.12e}",
                    "max_abs_overlap_with_lower_states": f"{s.max_overlap:.12e}",
                    "expected_internal_nodes": s.n - 1,
                    "pinn_internal_nodes": nodes[s.n],
                }
            )

    wave_path = DATA / "pinn_1d_wavefunctions.csv"
    with wave_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["x"]
        for s in states:
            fields += [f"psi_pinn_n{s.n}", f"psi_exact_n{s.n}"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, xi in enumerate(x):
            row: dict[str, object] = {"x": f"{xi:.12f}"}
            for s in states:
                row[f"psi_pinn_n{s.n}"] = f"{pinn[s.n][i]:.12e}"
                row[f"psi_exact_n{s.n}"] = f"{exact[s.n][i]:.12e}"
            writer.writerow(row)

    fig, axes = plt.subplots(len(states), 1, figsize=(7.0, 2.35 * len(states)), sharex=True)
    if len(states) == 1:
        axes = [axes]
    for ax, s in zip(axes, states):
        ax.plot(x, exact[s.n], lw=2.0, label="analytical")
        ax.plot(x, pinn[s.n], "--", lw=1.8, label="projected PINN")
        ax.set_ylabel(rf"$\psi_{s.n}(x)$")
        ax.set_title(
            rf"$n={s.n}$: "
            rf"$\epsilon_{{PINN}}={s.epsilon_pinn:.6f}$, "
            rf"error={100*s.relative_error:.3f}\%$, "
            rf"nodes={nodes[s.n]}/{s.n-1}"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel(r"$x/L$")
    fig.tight_layout()
    fig.savefig(FIGURES / "pinn_1d_wavefunctions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    n_values = np.array([s.n for s in states])
    exact_scaled = np.array([s.epsilon_exact / np.pi**2 for s in states])
    pinn_scaled = np.array([s.epsilon_pinn / np.pi**2 for s in states])
    ax.plot(n_values, exact_scaled, "o-", lw=2, label=r"analytical $n^2$")
    ax.plot(n_values, pinn_scaled, "s", ms=6, label="projected PINN")
    ax.set_xlabel("state index n")
    ax.set_ylabel(r"$\epsilon/\pi^2$")
    ax.set_title("1D infinite well: spectrum recovered by projected PINN")
    ax.set_xticks(n_values)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "pinn_1d_energy_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved outputs:")
    print(f"  {summary_path.relative_to(ROOT)}")
    print(f"  {wave_path.relative_to(ROOT)}")
    print(f"  {(FIGURES / 'pinn_1d_wavefunctions.png').relative_to(ROOT)}")
    print(f"  {(FIGURES / 'pinn_1d_energy_comparison.png').relative_to(ROOT)}")



def selection_score(result: StateResult, args: argparse.Namespace) -> float:
    """Score used only to choose among random restarts.

    In a 1D Sturm-Liouville problem, the nth state must have n-1 internal
    nodes. Among candidates with the correct nodal count, the variational
    principle favors the one with the lowest Rayleigh quotient. A small
    residual contribution avoids selecting a low-energy but poorly converged
    function.
    """
    expected_nodes = result.n - 1
    node_mismatch = abs(result.node_count - expected_nodes)
    return (
        result.epsilon_pinn
        + args.selection_residual_weight * result.residual_mse
        + args.selection_node_penalty * node_mismatch
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Projected 1D infinite-well PINN")
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--states", type=int, default=1)
    parser.add_argument("--hidden-width", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--fourier-features", type=int, default=8)
    parser.add_argument("--n-collocation", type=int, default=160)
    parser.add_argument("--n-plot-grid", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lbfgs-steps", type=int, default=250)
    parser.add_argument("--lbfgs-lr", type=float, default=0.8)
    parser.add_argument("--weight-pde", type=float, default=2.0e-4)
    parser.add_argument("--weight-norm", type=float, default=1.0)
    parser.add_argument("--weight-orthogonality", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--selection-residual-weight", type=float, default=1.0e-5)
    parser.add_argument("--selection-node-penalty", type=float, default=1.0e4)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA device is available.")

    print("Projected 1D infinite-well PINN")
    print(f"  root: {ROOT}")
    print(f"  device: {device}")
    print(f"  dtype: {args.dtype}")
    print(f"  states: 1..{args.states}")
    print("  method: hard Gram-Schmidt projection + Rayleigh quotient + PDE residual")
    print(f"  Fourier features: {args.fourier_features}")
    print(f"  restarts/state: {max(1, args.restarts)}")

    states: list[StateResult] = []
    for n in range(1, args.states + 1):
        best_result: StateResult | None = None
        best_score = float("inf")
        n_restarts = max(1, args.restarts)

        for restart in range(n_restarts):
            restart_seed = args.seed + 10000 * n + restart
            set_seed(restart_seed)
            if n_restarts > 1:
                print(f"\nRestart {restart + 1}/{n_restarts} for n={n} (seed={restart_seed})")

            candidate = train_state(n, states, args, device, dtype)
            score = selection_score(candidate, args)
            print(
                f"  candidate n={n}, restart={restart + 1}: "
                f"score={score:.6e}, epsilon={candidate.epsilon_pinn:.10f}, "
                f"error={100*candidate.relative_error:.4f}%, "
                f"nodes={candidate.node_count}/{n-1}, "
                f"res={candidate.residual_mse:.2e}"
            )
            if score < best_score:
                best_score = score
                best_result = candidate

        assert best_result is not None
        states.append(best_result)
        print(
            f"  selected n={n}: epsilon_PINN={best_result.epsilon_pinn:.10f}, "
            f"epsilon_exact={best_result.epsilon_exact:.10f}, "
            f"relative error={100*best_result.relative_error:.4f}%, "
            f"nodes={best_result.node_count}/{n-1}, "
            f"selection_score={best_score:.6e}"
        )

    save_outputs(states, args, device, dtype)

    print("\nSummary")
    print("  n    epsilon_exact      epsilon_PINN       relative_error     nodes")
    for s in states:
        print(
            f"  {s.n:<2d}   {s.epsilon_exact:14.8f}   "
            f"{s.epsilon_pinn:14.8f}   {100*s.relative_error:10.4f}%      "
            f"{s.node_count}/{s.n-1}"
        )


if __name__ == "__main__":
    main()
