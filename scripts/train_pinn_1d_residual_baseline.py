#!/usr/bin/env python3
"""Pointwise-residual 1D PINN baseline for the n=5 infinite-well state.

This script intentionally avoids Rayleigh minimization and explicit projection.
The trained objective is the strong-form residual together with a normalization
constraint, which is needed to avoid the trivial zero solution.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt


OUT_DATA = ROOT / "data"
OUT_FIG = ROOT / "figures"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def make_regime_figure(df: pd.DataFrame) -> None:
    setup_style()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    order = ["informed_5pct", "wide_100pct", "misleading_n1"]
    labels = ["informed\n5%", "wide\n100%", "near\n$n=1$"]
    success = [100.0 * df[df["init_regime"] == r]["success"].mean() for r in order]
    errors = [df[df["init_regime"] == r]["rel_error_rayleigh_percent"].to_numpy() for r in order]

    fig, axs = plt.subplots(1, 2, figsize=(5.35, 2.25), gridspec_kw={"width_ratios": [0.9, 1.25]})
    ax = axs[0]
    ax.bar(range(len(order)), success, color="0.18", width=0.62)
    ax.set_ylabel("success rate (%)")
    ax.set_xticks(range(len(order)), labels)
    ax.set_ylim(0, 105)
    ax.text(-0.18, 1.04, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axs[1]
    for i, vals in enumerate(errors):
        x = np.full_like(vals, i, dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) else []
        ax.plot(x + jitter, vals, "o", ms=3.0, mfc="white", mec="0.08", mew=0.7)
        if len(vals):
            ax.hlines(np.median(vals), i - 0.22, i + 0.22, color="0.08", lw=1.1)
    ax.set_ylabel("post-hoc Rayleigh error (%)")
    ax.set_xticks(range(len(order)), labels)
    ax.set_yscale("log")
    ax.text(-0.14, 1.04, "(b)", transform=ax.transAxes, fontweight="bold")
    fig.tight_layout(pad=0.25, w_pad=1.0)
    out = OUT_FIG / "Fig_residual_init_regimes"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


class WellNet(nn.Module):
    def __init__(self, hidden_width: int, hidden_layers: int, fourier_features: int) -> None:
        super().__init__()
        self.fourier_features = int(fourier_features)
        in_features = 1 + 2 * self.fourier_features
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layer = nn.Linear(in_features, hidden_width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            layers.append(nn.Tanh())
            in_features = hidden_width
        out = nn.Linear(in_features, 1)
        nn.init.xavier_normal_(out.weight)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = 2.0 * x - 1.0
        terms = [z]
        for k in range(1, self.fourier_features + 1):
            angle = k * torch.pi * x
            terms.append(torch.sin(angle))
            terms.append(torch.cos(angle))
        return torch.cat(terms, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 - x) * self.net(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.trapezoid(y, x)


def derivatives(psi: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dpsi = torch.autograd.grad(
        psi,
        x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
    )[0]
    ddpsi = torch.autograd.grad(
        dpsi,
        x,
        grad_outputs=torch.ones_like(dpsi),
        create_graph=True,
        retain_graph=True,
    )[0]
    return dpsi, ddpsi


def count_internal_nodes(x_np: np.ndarray, psi_np: np.ndarray, trim: float = 1.0e-3) -> int:
    mask = (x_np > x_np.min() + trim) & (x_np < x_np.max() - trim)
    y = psi_np[mask].copy()
    if y.size < 3:
        return 0
    threshold = 1.0e-3 * max(1.0, float(np.nanmax(np.abs(y))))
    y[np.abs(y) < threshold] = 0.0
    signs = np.sign(y)
    nonzero = signs[signs != 0]
    if nonzero.size < 2:
        return 0
    return int(np.sum(nonzero[1:] * nonzero[:-1] < 0))


def initial_epsilon(args: argparse.Namespace) -> float:
    target_exact = (args.target_n * math.pi) ** 2
    if args.init_regime == "informed_5pct":
        center = target_exact
        jitter = 0.05
    elif args.init_regime == "wide_100pct":
        center = target_exact
        jitter = 1.0
    elif args.init_regime == "misleading_n1":
        center = math.pi**2
        jitter = 0.05
    else:
        raise ValueError(f"unknown init regime: {args.init_regime}")
    return max(1.0e-6, center * (1.0 + jitter * np.random.uniform(-1.0, 1.0)))


def train_one(seed: int, args: argparse.Namespace) -> dict[str, object]:
    set_seed(seed)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    x = torch.linspace(0.0, 1.0, args.n_collocation, dtype=dtype, device=device).requires_grad_(True)
    x_col = x[:, None]

    model = WellNet(args.hidden_width, args.hidden_layers, args.fourier_features).to(device=device, dtype=dtype)

    target_exact = (args.target_n * math.pi) ** 2
    # Softplus keeps epsilon positive while allowing the optimizer to move it.
    init = initial_epsilon(args)
    rho = torch.nn.Parameter(torch.tensor(init, dtype=dtype, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [rho], lr=args.lr)
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        psi = model(x_col)[:, 0]
        dpsi, ddpsi = derivatives(psi, x)
        epsilon = torch.nn.functional.softplus(rho)
        norm = trapz(psi.pow(2), x)
        residual = -ddpsi - epsilon * psi
        residual_loss = trapz(residual[1:-1].pow(2), x[1:-1])
        norm_loss = (norm - 1.0).pow(2)
        loss = residual_loss + args.lambda_norm * norm_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [rho], args.grad_clip)
        opt.step()
        if args.verbose and (epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs):
            print(
                f"seed={seed:03d} epoch={epoch:5d}: loss={loss.item():.3e}, "
                f"eps={epsilon.item():.6f}, norm={norm.item():.6f}, res={residual_loss.item():.3e}"
            )

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            list(model.parameters()) + [rho],
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            psi = model(x_col)[:, 0]
            _, ddpsi = derivatives(psi, x)
            epsilon = torch.nn.functional.softplus(rho)
            norm = trapz(psi.pow(2), x)
            residual = -ddpsi - epsilon * psi
            loss = trapz(residual[1:-1].pow(2), x[1:-1]) + args.lambda_norm * (norm - 1.0).pow(2)
            loss.backward()
            return loss

        lbfgs.step(closure)

    psi = model(x_col)[:, 0]
    dpsi, ddpsi = derivatives(psi, x)
    epsilon = torch.nn.functional.softplus(rho)
    norm = trapz(psi.pow(2), x)
    residual = -ddpsi - epsilon * psi
    residual_loss = trapz(residual[1:-1].pow(2), x[1:-1])
    rayleigh = trapz(dpsi.pow(2), x) / (norm + 1.0e-14)

    x_np = x.detach().cpu().numpy()
    psi_np = psi.detach().cpu().numpy()
    norm_np = math.sqrt(max(float(np.trapz(psi_np**2, x_np)), 1.0e-30))
    psi_np = psi_np / norm_np
    nodes = count_internal_nodes(x_np, psi_np, trim=1.0 / args.n_collocation)
    err_param = 100.0 * abs(float(epsilon.detach().cpu()) - target_exact) / target_exact
    err_rayleigh = 100.0 * abs(float(rayleigh.detach().cpu()) - target_exact) / target_exact
    success = bool(err_rayleigh < args.success_error_percent and nodes == args.target_n - 1)

    return {
        "case": "1D n=5",
        "method": "Residual PINN (pointwise)",
        "init_regime": args.init_regime,
        "seed": seed,
        "success": success,
        "epsilon_initial": init,
        "epsilon_parameter": float(epsilon.detach().cpu()),
        "epsilon_rayleigh_posthoc": float(rayleigh.detach().cpu()),
        "epsilon_exact": target_exact,
        "rel_error_parameter_percent": err_param,
        "rel_error_rayleigh_percent": err_rayleigh,
        "nodes": nodes,
        "nodes_expected": args.target_n - 1,
        "nodal_destination_n": nodes + 1,
        "norm_integral": float(norm.detach().cpu()),
        "residual_integral": float(residual_loss.detach().cpu()),
        "runtime_s": time.time() - t0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-n", type=int, default=5)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--n-collocation", type=int, default=160)
    p.add_argument("--hidden-width", type=int, default=64)
    p.add_argument("--hidden-layers", type=int, default=3)
    p.add_argument("--fourier-features", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-norm", type=float, default=100.0)
    p.add_argument(
        "--init-regime",
        choices=["informed_5pct", "wide_100pct", "misleading_n1"],
        default="informed_5pct",
    )
    p.add_argument("--all-init-regimes", action="store_true")
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--lbfgs-lr", type=float, default=0.8)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--success-error-percent", type=float, default=1.0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--print-every", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.all_init_regimes:
        regimes = ["informed_5pct", "wide_100pct", "misleading_n1"]
        out = OUT_DATA / "ablation_residual_baseline_control.csv"
    else:
        regimes = [args.init_regime]
        out = OUT_DATA / "ablation_residual_baseline.csv"

    for regime in regimes:
        args.init_regime = regime
        print(f"\n=== init_regime={regime} ===")
        for k in range(args.seeds):
            seed = args.seed0 + k
            row = train_one(seed, args)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(
                f"seed={seed:03d}: success={row['success']}, "
                f"eps0={row['epsilon_initial']:.6f}, "
                f"eps={row['epsilon_parameter']:.6f}, "
                f"R={row['epsilon_rayleigh_posthoc']:.6f}, "
                f"errR={row['rel_error_rayleigh_percent']:.3f}%, "
                f"nodes={row['nodes']} (n~{row['nodal_destination_n']})"
            )
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if args.all_init_regimes:
        make_regime_figure(df)
    print("\nSummary:")
    summary = df.groupby("init_regime").agg(
        seeds=("seed", "count"),
        success_rate=("success", "mean"),
        mean_rayleigh_error_percent=("rel_error_rayleigh_percent", "mean"),
        max_rayleigh_error_percent=("rel_error_rayleigh_percent", "max"),
        nodal_destinations=("nodal_destination_n", lambda s: dict(sorted(s.value_counts().items()))),
    )
    print(summary.to_string())
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
