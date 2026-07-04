#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_statistics.py

Representative failure-mode ablation for the symmetry-aware Rayleigh--Ritz PINN paper.

Place in:
    PINN/scripts/run_ablation_statistics.py

Run from project root:
    python scripts/run_ablation_statistics.py --root . --seeds 20 --quick

Outputs:
    data/ablation_runs.csv
    data/ablation_summary.csv
    figures/Fig5_ablation_success_rates.pdf
    figures/Fig5_ablation_success_rates.png

Important:
    This is a representative ablation, not an exhaustive all-methods/all-geometries benchmark.
    It quantifies the stability of admissible formulations over random seeds.
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
# Style and utilities
# -----------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.7,
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


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def relerr(pred: float, exact: float) -> float:
    return 100.0 * abs(pred - exact) / max(abs(exact), 1e-14)


def trapz(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # torch.trapezoid is available in modern PyTorch; this wrapper keeps notation short.
    return torch.trapezoid(y, x)


def weighted_inner(f: torch.Tensor, g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.sum(w * f * g) / (torch.sum(w) + 1e-14)


def normalize_weighted(f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return f / torch.sqrt(weighted_inner(f, f, w) + 1e-14)


def count_nodes_1d(x: np.ndarray, y: np.ndarray) -> int:
    amp = max(float(np.max(np.abs(y))), 1e-12)
    mask = np.abs(y) > 1e-4 * amp
    yy = y[mask]
    if len(yy) < 3:
        return 0
    s = np.sign(yy)
    return int(np.sum(s[:-1] * s[1:] < 0))


# -----------------------------------------------------------------------------
# Neural network
# -----------------------------------------------------------------------------

class FourierMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, width: int = 64, depth: int = 3,
                 n_fourier: int = 6, sigma: float = 3.0, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.n_fourier = n_fourier
        if n_fourier > 0:
            self.register_buffer("B", sigma * torch.randn(in_dim, n_fourier, dtype=dtype))
            feat_dim = in_dim + 2 * n_fourier
        else:
            self.register_buffer("B", torch.empty(in_dim, 0, dtype=dtype))
            feat_dim = in_dim

        layers: List[nn.Module] = []
        last = feat_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width, dtype=dtype))
            layers.append(nn.Tanh())
            last = width
        layers.append(nn.Linear(last, out_dim, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_fourier <= 0:
            return x
        z = 2.0 * math.pi * x @ self.B
        return torch.cat([x, torch.sin(z), torch.cos(z)], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(x))


# -----------------------------------------------------------------------------
# 1D infinite well ablation
# -----------------------------------------------------------------------------

def psi_exact_1d(x: torch.Tensor, n: int) -> torch.Tensor:
    return math.sqrt(2.0) * torch.sin(n * math.pi * x)


def boundary_1d(x: torch.Tensor) -> torch.Tensor:
    return x * (1.0 - x)


def rayleigh_1d(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    psi_raw = boundary_1d(x_req) * model(x_req[:, None])[:, 0]
    dpsi = torch.autograd.grad(psi_raw, x_req, torch.ones_like(psi_raw), create_graph=True)[0]
    den = trapz(psi_raw**2, x_req) + 1e-14
    E = trapz(dpsi**2, x_req) / den
    psi = psi_raw / torch.sqrt(den)
    return E, psi


def project_1d(psi: torch.Tensor, x: torch.Tensor, n_target: int) -> torch.Tensor:
    out = psi
    for j in range(1, n_target):
        pj = psi_exact_1d(x, j)
        out = out - trapz(out * pj, x) * pj
    return out / torch.sqrt(trapz(out**2, x) + 1e-14)


def rayleigh_1d_projected(model: nn.Module, x: torch.Tensor, n_target: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().requires_grad_(True)
    psi_raw = boundary_1d(x_req) * model(x_req[:, None])[:, 0]
    psi = project_1d(psi_raw, x_req, n_target)
    dpsi = torch.autograd.grad(psi, x_req, torch.ones_like(psi), create_graph=True)[0]
    E = trapz(dpsi**2, x_req) / (trapz(psi**2, x_req) + 1e-14)
    return E, psi


def train_1d(method: str, seed: int, args: argparse.Namespace) -> Dict[str, object]:
    set_seed(seed)
    dtype = torch.float64
    device = torch.device(args.device)
    n_target = 5
    exact = (n_target * math.pi) ** 2
    x = torch.linspace(0, 1, args.nq, dtype=dtype, device=device)
    model = FourierMLP(1, 1, args.width, args.depth, args.n_fourier, dtype=dtype).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    t0 = time.time()

    for _ in range(args.epochs_1d):
        opt.zero_grad(set_to_none=True)
        if method == "projection":
            E, psi = rayleigh_1d_projected(model, x, n_target)
            loss = E
        elif method == "penalty":
            E, psi = rayleigh_1d(model, x)
            ortho = torch.tensor(0.0, dtype=dtype, device=device)
            for j in range(1, n_target):
                ortho = ortho + trapz(psi * psi_exact_1d(x, j), x)**2
            loss = E + args.lambda_ortho * ortho
        elif method == "residual":
            # Representative naive baseline: residual + norm. E is estimated by Rayleigh for stability.
            x_req = x.detach().clone().requires_grad_(True)
            psi_raw = boundary_1d(x_req) * model(x_req[:, None])[:, 0]
            dpsi = torch.autograd.grad(psi_raw, x_req, torch.ones_like(psi_raw), create_graph=True)[0]
            ddpsi = torch.autograd.grad(dpsi, x_req, torch.ones_like(dpsi), create_graph=True)[0]
            den = trapz(psi_raw**2, x_req) + 1e-14
            E = trapz(dpsi**2, x_req) / den
            res = -ddpsi - E.detach() * psi_raw
            loss = torch.mean(res**2) + 10.0 * (den - 1.0)**2
        else:
            raise ValueError(method)
        loss.backward()
        opt.step()

    with torch.enable_grad():
        if method == "projection":
            E, psi = rayleigh_1d_projected(model, x, n_target)
        else:
            E, psi = rayleigh_1d(model, x)

    energy = float(E.detach().cpu())
    x_np = x.detach().cpu().numpy()
    psi_np = psi.detach().cpu().numpy()
    nodes = count_nodes_1d(x_np, psi_np)
    err = relerr(energy, exact)
    success = (err < args.success_1d_error) and (nodes == n_target - 1)
    return {
        "case": "1D n=5",
        "method": method,
        "seed": seed,
        "success": bool(success),
        "energy": energy,
        "exact": exact,
        "rel_error_percent": err,
        "max_rel_error_percent": err,
        "nodes": nodes,
        "nodes_expected": n_target - 1,
        "subspace_weight": np.nan,
        "parity_score": np.nan,
        "runtime_s": time.time() - t0,
        "failure_mode": "stable" if success else ("wrong nodes/lower-state collapse" if nodes != n_target - 1 else "energy error"),
    }


# -----------------------------------------------------------------------------
# Disk m=1 doublet ablation
# -----------------------------------------------------------------------------

def disk_grid(nr: int, nt: int, dtype: torch.dtype, device: torch.device):
    r = torch.linspace(0.0, 1.0, nr, dtype=dtype, device=device)
    theta = torch.linspace(0.0, 2.0 * math.pi, nt + 1, dtype=dtype, device=device)[:-1]
    rr, tt = torch.meshgrid(r, theta, indexing="ij")
    x = rr * torch.cos(tt)
    y = rr * torch.sin(tt)
    pts = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
    w = rr.reshape(-1)
    return pts, w


def disk_trial(model: nn.Module, pts: torch.Tensor, out_index: int = 0) -> torch.Tensor:
    r2 = torch.sum(pts**2, dim=1)
    y = model(pts)
    return (1.0 - r2) * y[:, out_index]


def energy_disk(psi: torch.Tensor, pts_req: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(psi, pts_req, torch.ones_like(psi), create_graph=True)[0]
    return torch.sum(w * torch.sum(g**2, dim=1)) / (torch.sum(w * psi**2) + 1e-14)


def disk_m1_weight(psi: torch.Tensor, pts: torch.Tensor, w: torch.Tensor) -> float:
    r = torch.sqrt(torch.sum(pts**2, dim=1)).clamp_min(1e-12)
    theta = torch.atan2(pts[:, 1], pts[:, 0])
    alpha = 3.8317059702075125
    try:
        radial = torch.special.bessel_j1(alpha * r)
    except Exception:
        radial = r * (1.0 - r**2)
    c = normalize_weighted(radial * torch.cos(theta), w)
    s = radial * torch.sin(theta)
    s = normalize_weighted(s - weighted_inner(s, c, w) * c, w)
    p = normalize_weighted(psi, w)
    val = weighted_inner(p, c, w)**2 + weighted_inner(p, s, w)**2
    return float(torch.clamp(val, 0.0, 1.5).detach().cpu())


def train_disk(method: str, seed: int, args: argparse.Namespace) -> Dict[str, object]:
    set_seed(seed)
    dtype = torch.float64
    device = torch.device(args.device)
    pts, w = disk_grid(args.nr, args.nt, dtype, device)
    exact_m1 = 3.8317059702075125**2
    t0 = time.time()

    if method == "state_by_state":
        model = FourierMLP(2, 1, args.width, args.depth, args.n_fourier_2d, dtype=dtype).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for _ in range(args.epochs_2d):
            opt.zero_grad(set_to_none=True)
            pts_req = pts.detach().clone().requires_grad_(True)
            psi = disk_trial(model, pts_req)
            E = energy_disk(psi, pts_req, w)
            ground = normalize_weighted(1.0 - torch.sum(pts_req**2, dim=1), w)
            psi_n = normalize_weighted(psi, w)
            loss = E + args.lambda_ortho * weighted_inner(psi_n, ground, w)**2
            loss.backward()
            opt.step()
        with torch.enable_grad():
            pts_req = pts.detach().clone().requires_grad_(True)
            psi = disk_trial(model, pts_req)
            E = energy_disk(psi, pts_req, w)
        energy = float(E.detach().cpu())
        sw = disk_m1_weight(psi.detach(), pts, w)
        err = relerr(energy, exact_m1)
        success = (err < args.success_disk_error) and (sw > args.success_subspace_weight)

    elif method == "subspace_ritz":
        model = FourierMLP(2, 2, args.width, args.depth, args.n_fourier_2d, dtype=dtype).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        eigvals = None
        modes = None
        for _ in range(args.epochs_2d):
            opt.zero_grad(set_to_none=True)
            pts_req = pts.detach().clone().requires_grad_(True)
            Phi = torch.stack([disk_trial(model, pts_req, 0), disk_trial(model, pts_req, 1)], dim=1)
            grads = [torch.autograd.grad(Phi[:, k], pts_req, torch.ones_like(Phi[:, k]), create_graph=True, retain_graph=True)[0] for k in range(2)]
            K = torch.zeros((2, 2), dtype=dtype, device=device)
            M = torch.zeros((2, 2), dtype=dtype, device=device)
            for i in range(2):
                for j in range(2):
                    K[i, j] = torch.sum(w * torch.sum(grads[i] * grads[j], dim=1))
                    M[i, j] = torch.sum(w * Phi[:, i] * Phi[:, j])
            M = M + args.mass_reg * torch.eye(2, dtype=dtype, device=device)
            try:
                L = torch.linalg.cholesky(M)
                A = torch.cholesky_solve(K, L)
                eigvals = torch.linalg.eigvalsh(A)
                loss = eigvals[0] + eigvals[1]
            except Exception:
                loss = torch.tensor(1e6, dtype=dtype, device=device)
            loss.backward()
            opt.step()
        # evaluate modes
        with torch.enable_grad():
            pts_req = pts.detach().clone().requires_grad_(True)
            Phi = torch.stack([disk_trial(model, pts_req, 0), disk_trial(model, pts_req, 1)], dim=1)
            grads = [torch.autograd.grad(Phi[:, k], pts_req, torch.ones_like(Phi[:, k]), create_graph=True, retain_graph=True)[0] for k in range(2)]
            K = torch.zeros((2, 2), dtype=dtype, device=device)
            M = torch.zeros((2, 2), dtype=dtype, device=device)
            for i in range(2):
                for j in range(2):
                    K[i, j] = torch.sum(w * torch.sum(grads[i] * grads[j], dim=1))
                    M[i, j] = torch.sum(w * Phi[:, i] * Phi[:, j])
            M = M + args.mass_reg * torch.eye(2, dtype=dtype, device=device)
            L = torch.linalg.cholesky(M)
            A = torch.cholesky_solve(K, L)
            eigvals, eigvecs = torch.linalg.eigh(A)
            modes = Phi @ eigvecs
        w1 = disk_m1_weight(modes[:, 0].detach(), pts, w)
        w2 = disk_m1_weight(modes[:, 1].detach(), pts, w)
        sw = min(w1, w2)
        energy = float(0.5 * (eigvals[0] + eigvals[1]).detach().cpu())
        err = 0.5 * (relerr(float(eigvals[0]), exact_m1) + relerr(float(eigvals[1]), exact_m1))
        success = (err < args.success_disk_error) and (sw > args.success_subspace_weight)
    else:
        raise ValueError(method)

    return {
        "case": "disk m=1",
        "method": method,
        "seed": seed,
        "success": bool(success),
        "energy": energy,
        "exact": exact_m1,
        "rel_error_percent": float(err),
        "max_rel_error_percent": float(err),
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "subspace_weight": float(sw),
        "parity_score": np.nan,
        "runtime_s": time.time() - t0,
        "failure_mode": "stable" if success else ("low subspace weight" if sw < args.success_subspace_weight else "energy error"),
    }


# -----------------------------------------------------------------------------
# Ellipse splitting ablation
# -----------------------------------------------------------------------------

def ellipse_fd_reference(eta: float) -> Tuple[float, float]:
    # Project values already used in the manuscript logs.
    table = {
        0.8: (16.512141, 20.581937),
        0.9: (15.349390, 17.053649),
        1.0: (14.512858, 14.512858),
        1.1: (12.619688, 13.876876),
        1.2: (11.171859, 13.384234),
    }
    return table[round(float(eta), 1)]


def ellipse_grid(a: float, b: float, nr: int, nt: int, dtype: torch.dtype, device: torch.device):
    pts_uv, w = disk_grid(nr, nt, dtype, device)
    return pts_uv, w * (a * b)


def ellipse_sector_trial(model: nn.Module, uv: torch.Tensor, sector: str) -> torch.Tensor:
    u = uv[:, 0]
    v = uv[:, 1]
    r2 = u**2 + v**2
    amp = model(torch.stack([u**2, v**2], dim=1))[:, 0]
    if sector == "x":
        return (1.0 - r2) * u * amp
    if sector == "y":
        return (1.0 - r2) * v * amp
    raise ValueError(sector)


def ellipse_energy(psi: torch.Tensor, uv_req: torch.Tensor, w: torch.Tensor, a: float, b: float) -> torch.Tensor:
    g = torch.autograd.grad(psi, uv_req, torch.ones_like(psi), create_graph=True)[0]
    return torch.sum(w * ((g[:, 0]**2) / (a*a) + (g[:, 1]**2) / (b*b))) / (torch.sum(w * psi**2) + 1e-14)


def train_ellipse(method: str, seed: int, args: argparse.Namespace) -> Dict[str, object]:
    set_seed(seed)
    dtype = torch.float64
    device = torch.device(args.device)
    etas = [0.8, 0.9, 1.0, 1.1, 1.2]
    errors: List[float] = []
    ordering_ok: List[bool] = []
    t0 = time.time()

    if method == "symmetry_sectors":
        for eta in etas:
            a, b = 1.0, eta
            uv, w = ellipse_grid(a, b, args.nr, args.nt, dtype, device)
            energies: Dict[str, float] = {}
            for sector in ["x", "y"]:
                model = FourierMLP(2, 1, args.width, args.depth, args.n_fourier_2d, dtype=dtype).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=args.lr)
                for _ in range(args.epochs_ellipse):
                    opt.zero_grad(set_to_none=True)
                    uv_req = uv.detach().clone().requires_grad_(True)
                    psi = ellipse_sector_trial(model, uv_req, sector)
                    E = ellipse_energy(psi, uv_req, w, a, b)
                    E.backward()
                    opt.step()
                with torch.enable_grad():
                    uv_req = uv.detach().clone().requires_grad_(True)
                    psi = ellipse_sector_trial(model, uv_req, sector)
                    E = ellipse_energy(psi, uv_req, w, a, b)
                energies[sector] = float(E.detach().cpu())
            fd2, fd3 = ellipse_fd_reference(eta)
            pred = sorted([energies["x"], energies["y"]])
            exact = [fd2, fd3]
            errors += [relerr(pred[0], exact[0]), relerr(pred[1], exact[1])]
            if eta < 1:
                ordering_ok.append(energies["y"] > energies["x"])
            elif eta > 1:
                ordering_ok.append(energies["y"] < energies["x"])
            else:
                ordering_ok.append(abs(energies["y"] - energies["x"]) < args.success_ellipse_degen_abs)

    elif method == "generic_subspace":
        for eta in etas:
            a, b = 1.0, eta
            uv, w = ellipse_grid(a, b, args.nr, args.nt, dtype, device)
            model = FourierMLP(2, 2, args.width, args.depth, args.n_fourier_2d, dtype=dtype).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            eigvals = None
            for _ in range(args.epochs_ellipse):
                opt.zero_grad(set_to_none=True)
                uv_req = uv.detach().clone().requires_grad_(True)
                r2 = torch.sum(uv_req**2, dim=1)
                Phi = (1.0 - r2)[:, None] * model(uv_req)
                grads = [torch.autograd.grad(Phi[:, k], uv_req, torch.ones_like(Phi[:, k]), create_graph=True, retain_graph=True)[0] for k in range(2)]
                K = torch.zeros((2, 2), dtype=dtype, device=device)
                M = torch.zeros((2, 2), dtype=dtype, device=device)
                for i in range(2):
                    for j in range(2):
                        K[i, j] = torch.sum(w * ((grads[i][:, 0] * grads[j][:, 0]) / (a*a) + (grads[i][:, 1] * grads[j][:, 1]) / (b*b)))
                        M[i, j] = torch.sum(w * Phi[:, i] * Phi[:, j])
                M = M + args.mass_reg * torch.eye(2, dtype=dtype, device=device)
                try:
                    L = torch.linalg.cholesky(M)
                    A = torch.cholesky_solve(K, L)
                    eigvals = torch.linalg.eigvalsh(A)
                    loss = eigvals[0] + eigvals[1]
                except Exception:
                    loss = torch.tensor(1e6, dtype=dtype, device=device)
                loss.backward()
                opt.step()
            fd2, fd3 = ellipse_fd_reference(eta)
            pred = sorted([float(eigvals[0].detach().cpu()), float(eigvals[1].detach().cpu())]) if eigvals is not None else [np.inf, np.inf]
            errors += [relerr(pred[0], fd2), relerr(pred[1], fd3)]
            ordering_ok.append(False)
    else:
        raise ValueError(method)

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    success = (mean_err < args.success_ellipse_error_mean) and all(ordering_ok)
    return {
        "case": "ellipse split",
        "method": method,
        "seed": seed,
        "success": bool(success),
        "energy": np.nan,
        "exact": np.nan,
        "rel_error_percent": mean_err,
        "max_rel_error_percent": max_err,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "subspace_weight": np.nan,
        "parity_score": float(np.mean(ordering_ok)),
        "runtime_s": time.time() - t0,
        "failure_mode": "stable branch tracking" if success else "branch drift/missing parity sector",
    }


# -----------------------------------------------------------------------------
# Summary and plotting
# -----------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, method), g in df.groupby(["case", "method"]):
        rows.append({
            "case": case,
            "method": method,
            "n_seeds": len(g),
            "success_rate_percent": 100.0 * g["success"].mean(),
            "mean_rel_error_percent": pd.to_numeric(g["rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean(),
            "std_rel_error_percent": pd.to_numeric(g["rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).std(ddof=1),
            "max_rel_error_percent": pd.to_numeric(g["max_rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).max(),
            "mean_subspace_weight": pd.to_numeric(g["subspace_weight"], errors="coerce").mean(),
            "mean_parity_score": pd.to_numeric(g["parity_score"], errors="coerce").mean(),
            "mean_runtime_s": pd.to_numeric(g["runtime_s"], errors="coerce").mean(),
            "typical_failure_mode": g["failure_mode"].mode().iloc[0] if len(g) else "",
        })
    return pd.DataFrame(rows)


def plot_success(summary: pd.DataFrame, fig_dir: Path) -> None:
    setup_style()
    labels = [f"{r.case}\n{r.method}" for _, r in summary.iterrows()]
    vals = summary["success_rate_percent"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(vals))
    ax.bar(x, vals, color="0.58", edgecolor="0.15", linewidth=0.7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Representative failure-mode ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(100.0, color="0.15", lw=0.8)
    fig.savefig(fig_dir / "Fig5_ablation_success_rates.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(fig_dir / "Fig5_ablation_success_rates.png", dpi=600, bbox_inches="tight", pad_inches=0.025)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--n_fourier", type=int, default=6)
    p.add_argument("--n_fourier_2d", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--epochs", type=int, default=1600)
    p.add_argument("--epochs_1d", type=int, default=None)
    p.add_argument("--epochs_2d", type=int, default=None)
    p.add_argument("--epochs_ellipse", type=int, default=None)
    p.add_argument("--nq", type=int, default=160)
    p.add_argument("--nr", type=int, default=45)
    p.add_argument("--nt", type=int, default=96)
    p.add_argument("--lambda_ortho", type=float, default=200.0)
    p.add_argument("--mass_reg", type=float, default=1e-8)
    p.add_argument("--success_1d_error", type=float, default=0.1)
    p.add_argument("--success_disk_error", type=float, default=0.2)
    p.add_argument("--success_subspace_weight", type=float, default=0.90)
    p.add_argument("--success_ellipse_error_mean", type=float, default=3.0)
    p.add_argument("--success_ellipse_degen_abs", type=float, default=0.5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 700)
        args.nq = min(args.nq, 120)
        args.nr = min(args.nr, 32)
        args.nt = min(args.nt, 64)
        args.width = min(args.width, 48)
    args.epochs_1d = args.epochs if args.epochs_1d is None else args.epochs_1d
    args.epochs_2d = args.epochs if args.epochs_2d is None else args.epochs_2d
    args.epochs_ellipse = args.epochs if args.epochs_ellipse is None else args.epochs_ellipse

    root = Path(args.root).resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)

    plan = []
    for seed in range(args.seeds):
        plan.extend([
            ("1d", "residual", seed),
            ("1d", "penalty", seed),
            ("1d", "projection", seed),
            ("disk", "state_by_state", seed),
            ("disk", "subspace_ritz", seed),
            ("ellipse", "generic_subspace", seed),
            ("ellipse", "symmetry_sectors", seed),
        ])

    print("Representative ablation")
    print(f"  root: {root}")
    print(f"  seeds: {args.seeds}")
    print(f"  quick: {args.quick}")
    print(f"  total runs: {len(plan)}")

    rows: List[Dict[str, object]] = []
    for i, (case, method, seed) in enumerate(plan, start=1):
        print(f"[{i:03d}/{len(plan):03d}] {case:8s} {method:16s} seed={seed}")
        try:
            if case == "1d":
                row = train_1d(method, seed, args)
            elif case == "disk":
                row = train_disk(method, seed, args)
            elif case == "ellipse":
                row = train_ellipse(method, seed, args)
            else:
                raise ValueError(case)
        except Exception as exc:
            row = {
                "case": case,
                "method": method,
                "seed": seed,
                "success": False,
                "energy": np.nan,
                "exact": np.nan,
                "rel_error_percent": np.inf,
                "max_rel_error_percent": np.inf,
                "nodes": np.nan,
                "nodes_expected": np.nan,
                "subspace_weight": np.nan,
                "parity_score": np.nan,
                "runtime_s": np.nan,
                "failure_mode": f"exception: {type(exc).__name__}: {exc}",
            }
            print("    ERROR:", row["failure_mode"])
        rows.append(row)
        pd.DataFrame(rows).to_csv(data_dir / "ablation_runs.csv", index=False)

    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    runs.to_csv(data_dir / "ablation_runs.csv", index=False)
    summary.to_csv(data_dir / "ablation_summary.csv", index=False)
    plot_success(summary, fig_dir)

    print("\nSaved outputs:")
    print(f"  {data_dir / 'ablation_runs.csv'}")
    print(f"  {data_dir / 'ablation_summary.csv'}")
    print(f"  {fig_dir / 'Fig5_ablation_success_rates.pdf'}")
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
