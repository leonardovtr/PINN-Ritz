#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Seed-dependent basis selection inside the degenerate square-well eigenspace.

This script reuses train_pinn_square_2d_subspace.py and tests whether the
Rayleigh--Ritz neural eigensolver selects a unique analytical basis or different
orthonormal rotations inside the same degenerate eigenspace.

Target eigenspace:
    span{psi_12, psi_21}

Outputs:
    data/square_basis_selection.csv
    figures/square_basis_selection.pdf
    figures/square_basis_selection.png
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# Make sure the local scripts directory is importable.
THIS_FILE = Path(__file__).resolve()
SCRIPTS_DIR = THIS_FILE.parent
ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import train_pinn_square_2d_subspace as sq  # noqa: E402


def setup_publication_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 10.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "lines.linewidth": 1.4,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze seed-dependent basis selection in the square-well degenerate eigenspace."
    )
    p.add_argument("--root", type=str, default=str(ROOT))
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed0", type=int, default=1000)

    # Training settings passed to the original square script.
    p.add_argument("--states", type=int, default=3,
                   help="Use 3 to learn ground state plus the (1,2)/(2,1) doublet.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--n-quad", type=int, default=32)
    p.add_argument("--n-validate", type=int, default=64)
    p.add_argument("--hidden-width", type=int, default=48)
    p.add_argument("--hidden-layers", type=int, default=2)
    p.add_argument("--fourier-features", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lbfgs-steps", type=int, default=0)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--log-every", type=int, default=1000000,
                   help="Large default suppresses most per-epoch logs.")
    p.add_argument("--check-every", type=int, default=100)

    return p.parse_args()


def make_square_args(user_args, seed: int):
    """Build an argument namespace using the original parser."""
    argv = [
        "--states", str(user_args.states),
        "--lx", "1.0",
        "--ly", "1.0",
        "--n-quad", str(user_args.n_quad),
        "--n-validate", str(user_args.n_validate),
        "--hidden-width", str(user_args.hidden_width),
        "--hidden-layers", str(user_args.hidden_layers),
        "--fourier-features", str(user_args.fourier_features),
        "--restarts", str(user_args.restarts),
        "--epochs", str(user_args.epochs),
        "--lr", str(user_args.lr),
        "--lbfgs-steps", str(user_args.lbfgs_steps),
        "--dtype", user_args.dtype,
        "--device", user_args.device,
        "--threads", str(user_args.threads),
        "--seed", str(seed),
        "--log-every", str(user_args.log_every),
        "--check-every", str(user_args.check_every),
        "--max-exact-n", "5",
    ]
    return sq.parse_args(argv)


def compute_doublet_diagnostics(model, args, device, dtype):
    """Compute Ritz modes and their projections onto psi_12 and psi_21."""
    xyv, wv = sq.gauss_legendre_square(args.n_validate, args.lx, args.ly, device, dtype)

    eigvals, modes, _, _ = sq.validation_eigs_grad(model, xyv, wv, args)
    eigvals_np = eigvals.detach().cpu().numpy()

    psi12 = sq.exact_mode_torch(xyv, 1, 2, args.lx, args.ly)
    psi21 = sq.exact_mode_torch(xyv, 2, 1, args.lx, args.ly)

    # In states=3, index 0 is ground, indices 1 and 2 are the square doublet.
    # For safety, use the two modes with largest total weight in span{psi12,psi21}.
    rows_modes = []
    for i in range(args.states):
        c12 = float(torch.sum(wv * modes[:, i] * psi12).detach().cpu())
        c21 = float(torch.sum(wv * modes[:, i] * psi21).detach().cpu())
        w12 = c12 * c12
        w21 = c21 * c21
        rows_modes.append({
            "mode_index": i + 1,
            "epsilon": eigvals_np[i],
            "c12": c12,
            "c21": c21,
            "w12": w12,
            "w21": w21,
            "w_subspace": w12 + w21,
        })

    rows_sorted = sorted(rows_modes, key=lambda r: r["w_subspace"], reverse=True)
    doublet = rows_sorted[:2]

    # Overlap matrix between learned doublet modes and analytical doublet.
    C = np.array([[doublet[0]["c12"], doublet[0]["c21"]],
                  [doublet[1]["c12"], doublet[1]["c21"]]], dtype=float)

    # Singular values of C are cosines of principal angles if both bases are orthonormal.
    svals = np.linalg.svd(C, compute_uv=False)
    svals = np.clip(svals, 0.0, 1.0)
    angles_deg = np.degrees(np.arccos(svals))
    max_angle_deg = float(np.max(angles_deg))

    # Basis orientation of the first selected doublet mode.
    # Because signs are arbitrary, map angle modulo pi.
    theta = math.atan2(doublet[0]["c21"], doublet[0]["c12"])
    theta_mod_pi = theta % math.pi

    return rows_modes, doublet, C, svals, max_angle_deg, theta_mod_pi


def add_panel_label(ax, label: str):
    ax.text(
        0.035, 0.94, label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def make_figure(df_modes: pd.DataFrame, df_seed: pd.DataFrame, fig_dir: Path):
    setup_publication_style()

    # Use the selected first doublet mode from each seed.
    d = df_modes[df_modes["selected_doublet_rank"] == 1].copy()
    d = d.sort_values("seed_id")

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.05))
    ax1, ax2 = axs

    # Panel a: weights of one learned mode inside the analytical doublet.
    ax1.scatter(d["w12"], d["w21"], s=28, alpha=0.85, edgecolor="0.15", linewidth=0.3)
    xx = np.linspace(0, 1, 200)
    ax1.plot(xx, 1 - xx, "--", color="0.45", lw=1.1, label=r"$w_{12}+w_{21}=1$")
    ax1.set_xlim(-0.04, 1.04)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel(r"$w_{12}=|\langle \phi,\psi_{1,2}\rangle|^2$")
    ax1.set_ylabel(r"$w_{21}=|\langle \phi,\psi_{2,1}\rangle|^2$")
    ax1.set_title("basis selection inside the doublet")
    ax1.legend(frameon=False, loc="upper right")
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(2))
    add_panel_label(ax1, "(a)")

    # Panel b: subspace recovery and principal angle.
    ax2.plot(
        df_seed["seed_id"],
        df_seed["mean_doublet_subspace_weight"],
        "o",
        ms=4.0,
        label=r"mean $w_{\mathcal{S}}$",
    )
    ax2.set_xlabel("seed")
    ax2.set_ylabel(r"mean subspace weight")
    ax2.set_ylim(0.90, 1.005)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax2b = ax2.twinx()
    ax2b.plot(
        df_seed["seed_id"],
        df_seed["max_principal_angle_deg"],
        "s",
        ms=3.5,
        color="0.45",
        label="max principal angle",
    )
    ax2b.set_ylabel("max principal angle (deg)")
    ymax = max(1.0, 1.15 * float(df_seed["max_principal_angle_deg"].max()))
    ax2b.set_ylim(0.0, ymax)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="lower right")
    ax2.set_title("subspace recovery across seeds")
    add_panel_label(ax2, "(b)")

    fig.subplots_adjust(left=0.085, right=0.91, bottom=0.18, top=0.88, wspace=0.34)

    pdf_path = fig_dir / "square_basis_selection.pdf"
    png_path = fig_dir / "square_basis_selection.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)

    return pdf_path, png_path


def main():
    user_args = parse_args()
    root = Path(user_args.root).resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    if user_args.threads > 0:
        torch.set_num_threads(user_args.threads)

    device = torch.device(user_args.device)
    dtype = torch.float64 if user_args.dtype == "float64" else torch.float32

    all_mode_rows = []
    seed_rows = []

    print("Square-well degenerate-basis selection analysis")
    print(f"  root: {root}")
    print(f"  seeds: {user_args.seeds}")
    print(f"  training: states={user_args.states}, epochs={user_args.epochs}, restarts={user_args.restarts}")
    print(f"  device={device}, dtype={user_args.dtype}")

    for k in range(user_args.seeds):
        seed = user_args.seed0 + k
        print(f"\n=== seed {k + 1}/{user_args.seeds}: {seed} ===")

        args = make_square_args(user_args, seed)
        sq.set_seed(seed)

        model = sq.train(args, device, dtype)
        rows_modes, doublet, C, svals, max_angle_deg, theta_mod_pi = compute_doublet_diagnostics(
            model, args, device, dtype
        )

        # Mark which two learned modes were selected as doublet modes.
        selected_indices = [doublet[0]["mode_index"], doublet[1]["mode_index"]]
        for r in rows_modes:
            r = dict(r)
            r["seed_id"] = k
            r["seed"] = seed
            if r["mode_index"] == selected_indices[0]:
                r["selected_doublet_rank"] = 1
            elif r["mode_index"] == selected_indices[1]:
                r["selected_doublet_rank"] = 2
            else:
                r["selected_doublet_rank"] = 0
            all_mode_rows.append(r)

        mean_w = float(np.mean([doublet[0]["w_subspace"], doublet[1]["w_subspace"]]))
        min_w = float(np.min([doublet[0]["w_subspace"], doublet[1]["w_subspace"]]))

        seed_rows.append({
            "seed_id": k,
            "seed": seed,
            "epsilon_doublet_1": doublet[0]["epsilon"],
            "epsilon_doublet_2": doublet[1]["epsilon"],
            "mean_doublet_subspace_weight": mean_w,
            "min_doublet_subspace_weight": min_w,
            "singular_value_1": float(svals[0]),
            "singular_value_2": float(svals[1]),
            "max_principal_angle_deg": max_angle_deg,
            "basis_angle_mode1_rad_mod_pi": float(theta_mod_pi),
            "basis_angle_mode1_deg_mod_180": float(np.degrees(theta_mod_pi)),
        })

        print(
            f"  mean w_subspace={mean_w:.6f}, "
            f"min w_subspace={min_w:.6f}, "
            f"max principal angle={max_angle_deg:.4f} deg"
        )
        print(
            f"  selected mode 1 weights: w12={doublet[0]['w12']:.4f}, "
            f"w21={doublet[0]['w21']:.4f}"
        )

    df_modes = pd.DataFrame(all_mode_rows)
    df_seed = pd.DataFrame(seed_rows)

    modes_path = data_dir / "square_basis_selection_modes.csv"
    seed_path = data_dir / "square_basis_selection_summary.csv"
    df_modes.to_csv(modes_path, index=False)
    df_seed.to_csv(seed_path, index=False)

    pdf_path, png_path = make_figure(df_modes, df_seed, fig_dir)

    print("\nSaved outputs:")
    print(f"  {modes_path.relative_to(root)}")
    print(f"  {seed_path.relative_to(root)}")
    print(f"  {pdf_path.relative_to(root)}")
    print(f"  {png_path.relative_to(root)}")

    print("\nCompact summary:")
    print(df_seed[[
        "seed_id",
        "mean_doublet_subspace_weight",
        "min_doublet_subspace_weight",
        "max_principal_angle_deg",
        "basis_angle_mode1_deg_mod_180",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
