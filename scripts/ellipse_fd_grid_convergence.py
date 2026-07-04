#!/usr/bin/env python3
"""Finite-difference grid-convergence check for the elliptic well reference."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh


ROOT = Path(__file__).resolve().parents[1]
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


def finite_difference_ellipse_eigs(a: float, b: float, n_grid: int, k: int = 5) -> np.ndarray:
    """Five-point Dirichlet finite-difference spectrum on an elliptic mask.

    Interior grid points satisfying (x/a)^2 + (y/b)^2 < 1 are retained. If a
    stencil neighbor lies outside the mask, its value is the imposed zero
    Dirichlet boundary value, so no off-diagonal entry is added.
    """
    if n_grid % 2 == 0:
        n_grid += 1

    xs = np.linspace(-a, a, n_grid)
    ys = np.linspace(-b, b, n_grid)
    hx = xs[1] - xs[0]
    hy = ys[1] - ys[0]

    index: dict[tuple[int, int], int] = {}
    pts: list[tuple[int, int]] = []
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            if (x / a) ** 2 + (y / b) ** 2 < 1.0:
                index[(ix, iy)] = len(pts)
                pts.append((ix, iy))

    if len(pts) <= k + 2:
        raise RuntimeError(f"too few interior points for n_grid={n_grid}")

    mat = lil_matrix((len(pts), len(pts)), dtype=float)
    diag = 2.0 / hx**2 + 2.0 / hy**2
    neighbors = [
        (-1, 0, -1.0 / hx**2),
        (1, 0, -1.0 / hx**2),
        (0, -1, -1.0 / hy**2),
        (0, 1, -1.0 / hy**2),
    ]

    for row, (ix, iy) in enumerate(pts):
        mat[row, row] = diag
        for dix, diy, coeff in neighbors:
            col = index.get((ix + dix, iy + diy))
            if col is not None:
                mat[row, col] = coeff

    vals = eigsh(mat.tocsr(), k=k, which="SM", return_eigenvectors=False, tol=1e-9)
    vals.sort()
    return vals


def ellipse_interior_count(a: float, b: float, n_grid: int) -> int:
    xs = np.linspace(-a, a, n_grid)
    ys = np.linspace(-b, b, n_grid)
    return int(sum((x / a) ** 2 + (y / b) ** 2 < 1.0 for y in ys for x in xs))


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for eta in args.etas:
        previous: dict[str, float] | None = None
        for grid_size in args.grid_sizes:
            vals = finite_difference_ellipse_eigs(a=1.0, b=eta, n_grid=grid_size, k=args.k)
            e2 = float(vals[1])
            e3 = float(vals[2])
            if previous is None:
                de2 = np.nan
                de3 = np.nan
                dmax = np.nan
            else:
                de2 = 100.0 * abs(e2 - previous["E2_FD"]) / abs(previous["E2_FD"])
                de3 = 100.0 * abs(e3 - previous["E3_FD"]) / abs(previous["E3_FD"])
                dmax = max(de2, de3)
            rows.append({
                "eta": eta,
                "grid_size": grid_size,
                "interior_points": ellipse_interior_count(1.0, eta, grid_size),
                "E2_FD": e2,
                "E3_FD": e3,
                "delta_E2_percent_vs_previous_grid": de2,
                "delta_E3_percent_vs_previous_grid": de3,
                "delta_percent_vs_previous_grid": dmax,
            })
            previous = {"E2_FD": e2, "E3_FD": e3}
            print(
                f"eta={eta:.2f}, grid={grid_size}: E2={e2:.8f}, E3={e3:.8f}, "
                f"delta={dmax if np.isfinite(dmax) else np.nan:.4g}%"
            )
    return pd.DataFrame(rows)


def make_figure(df: pd.DataFrame) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.25))
    styles = [
        ("0.08", "-", "o"),
        ("0.25", "--", "s"),
        ("0.42", ":", "^"),
        ("0.58", "-.", "D"),
        ("0.72", (0, (3, 1, 1, 1)), "x"),
    ]
    for idx, (eta, group) in enumerate(df.groupby("eta")):
        g = group.dropna(subset=["delta_percent_vs_previous_grid"])
        color, linestyle, marker = styles[idx % len(styles)]
        ax.plot(
            g["grid_size"],
            g["delta_percent_vs_previous_grid"],
            marker=marker,
            linestyle=linestyle,
            color=color,
            mfc="white" if marker != "x" else color,
            mec=color,
            mew=0.8,
            ms=3.5,
            lw=1.1,
            label=rf"$\eta={eta:.2f}$",
        )
    ax.set_xlabel("grid size")
    ax.set_ylabel("successive FD change (%)")
    ax.set_xticks(sorted(df["grid_size"].unique()))
    ax.set_yscale("log")
    ax.legend(ncol=2, handlelength=1.8, columnspacing=0.9)
    fig.tight_layout(pad=0.25)
    out = OUT_FIG / "Fig_ellipse_fd_convergence"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--etas", nargs="+", type=float, default=[0.80, 0.90, 1.00, 1.10, 1.20])
    p.add_argument("--grid-sizes", nargs="+", type=int, default=[101, 151, 201])
    p.add_argument("--k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    df = run(args)
    out_csv = OUT_DATA / "ellipse_fd_grid_convergence.csv"
    df.to_csv(out_csv, index=False)
    make_figure(df)
    print(f"\nSaved {out_csv}")
    print(f"Saved {OUT_FIG / 'Fig_ellipse_fd_convergence.pdf'}")
    print(f"Saved {OUT_FIG / 'Fig_ellipse_fd_convergence.png'}")


if __name__ == "__main__":
    main()
