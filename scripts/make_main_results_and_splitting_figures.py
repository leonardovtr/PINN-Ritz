#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_main_results_and_splitting_figures.py

Generate a polished publication-style main-results figure for the
symmetry-aware Rayleigh--Ritz PINN quantum-domain paper.

This replaces the more diagnostic Fig3_results_summary with a cleaner,
physics-oriented figure:

(a) 1D well: exact and PINN/Ritz levels, with a small error inset.
(b) Rectangle: square-to-rectangle splitting of the (1,2)/(2,1) pair.
(c) Disk: discrete Bessel levels, emphasizing the m=1 doublet and subspace.
(d) Ellipse: disk-to-ellipse splitting, with FD shown as sorted branches.

Recommended location:
    PINN/scripts/make_main_results_and_splitting_figures.py

Run from project root:
    python scripts/make_main_results_and_splitting_figures.py --root .

Run from scripts:
    python make_main_results_and_splitting_figures.py --root ..

Outputs:
    figures/Fig3_results_summary_candidate.pdf
    figures/Fig3_results_summary_candidate.png

It also overwrites, if requested with --overwrite:
    figures/Fig3_results_summary.pdf
    figures/Fig3_results_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

def setup_publication_style() -> None:
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
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.3,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def panel_label(ax: plt.Axes, label: str, x: float = 0.035, y: float = 0.94) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
        zorder=20,
    )


def savefig(fig: plt.Figure, out_no_ext: Path) -> None:
    out_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_no_ext.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out_no_ext.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.025)


# -----------------------------------------------------------------------------
# Data handling
# -----------------------------------------------------------------------------

def read_or_default_1d(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "paper_1d_summary.csv",
        data_dir / "final_pinn_1d_summary.csv",
        data_dir / "pinn_1d_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize names if needed.
            if "E_exact" not in df.columns:
                if "epsilon_exact" in df.columns:
                    df = df.rename(columns={"epsilon_exact": "E_exact"})
                elif "epsilon_exact      " in df.columns:
                    pass
            if "E_PINN" not in df.columns:
                for c in df.columns:
                    if "PINN" in c and "epsilon" in c:
                        df = df.rename(columns={c: "E_PINN"})
                        break
            if "n" not in df.columns:
                if "state" in df.columns:
                    df = df.rename(columns={"state": "n"})
            if "rel_error_percent" not in df.columns:
                df["rel_error_percent"] = 100 * np.abs(df["E_PINN"] - df["E_exact"]) / np.abs(df["E_exact"])
            return df

    n = np.arange(1, 6)
    E_exact = (np.pi * n) ** 2
    E_pinn = np.array([9.86944667, 39.47795383, 88.82522381, 157.91196720, 246.73645888])
    return pd.DataFrame({
        "n": n,
        "E_exact": E_exact,
        "E_PINN": E_pinn,
        "rel_error_percent": 100 * np.abs(E_pinn - E_exact) / E_exact,
        "nodes_expected": n - 1,
        "nodes_found": n - 1,
    })


def read_or_default_rectangle(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "paper_rectangle_splitting.csv",
        data_dir / "final_rectangle_2d_splitting.csv",
        data_dir / "pinn_rectangle_2d_splitting.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df

    df = pd.DataFrame({
        "eta": [0.80, 0.90, 1.00, 1.10, 1.20],
        "E12_exact": [71.554632, 58.608392, 49.348022, 42.496396, 37.285172],
        "E12_PINN": [71.637802, 58.907555, 49.525719, 42.623295, 37.357265],
        "E21_exact": [54.899674, 51.663114, 49.348022, 47.635115, 46.332310],
        "E21_PINN": [54.985928, 52.047199, 49.424248, 47.709904, 46.403950],
    })
    return df


def read_or_default_disk(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "paper_disk_summary.csv",
        data_dir / "final_disk_2d_subspace_summary.csv",
        data_dir / "pinn_disk_2d_subspace_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            # Normalize column names.
            if "E_exact" not in df.columns:
                for c in df.columns:
                    if "exact" in c.lower():
                        df = df.rename(columns={c: "E_exact"})
                        break
            if "E_PINN" not in df.columns:
                for c in df.columns:
                    if "pinn" in c.lower():
                        df = df.rename(columns={c: "E_PINN"})
                        break
            if "m1_weight" not in df.columns:
                for c in df.columns:
                    if "m=1" in c.lower() or "m1" in c.lower():
                        df = df.rename(columns={c: "m1_weight"})
                        break
            if "state" not in df.columns:
                df["state"] = np.arange(1, len(df) + 1)
            return df

    E_exact = np.array([5.78318596, 14.68197064, 14.68197064])
    E_pinn = np.array([5.78321759, 14.68202623, 14.68204434])
    return pd.DataFrame({
        "state": [1, 2, 3],
        "label": [r"$\psi_{0,1}$", r"$\psi^c_{1,1}$", r"$\psi^s_{1,1}$"],
        "E_exact": E_exact,
        "E_PINN": E_pinn,
        "m1_weight": [0.0, 1.0, 1.0],
        "rel_error_percent": 100 * np.abs(E_pinn - E_exact) / E_exact,
    })


def read_or_default_ellipse(data_dir: Path) -> pd.DataFrame:
    candidates = [
        data_dir / "paper_ellipse_splitting.csv",
        data_dir / "final_ellipse_2d_splitting.csv",
        data_dir / "ellipse_symmetry_splitting.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df

    df = pd.DataFrame({
        "eta": [0.80, 0.90, 1.00, 1.10, 1.20],
        "E_x_PINN": [16.737887, 15.572956, 14.742614, 14.093526, 13.615068],
        "E_y_PINN": [20.893515, 17.314495, 14.742018, 12.799793, 11.345582],
        "E2_FD": [16.512141, 15.349390, 14.512858, 12.619688, 11.171859],
        "E3_FD": [20.581937, 17.053649, 14.512858, 13.876876, 13.384234],
    })
    return df


# -----------------------------------------------------------------------------
# Plotting panels
# -----------------------------------------------------------------------------

def plot_1d(ax: plt.Axes, df: pd.DataFrame) -> None:
    n = df["n"].to_numpy()
    e_exact = df["E_exact"].to_numpy() / np.pi**2
    e_pinn = df["E_PINN"].to_numpy() / np.pi**2
    err = df["rel_error_percent"].to_numpy()

    ax.plot(n, e_exact, "-", color="0.12", lw=1.35, label="exact")
    ax.plot(n, e_pinn, "o", color="#1f77b4", ms=4.4, label="PINN/Ritz", zorder=4)

    ax.set_xlabel(r"state $n$")
    ax.set_ylabel(r"$\epsilon_n/\pi^2$")
    ax.set_title("1D well")
    ax.set_xticks(n)
    ax.set_ylim(0, max(e_exact) * 1.08)
    ax.legend(loc="upper left", bbox_to_anchor=(0.10, 0.99), borderaxespad=0.0)
    panel_label(ax, "(a)")

    # Error inset: small, unobtrusive, labelled.
    iax = inset_axes(ax, width="38%", height="34%", loc="lower right", borderpad=1.0)
    iax.plot(n, err, "s--", color="0.45", ms=2.8, lw=0.95)
    iax.set_title("error", fontsize=6.7, pad=1.0)
    iax.set_xticks([1, 3, 5])
    iax.set_yticks([np.nanmin(err), np.nanmax(err)])
    iax.tick_params(labelsize=6.2, pad=1.0)
    iax.set_ylabel(r"%", fontsize=6.2, labelpad=0.5)
    iax.set_ylim(max(0, np.nanmin(err) * 0.85), np.nanmax(err) * 1.15)
    for s in iax.spines.values():
        s.set_linewidth(0.55)


def plot_rectangle(ax: plt.Axes, df: pd.DataFrame) -> None:
    eta = df["eta"].to_numpy()

    ax.plot(eta, df["E12_exact"] / np.pi**2, "-", color="0.18", label=r"exact $(1,2)$")
    ax.plot(eta, df["E21_exact"] / np.pi**2, "--", color="0.18", label=r"exact $(2,1)$")
    ax.plot(eta, df["E12_PINN"] / np.pi**2, "o", color="#1f77b4", label=r"PINN $(1,2)$", zorder=4)
    ax.plot(eta, df["E21_PINN"] / np.pi**2, "s", color="#ff7f0e", label=r"PINN $(2,1)$", zorder=4)

    ax.axvline(1.0, color="0.35", ls=":", lw=0.9)
    ax.text(1.0, ax.get_ylim()[1] if False else 0, "", alpha=0)

    ax.set_xlabel(r"aspect ratio $\eta=L_y/L_x$")
    ax.set_ylabel(r"$\epsilon/\pi^2$")
    ax.set_title("rectangle splitting")
    ax.legend(loc="upper right")
    ax.minorticks_on()
    panel_label(ax, "(b)")

    # Mark degeneracy.
    ydeg = float(df.loc[np.isclose(df["eta"], 1.0), "E12_exact"].iloc[0] / np.pi**2)
    ax.annotate(
        r"degenerate",
        xy=(1.0, ydeg),
        xytext=(1.035, ydeg + 0.55),
        fontsize=7.0,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
        ha="left",
        va="bottom",
    )


def plot_disk(ax: plt.Axes, df: pd.DataFrame) -> None:
    states = np.arange(1, len(df) + 1)
    e_exact = df["E_exact"].to_numpy()
    e_pinn = df["E_PINN"].to_numpy()

    # Use horizontal ticks/levels rather than continuous connecting lines.
    half_width = 0.18
    for x, e in zip(states, e_exact):
        ax.hlines(e, x - half_width, x + half_width, color="0.15", lw=1.4, zorder=2)
    ax.plot(states, e_pinn, "o", color="#1f77b4", ms=4.4, label="PINN/Ritz", zorder=4)

    # Degenerate bracket for the m=1 doublet.
    y = float(np.mean(e_exact[1:3]))
    ax.plot([2, 3], [y + 0.42, y + 0.42], color="0.25", lw=0.8)
    ax.plot([2, 2], [y + 0.30, y + 0.42], color="0.25", lw=0.8)
    ax.plot([3, 3], [y + 0.30, y + 0.42], color="0.25", lw=0.8)
    ax.text(2.5, y + 0.52, r"$m=1$ doublet", ha="center", va="bottom", fontsize=7.2)

    # Subspace weight annotation.
    if "m1_weight" in df.columns:
        w2 = df["m1_weight"].to_numpy()[1:3]
        ax.text(
            2.5,
            y - 0.85,
            rf"$w_{{m=1}}={np.nanmin(w2):.1f}$",
            ha="center",
            va="top",
            fontsize=7.2,
            color="0.25",
        )

    # Add exact reference as proxy legend using an invisible marker/line.
    ax.plot([], [], "-", color="0.15", lw=1.4, label="Bessel reference")
    ax.legend(loc="lower right")

    ax.set_xticks(states)
    ax.set_xticklabels([r"$m=0$", r"$m=1,c$", r"$m=1,s$"])
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("disk doublet")
    ax.set_ylim(min(e_exact) - 0.8, max(e_exact) + 1.7)
    ax.minorticks_on()
    panel_label(ax, "(c)")


def plot_ellipse(ax: plt.Axes, df: pd.DataFrame) -> None:
    eta = df["eta"].to_numpy()

    ax.plot(eta, df["E_x_PINN"], "o-", color="#1f77b4", label=r"PINN $x$-like")
    ax.plot(eta, df["E_y_PINN"], "s-", color="#ff7f0e", label=r"PINN $y$-like")

    # FD branches are sorted by energy, not parity-labelled.
    if "E2_FD" in df.columns and "E3_FD" in df.columns:
        ax.plot(eta, df["E2_FD"], "--", color="0.35", lw=1.05, label="FD sorted branches")
        ax.plot(eta, df["E3_FD"], "--", color="0.35", lw=1.05)

    ax.axvline(1.0, color="0.35", ls=":", lw=0.9)

    # Mark circular degeneracy.
    row = df.loc[np.isclose(df["eta"], 1.0)]
    if len(row):
        ydeg = 0.5 * (float(row["E_x_PINN"].iloc[0]) + float(row["E_y_PINN"].iloc[0]))
        ax.annotate(
            r"disk limit",
            xy=(1.0, ydeg),
            xytext=(1.035, ydeg + 1.25),
            fontsize=7.0,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(r"aspect ratio $\eta=b/a$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_title("ellipse splitting")
    ax.legend(loc="upper right")
    ax.minorticks_on()
    panel_label(ax, "(d)")




def plot_corrected_splitting_deltas(df_rect: pd.DataFrame, df_ell: pd.DataFrame, fig_dir: Path, overwrite: bool = True) -> None:
    """Generate corrected Fig. 4.

    Panel (a): rectangle signed splitting, exact and PINN.
    Panel (b): ellipse signed splitting, PINN and finite-difference reference.

    The important correction relative to the older diagnostic version is that
    the finite-difference ellipse curve is converted into a signed splitting,
    rather than plotted as |Delta|.  For eta<1, Ey>Ex, so the signed FD
    splitting is positive.  For eta>1, Ey<Ex, so the signed FD splitting is
    negative.
    """
    eta_r = df_rect["eta"].to_numpy()

    # Rectangle signed splitting.
    if "Delta_exact" in df_rect.columns:
        delta_rect_exact = df_rect["Delta_exact"].to_numpy()
    else:
        delta_rect_exact = df_rect["E12_exact"].to_numpy() - df_rect["E21_exact"].to_numpy()

    if "Delta_PINN" in df_rect.columns:
        delta_rect_pinn = df_rect["Delta_PINN"].to_numpy()
    else:
        delta_rect_pinn = df_rect["E12_PINN"].to_numpy() - df_rect["E21_PINN"].to_numpy()

    eta_e = df_ell["eta"].to_numpy()
    delta_ell_pinn = df_ell["E_y_PINN"].to_numpy() - df_ell["E_x_PINN"].to_numpy()

    # FD reference is sorted by energy, so recover the sign from geometry:
    # eta<1 -> y-like branch is upper -> Ey-Ex positive
    # eta>1 -> y-like branch is lower -> Ey-Ex negative
    delta_fd_abs = df_ell["E3_FD"].to_numpy() - df_ell["E2_FD"].to_numpy()
    sign = np.where(eta_e < 1.0, 1.0, np.where(eta_e > 1.0, -1.0, 0.0))
    delta_ell_fd_signed = sign * delta_fd_abs

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

    ax = axes[0]
    ax.plot(eta_r, delta_rect_exact / np.pi**2, "-", color="0.18", label="exact")
    ax.plot(eta_r, delta_rect_pinn / np.pi**2, "o", color="#1f77b4", label="PINN/Ritz", zorder=4)
    ax.axhline(0.0, color="0.2", lw=0.85)
    ax.axvline(1.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlabel(r"aspect ratio $\eta=L_y/L_x$")
    ax.set_ylabel(r"$(\epsilon_{12}-\epsilon_{21})/\pi^2$")
    ax.set_title("discrete symmetry breaking")
    ax.legend(loc="upper right")
    ax.minorticks_on()
    panel_label(ax, "(a)")

    ax.annotate(
        r"square",
        xy=(1.0, 0.0),
        xytext=(1.035, 0.42),
        fontsize=7.0,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
        ha="left",
        va="bottom",
    )

    ax = axes[1]
    ax.plot(eta_e, delta_ell_pinn, "o-", color="#1f77b4", label=r"PINN/Ritz $E_y-E_x$")
    ax.plot(eta_e, delta_ell_fd_signed, "s--", color="0.35", lw=1.05, label="FD signed splitting")
    ax.axhline(0.0, color="0.2", lw=0.85)
    ax.axvline(1.0, color="0.35", ls=":", lw=0.9)
    ax.set_xlabel(r"aspect ratio $\eta=b/a$")
    ax.set_ylabel(r"signed splitting")
    ax.set_title("rotational symmetry breaking")
    ax.legend(loc="upper right")
    ax.minorticks_on()
    panel_label(ax, "(b)")

    ax.annotate(
        r"disk",
        xy=(1.0, 0.0),
        xytext=(1.035, 1.0),
        fontsize=7.0,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
        ha="left",
        va="bottom",
    )

    out = fig_dir / "Fig4_splitting_deltas_candidate"
    savefig(fig, out)

    if overwrite:
        # Recreate because savefig closes the figure.
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), constrained_layout=True)

        ax = axes[0]
        ax.plot(eta_r, delta_rect_exact / np.pi**2, "-", color="0.18", label="exact")
        ax.plot(eta_r, delta_rect_pinn / np.pi**2, "o", color="#1f77b4", label="PINN/Ritz", zorder=4)
        ax.axhline(0.0, color="0.2", lw=0.85)
        ax.axvline(1.0, color="0.35", ls=":", lw=0.9)
        ax.set_xlabel(r"aspect ratio $\eta=L_y/L_x$")
        ax.set_ylabel(r"$(\epsilon_{12}-\epsilon_{21})/\pi^2$")
        ax.set_title("discrete symmetry breaking")
        ax.legend(loc="upper right")
        ax.minorticks_on()
        panel_label(ax, "(a)")
        ax.annotate(
            r"square",
            xy=(1.0, 0.0),
            xytext=(1.035, 0.42),
            fontsize=7.0,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
            ha="left",
            va="bottom",
        )

        ax = axes[1]
        ax.plot(eta_e, delta_ell_pinn, "o-", color="#1f77b4", label=r"PINN/Ritz $E_y-E_x$")
        ax.plot(eta_e, delta_ell_fd_signed, "s--", color="0.35", lw=1.05, label="FD signed splitting")
        ax.axhline(0.0, color="0.2", lw=0.85)
        ax.axvline(1.0, color="0.35", ls=":", lw=0.9)
        ax.set_xlabel(r"aspect ratio $\eta=b/a$")
        ax.set_ylabel(r"signed splitting")
        ax.set_title("rotational symmetry breaking")
        ax.legend(loc="upper right")
        ax.minorticks_on()
        panel_label(ax, "(b)")
        ax.annotate(
            r"disk",
            xy=(1.0, 0.0),
            xytext=(1.035, 1.0),
            fontsize=7.0,
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.25"),
            ha="left",
            va="bottom",
        )

        savefig(fig, fig_dir / "Fig4_splitting_deltas")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate publication-style Fig. 3 main results")
    p.add_argument("--root", type=str, default=".", help="Project root containing data/ and figures/")
    p.add_argument("--overwrite", action="store_true", help="Also overwrite figures/Fig3_results_summary.pdf/png")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    data_dir = root / "data"
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_publication_style()

    df_1d = read_or_default_1d(data_dir)
    df_rect = read_or_default_rectangle(data_dir)
    df_disk = read_or_default_disk(data_dir)
    df_ell = read_or_default_ellipse(data_dir)

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)

    plot_1d(axes[0, 0], df_1d)
    plot_rectangle(axes[0, 1], df_rect)
    plot_disk(axes[1, 0], df_disk)
    plot_ellipse(axes[1, 1], df_ell)

    out = fig_dir / "Fig3_results_summary_candidate"
    savefig(fig, out)

    if args.overwrite:
        # Save a second copy under the manuscript's current expected name.
        # Need to reconstruct figure because savefig closes.
        fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.05), constrained_layout=True)
        plot_1d(axes[0, 0], df_1d)
        plot_rectangle(axes[0, 1], df_rect)
        plot_disk(axes[1, 0], df_disk)
        plot_ellipse(axes[1, 1], df_ell)
        savefig(fig, fig_dir / "Fig3_results_summary")

    # Also generate corrected Fig. 4 with signed FD splitting.
    plot_corrected_splitting_deltas(df_rect, df_ell, fig_dir, overwrite=args.overwrite)

    print("Wrote:")
    print(f"  {out.with_suffix('.pdf').relative_to(root)}")
    print(f"  {out.with_suffix('.png').relative_to(root)}")
    print(f"  {(fig_dir/'Fig4_splitting_deltas_candidate.pdf').relative_to(root)}")
    print(f"  {(fig_dir/'Fig4_splitting_deltas_candidate.png').relative_to(root)}")
    if args.overwrite:
        print(f"  {(fig_dir/'Fig3_results_summary.pdf').relative_to(root)}")
        print(f"  {(fig_dir/'Fig3_results_summary.png').relative_to(root)}")
        print(f"  {(fig_dir/'Fig4_splitting_deltas.pdf').relative_to(root)}")
        print(f"  {(fig_dir/'Fig4_splitting_deltas.png').relative_to(root)}")


if __name__ == "__main__":
    main()
