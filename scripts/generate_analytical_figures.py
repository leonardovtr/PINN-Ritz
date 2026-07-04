#!/usr/bin/env python3
"""Generate analytical spectra and figures for the teaching project."""

from __future__ import annotations

import csv
import os
import sys
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
from matplotlib.ticker import FuncFormatter

from quantum_dots.analytical import (
    box2d_energy,
    box2d_spectrum,
    box2d_wavefunction,
    disk_real_mode,
    disk_spectrum,
    find_degenerate_groups,
    infinite_well_1d_energy,
)
from quantum_dots.plotting import COLORS, draw_domains, save_figure, set_style


FIGURES = ROOT / "figures"
DATA = ROOT / "data"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_1d_table() -> None:
    rows = [
        {"n": n, "epsilon": f"{infinite_well_1d_energy(n):.12f}"}
        for n in range(1, 7)
    ]
    write_rows(DATA / "well_1d_spectrum.csv", ["n", "epsilon"], rows)


def generate_square_spectrum() -> None:
    states = box2d_spectrum(max_n=5)
    rows = [
        {"rank": i, "nx": s.nx, "ny": s.ny, "epsilon": f"{s.epsilon:.12f}"}
        for i, s in enumerate(states, start=1)
    ]
    write_rows(DATA / "square_spectrum.csv", ["rank", "nx", "ny", "epsilon"], rows)

    groups = find_degenerate_groups(states)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    rank = 1
    tick_positions = []
    tick_labels = []
    for group in groups:
        color = COLORS["red"] if len(group) > 1 else COLORS["blue"]
        xs = list(range(rank, rank + len(group)))
        ys = [state.epsilon for state in group]
        ax.scatter(xs, ys, s=36, color=color, zorder=3)
        if len(group) > 1:
            ax.hlines(group[0].epsilon, xs[0] - 0.35, xs[-1] + 0.35, color=color, lw=1.2)
            label = ", ".join(f"({s.nx},{s.ny})" for s in group)
            ax.annotate(
                label,
                xy=(np.mean(xs), group[0].epsilon),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )
        for state, x in zip(group, xs):
            tick_positions.append(x)
            tick_labels.append(f"{state.nx},{state.ny}")
        rank += len(group)

    ax.set_title("Square box: symmetry produces repeated energies")
    ax.set_xlabel("state ordered by energy")
    ax.set_ylabel(r"$\epsilon/\pi^2$")
    ax.set_xticks(tick_positions[:18])
    ax.set_xticklabels(tick_labels[:18], rotation=45, ha="right")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / np.pi**2:.0f}"))
    ax.grid(color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.text(
        0.98,
        0.05,
        "red: degenerate multiplets",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["red"],
        fontsize=9,
    )
    save_figure(fig, FIGURES / "square_spectrum_degeneracy.png")


def generate_rectangle_splitting() -> None:
    etas = np.linspace(0.65, 1.45, 161)
    rows = []
    e12 = []
    e21 = []
    for eta in etas:
        energy_12 = box2d_energy(1, 2, lx=1.0, ly=eta)
        energy_21 = box2d_energy(2, 1, lx=1.0, ly=eta)
        e12.append(energy_12)
        e21.append(energy_21)
        rows.append(
            {
                "eta_Ly_over_Lx": f"{eta:.6f}",
                "epsilon_1_2": f"{energy_12:.12f}",
                "epsilon_2_1": f"{energy_21:.12f}",
                "splitting": f"{abs(energy_21 - energy_12):.12f}",
            }
        )
    write_rows(
        DATA / "rectangle_splitting.csv",
        ["eta_Ly_over_Lx", "epsilon_1_2", "epsilon_2_1", "splitting"],
        rows,
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.plot(etas, np.array(e12) / np.pi**2, color=COLORS["blue"], lw=2, label=r"$(1,2)$")
    ax.plot(etas, np.array(e21) / np.pi**2, color=COLORS["red"], lw=2, label=r"$(2,1)$")
    ax.axvline(1.0, color=COLORS["ink"], lw=1, ls="--")
    ax.annotate(
        "square\nsame energy",
        xy=(1.0, 5.0),
        xytext=(1.06, 6.8),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": COLORS["ink"]},
        fontsize=9,
    )
    ax.set_title(r"Rectangle: geometric deformation lifts $E_{1,2}=E_{2,1}$")
    ax.set_xlabel(r"$\eta=L_y/L_x$")
    ax.set_ylabel(r"$\epsilon/\pi^2$")
    ax.grid(color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.legend()
    save_figure(fig, FIGURES / "rectangle_splitting.png")


def generate_square_modes() -> None:
    grid = np.linspace(0, 1, 240)
    x, y = np.meshgrid(grid, grid, indexing="xy")
    psi_12 = box2d_wavefunction(x, y, 1, 2)
    psi_21 = box2d_wavefunction(x, y, 2, 1)
    combo = (psi_12 + psi_21) / np.sqrt(2)
    fields = [
        (psi_12, r"$\psi_{1,2}$"),
        (psi_21, r"$\psi_{2,1}$"),
        (combo, r"$(\psi_{1,2}+\psi_{2,1})/\sqrt{2}$"),
        (combo**2, r"density of linear combination"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        if "density" in title:
            image = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
        else:
            vmax = np.max(np.abs(field))
            image = ax.imshow(
                field,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle("Degenerate square-box subspace: the network may learn any valid mixture")
    save_figure(fig, FIGURES / "square_modes_linear_combination.png")


def generate_disk_figures() -> None:
    states = disk_spectrum(max_m=4, max_radial=3)
    rows = [
        {
            "rank": i,
            "m": s.m,
            "radial_index": s.radial_index,
            "bessel_zero": f"{s.zero:.12f}",
            "epsilon": f"{s.epsilon:.12f}",
            "degeneracy": s.degeneracy,
        }
        for i, s in enumerate(states, start=1)
    ]
    write_rows(
        DATA / "disk_spectrum.csv",
        ["rank", "m", "radial_index", "bessel_zero", "epsilon", "degeneracy"],
        rows,
    )

    grid = np.linspace(-1, 1, 260)
    x, y = np.meshgrid(grid, grid, indexing="xy")
    fields = [
        (disk_real_mode(x, y, 0, 1), r"$m=0,n=1$"),
        (disk_real_mode(x, y, 1, 1, "cos"), r"$m=1,n=1,\cos\theta$"),
        (disk_real_mode(x, y, 1, 1, "sin"), r"$m=1,n=1,\sin\theta$"),
        (disk_real_mode(x, y, 2, 1, "cos"), r"$m=2,n=1,\cos2\theta$"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.8), constrained_layout=True)
    for ax, (field, title) in zip(axes, fields):
        vmax = np.nanmax(np.abs(field))
        image = ax.imshow(
            field,
            origin="lower",
            extent=(-1, 1, -1, 1),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Disk modes: angular symmetry gives paired real modes for m > 0")
    save_figure(fig, FIGURES / "disk_bessel_modes.png")

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for i, state in enumerate(states[:12], start=1):
        color = COLORS["blue"] if state.m == 0 else COLORS["red"]
        marker = "o" if state.m == 0 else "s"
        ax.scatter(i, state.epsilon, s=42, color=color, marker=marker, zorder=3)
        label = f"m={state.m}, n={state.radial_index}"
        ax.annotate(label, (i, state.epsilon), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        if state.degeneracy == 2:
            ax.hlines(state.epsilon, i - 0.28, i + 0.28, color=color, lw=1.2)
    ax.set_title(r"Disk: $m>0$ levels are twofold degenerate")
    ax.set_xlabel("state ordered by energy")
    ax.set_ylabel(r"$\epsilon=\alpha_{m,n}^2$")
    ax.grid(color=COLORS["grid"], lw=0.6, alpha=0.8)
    ax.text(
        0.98,
        0.05,
        "red: cos/sin angular pair",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=COLORS["red"],
        fontsize=9,
    )
    save_figure(fig, FIGURES / "disk_spectrum_degeneracy.png")


def generate_domain_figure() -> None:
    draw_domains(FIGURES / "domains.png")


def main() -> None:
    set_style()
    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    generate_domain_figure()
    generate_1d_table()
    generate_square_spectrum()
    generate_rectangle_splitting()
    generate_square_modes()
    generate_disk_figures()
    print(f"Wrote figures to {FIGURES}")
    print(f"Wrote data to {DATA}")


if __name__ == "__main__":
    main()
