"""Plotting helpers for the analytical quantum-dot figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, Rectangle


COLORS = {
    "ink": "#202124",
    "muted": "#5f6368",
    "blue": "#2864b4",
    "red": "#c7423d",
    "green": "#20815a",
    "gold": "#b98000",
    "grid": "#d8dde3",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.edgecolor": COLORS["ink"],
            "axes.linewidth": 0.8,
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_domains(path: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.2))
    names = ["1D well", "square", "rectangle", "disk", "ellipse"]

    for ax, name in zip(axes, names):
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.axis("off")

    axes[0].plot([-0.85, 0.85], [0, 0], color=COLORS["blue"], lw=3)
    axes[0].plot([-0.85, -0.85], [-0.2, 0.2], color=COLORS["ink"], lw=2)
    axes[0].plot([0.85, 0.85], [-0.2, 0.2], color=COLORS["ink"], lw=2)

    axes[1].add_patch(Rectangle((-0.7, -0.7), 1.4, 1.4, fill=False, lw=2, ec=COLORS["blue"]))
    axes[2].add_patch(Rectangle((-0.9, -0.55), 1.8, 1.1, fill=False, lw=2, ec=COLORS["red"]))
    axes[3].add_patch(Circle((0, 0), 0.75, fill=False, lw=2, ec=COLORS["green"]))
    axes[4].add_patch(Ellipse((0, 0), 1.8, 1.05, fill=False, lw=2, ec=COLORS["gold"]))

    save_figure(fig, path)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        color=COLORS["ink"],
    )


def symmetric_limits(values: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    vmax = np.nanmax(np.abs(values))
    vmax *= 1.0 + margin
    return -vmax, vmax
