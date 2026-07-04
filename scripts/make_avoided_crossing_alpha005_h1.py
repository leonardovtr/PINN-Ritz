#!/usr/bin/env python3
"""Build the publication-style avoided-crossing figure for the alpha=0.05 run."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.7,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def panel_label(ax, text: str) -> None:
    ax.text(
        -0.12,
        1.04,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="bold",
        clip_on=False,
    )


def main() -> None:
    setup_style()
    summary = pd.read_csv(DATA / "pinn_avoided_crossing_2d_v6_summary_alpha005_h1.csv")
    char = pd.read_csv(DATA / "pinn_avoided_crossing_2d_v6_character_alpha005_h1.csv")
    lower = char[char["branch"] == "lower"].copy()

    eta = summary["eta"]
    fig = plt.figure(figsize=(7.05, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.05, 0.82], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(eta, summary["E12_uncoupled"], "--", color="0.68", lw=0.9, label=r"uncoupled $(1,2)$")
    ax.plot(eta, summary["E21_uncoupled"], ":", color="0.58", lw=1.2, label=r"uncoupled $(2,1)$")
    ax.plot(eta, summary["E2_ref"], "-", color="0.08", lw=1.15, label="reference")
    ax.plot(eta, summary["E3_ref"], "-", color="0.08", lw=1.15)
    ax.plot(eta, summary["E2_pinn"], "o", mfc="white", mec="0.08", mew=0.75, ms=3.0,
            label=r"RR-PINN, $\alpha=0.05$")
    ax.plot(eta, summary["E3_pinn"], "s", mfc="white", mec="0.08", mew=0.75, ms=3.0)
    ax.set_xlabel(r"$\eta=L_y/L_x$")
    ax.set_ylabel(r"$\epsilon$")
    ax.set_xlim(0.81, 1.19)
    ax.set_ylim(36.3, 73.0)
    panel_label(ax, "(a)")
    ax.legend(loc="upper right", handlelength=2.1, borderpad=0.1, labelspacing=0.35)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(lower["eta"], lower["w12_ref"], "-", color="0.08", lw=1.15, label=r"ref. $w_{12}$")
    ax.plot(lower["eta"], lower["w21_ref"], "--", color="0.08", lw=1.15, label=r"ref. $w_{21}$")
    ax.plot(lower["eta"], lower["w12_pinn"], "o", mfc="white", mec="0.08", mew=0.75, ms=3.0,
            label=r"PINN $w_{12}$")
    ax.plot(lower["eta"], lower["w21_pinn"], "s", mfc="white", mec="0.08", mew=0.75, ms=3.0,
            label=r"PINN $w_{21}$")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel("lower-branch weight")
    ax.set_xlim(0.81, 1.19)
    ax.set_ylim(-0.30, 1.04)
    ax.set_yticks([0.0, 0.5, 1.0])
    panel_label(ax, "(b)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.54, 0.02),
        ncol=2,
        handlelength=1.8,
        borderpad=0.1,
        labelspacing=0.22,
        columnspacing=0.8,
        fontsize=5.8,
    )

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(eta, summary["gap_ref"], "-", color="0.08", lw=1.15, label="reference")
    ax.plot(eta, summary["gap_pinn"], "o", mfc="white", mec="0.08", mew=0.75, ms=3.0,
            label="RR-PINN")
    ax.set_xlabel(r"$\eta$")
    ax.set_ylabel(r"$E_3-E_2$")
    ax.set_xlim(0.81, 1.19)
    ax.set_ylim(6.9, 17.4)
    panel_label(ax, "(c)")
    ax.legend(loc="upper right", handlelength=1.8, borderpad=0.1, labelspacing=0.35)

    out = FIG / "Fig5_avoided_crossing_alpha005_h1"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.018)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.018)
    plt.close(fig)
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
