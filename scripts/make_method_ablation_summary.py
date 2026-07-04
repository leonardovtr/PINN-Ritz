#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_method_ablation_summary.py

Create a publication-style methodological ablation/failure-mode summary for the
quantum-domain PINN/Rayleigh--Ritz project.

Recommended location:
    PINN/scripts/make_method_ablation_summary.py

Run from PINN/scripts:
    python make_method_ablation_summary.py

Run from PINN root:
    python scripts/make_method_ablation_summary.py

Outputs:
    data/method_ablation_summary.csv
    data/method_physics_summary.csv
    figures/Fig_method_ablation_matrix.png/pdf
    figures/Fig_method_hierarchy.png/pdf
    figures/Fig_physics_chain_summary.png/pdf

The script is defensive: if a CSV is missing, it marks the corresponding
entry as missing rather than crashing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------

def setup_publication_style(usetex: bool = False) -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "text.usetex": bool(usetex),
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.75,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def save_figure(fig: plt.Figure, path_no_ext: Path) -> None:
    path_no_ext.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_no_ext.with_suffix(".png"), dpi=600)
    fig.savefig(path_no_ext.with_suffix(".pdf"))
    plt.close(fig)


# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def project_root_from_script() -> Path:
    here = Path(__file__).resolve()
    return here.parent.parent if here.parent.name == "scripts" else Path.cwd().resolve()


def first_existing(root: Path, candidates: List[str]) -> Optional[Path]:
    for rel in candidates:
        p = root / rel
        if p.exists():
            return p
    return None


def read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    return df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]


def status_from_error(err_percent: float, good: float, partial: float) -> str:
    if not np.isfinite(err_percent):
        return "missing"
    if err_percent <= good:
        return "success"
    if err_percent <= partial:
        return "partial"
    return "failure"


def score_from_status(status: str) -> int:
    return {
        "failure": 0,
        "partial": 1,
        "success": 2,
        "missing": -1,
        "not_applicable": -2,
    }.get(status, -1)


# -----------------------------------------------------------------------------
# Extract final physics metrics from CSVs
# -----------------------------------------------------------------------------

def extract_1d(root: Path) -> Tuple[List[dict], Optional[str]]:
    path = first_existing(root, [
        "data/pinn_1d_projected_summary.csv",
        "data/pinn_1d_summary.csv",
    ])
    df = read_csv(path)
    if df is None:
        return [], None

    err_col = next((c for c in df.columns if "relative" in c.lower() and "error" in c.lower()), None)
    maxerr = np.nan
    if err_col:
        vals = pd.to_numeric(df[err_col], errors="coerce").to_numpy()
        maxerr = float(np.nanmax(vals))
        if maxerr < 1e-2:  # probably fraction
            maxerr *= 100.0

    node_ok = np.nan
    if "nodes" in df.columns:
        parts = df["nodes"].astype(str).str.split("/", expand=True)
        if parts.shape[1] >= 2:
            node_ok = float((pd.to_numeric(parts[0]) == pd.to_numeric(parts[1])).mean())
    elif {"nodes_found", "nodes_expected"}.issubset(df.columns):
        node_ok = float((df["nodes_found"] == df["nodes_expected"]).mean())

    rows = [{
        "case": "1D well",
        "quantity": "max relative error",
        "value": maxerr,
        "unit": "%",
        "status": status_from_error(maxerr, good=0.05, partial=1.0),
        "source": str(path.relative_to(root)),
        "note": "Projected Rayleigh PINN for excited states.",
    }]
    if np.isfinite(node_ok):
        rows.append({
            "case": "1D well",
            "quantity": "correct nodal count",
            "value": 100.0 * node_ok,
            "unit": "%",
            "status": "success" if node_ok == 1.0 else "partial",
            "source": str(path.relative_to(root)),
            "note": "State n should have n-1 internal nodes.",
        })
    return rows, str(path.relative_to(root))


def extract_square(root: Path) -> Tuple[List[dict], Optional[str]]:
    path = first_existing(root, [
        "data/pinn_square_2d_subspace_summary.csv",
        "data/final_square_2d_subspace_summary.csv",
        "data/pinn_square_2d_summary.csv",
    ])
    df = read_csv(path)
    if df is None:
        return [], None

    err_col = next((c for c in df.columns if "rel" in c.lower() and "err" in c.lower()), None)
    deg_col = next((c for c in df.columns if "deg" in c.lower() and "weight" in c.lower()), None)

    rows = []
    maxerr = np.nan
    if err_col:
        maxerr = float(np.nanmax(pd.to_numeric(df[err_col], errors="coerce")))
        if maxerr < 1e-2:
            maxerr *= 100.0
    rows.append({
        "case": "square",
        "quantity": "max relative error",
        "value": maxerr,
        "unit": "%",
        "status": status_from_error(maxerr, good=1.0, partial=5.0),
        "source": str(path.relative_to(root)),
        "note": "Square subspace calculation.",
    })

    if deg_col:
        w = pd.to_numeric(df.iloc[1:3][deg_col], errors="coerce")
        minw = float(np.nanmin(w))
        rows.append({
            "case": "square",
            "quantity": "min degenerate-subspace weight",
            "value": minw,
            "unit": "",
            "status": "success" if minw >= 0.95 else ("partial" if minw >= 0.80 else "failure"),
            "source": str(path.relative_to(root)),
            "note": "Weight in span{psi_12, psi_21}.",
        })
    return rows, str(path.relative_to(root))


def extract_rectangle(root: Path) -> Tuple[List[dict], Optional[str]]:
    path = first_existing(root, [
        "data/final_rectangle_2d_splitting.csv",
        "data/pinn_rectangle_2d_splitting.csv",
    ])
    df = read_csv(path)
    if df is None:
        return [], None

    rows = []
    req = ["E12_exact", "E12_PINN", "E21_exact", "E21_PINN"]
    if all(c in df.columns for c in req):
        err12 = 100 * np.abs(df["E12_PINN"] - df["E12_exact"]) / np.abs(df["E12_exact"])
        err21 = 100 * np.abs(df["E21_PINN"] - df["E21_exact"]) / np.abs(df["E21_exact"])
        maxerr = float(np.nanmax(np.r_[err12, err21]))
    else:
        maxerr = np.nan

    rows.append({
        "case": "rectangle",
        "quantity": "max branch error",
        "value": maxerr,
        "unit": "%",
        "status": status_from_error(maxerr, good=1.0, partial=5.0),
        "source": str(path.relative_to(root)),
        "note": "PINN points versus analytic rectangle branches.",
    })

    sign_status = "missing"
    if {"eta", "Delta_PINN"}.issubset(df.columns):
        below = df[df["eta"] < 1]["Delta_PINN"]
        above = df[df["eta"] > 1]["Delta_PINN"]
        near = df[np.isclose(df["eta"], 1)]["Delta_PINN"]
        ok = True
        if len(below): ok = ok and bool(np.nanmean(below) > 0)
        if len(above): ok = ok and bool(np.nanmean(above) < 0)
        if len(near): ok = ok and abs(float(near.iloc[0])) < 0.5
        sign_status = "success" if ok else "failure"
    rows.append({
        "case": "rectangle",
        "quantity": "splitting sign reversal",
        "value": 1.0 if sign_status == "success" else 0.0,
        "unit": "",
        "status": sign_status,
        "source": str(path.relative_to(root)),
        "note": "Delta changes sign across eta=1.",
    })
    return rows, str(path.relative_to(root))


def extract_disk(root: Path) -> Tuple[List[dict], Optional[str]]:
    path = first_existing(root, [
        "data/final_disk_2d_subspace_summary.csv",
        "data/pinn_disk_2d_subspace_summary.csv",
    ])
    df = read_csv(path)
    if df is None:
        return [], None

    err_col = next((c for c in df.columns if "rel" in c.lower() and "err" in c.lower()), None)
    w_col = next((c for c in df.columns if "m=1" in c.lower() or "m1" in c.lower()), None)
    rows = []
    maxerr = np.nan
    if err_col:
        maxerr = float(np.nanmax(pd.to_numeric(df[err_col], errors="coerce")))
        if maxerr < 1e-2:
            maxerr *= 100.0
    rows.append({
        "case": "disk",
        "quantity": "max relative error",
        "value": maxerr,
        "unit": "%",
        "status": status_from_error(maxerr, good=0.1, partial=2.0),
        "source": str(path.relative_to(root)),
        "note": "Disk Bessel benchmark.",
    })
    if w_col:
        weights = pd.to_numeric(df[w_col], errors="coerce").to_numpy()
        minw = float(np.nanmin(weights[1:3])) if len(weights) >= 3 else np.nan
        rows.append({
            "case": "disk",
            "quantity": "min m=1 doublet weight",
            "value": minw,
            "unit": "",
            "status": "success" if minw >= 0.95 else ("partial" if minw >= 0.80 else "failure"),
            "source": str(path.relative_to(root)),
            "note": "Weight of states 2 and 3 in the circular m=1 subspace.",
        })
    return rows, str(path.relative_to(root))


def extract_ellipse(root: Path) -> Tuple[List[dict], Optional[str]]:
    path = first_existing(root, [
        "data/final_ellipse_2d_splitting.csv",
        "data/ellipse_symmetry_splitting.csv",
    ])
    df = read_csv(path)
    if df is None:
        return [], None

    req = ["E_x_PINN", "E_y_PINN", "E2_FD", "E3_FD", "eta"]
    if all(c in df.columns for c in req):
        low_p = np.minimum(df["E_x_PINN"], df["E_y_PINN"])
        up_p = np.maximum(df["E_x_PINN"], df["E_y_PINN"])
        low_f = df["E2_FD"]
        up_f = df["E3_FD"]
        err_low = 100 * np.abs(low_p - low_f) / np.abs(low_f)
        err_up = 100 * np.abs(up_p - up_f) / np.abs(up_f)
        maxerr = float(np.nanmax(np.r_[err_low, err_up]))

        row1 = df[np.isclose(df["eta"], 1.0)]
        split_at_one = abs(float(row1["E_x_PINN"].iloc[0] - row1["E_y_PINN"].iloc[0])) if len(row1) else np.nan

        below = df[df["eta"] < 1]
        above = df[df["eta"] > 1]
        order_ok = True
        if len(below): order_ok = order_ok and bool(np.all(below["E_y_PINN"] > below["E_x_PINN"]))
        if len(above): order_ok = order_ok and bool(np.all(above["E_x_PINN"] > above["E_y_PINN"]))
    else:
        maxerr, split_at_one, order_ok = np.nan, np.nan, False

    rows = [
        {
            "case": "ellipse",
            "quantity": "max sorted-branch error vs FD",
            "value": maxerr,
            "unit": "%",
            "status": status_from_error(maxerr, good=3.0, partial=8.0),
            "source": str(path.relative_to(root)),
            "note": "Symmetry-adapted x/y sectors; independent FD validation.",
        },
        {
            "case": "ellipse",
            "quantity": "splitting at eta=1",
            "value": split_at_one,
            "unit": "",
            "status": "success" if np.isfinite(split_at_one) and split_at_one < 0.02 else ("partial" if np.isfinite(split_at_one) and split_at_one < 0.2 else "failure"),
            "source": str(path.relative_to(root)),
            "note": "Circular limit should recover the m=1 degeneracy.",
        },
        {
            "case": "ellipse",
            "quantity": "branch ordering",
            "value": 1.0 if order_ok else 0.0,
            "unit": "",
            "status": "success" if order_ok else "failure",
            "source": str(path.relative_to(root)),
            "note": "eta<1 gives Ey>Ex; eta>1 gives Ex>Ey.",
        },
    ]
    return rows, str(path.relative_to(root))


# -----------------------------------------------------------------------------
# Curated ablation matrix
# -----------------------------------------------------------------------------

def build_method_entries(sources: Dict[str, Optional[str]]) -> pd.DataFrame:
    rows = []
    def add(case: str, method: str, status: str, evidence: str, source_key: str = ""):
        rows.append({
            "case": case,
            "method": method,
            "score": score_from_status(status),
            "label": status,
            "evidence": evidence,
            "source": sources.get(source_key) or "",
        })

    # 1D
    add("1D excited states", "residual PINN", "partial", "Recovers the n=5 state only with an eigenvalue initialization already close to the target energy (20/20 seeds); a wide or misleading initialization collapses onto a different nodal count (<=1/20 seeds), so it is not a robust free excited-state search (see initialization control).")
    add("1D excited states", "orthogonality penalty", "partial", "Works for low states but failed for higher excitation before hard projection.")
    add("1D excited states", "orthogonal projection", "success", "Recovered n=1..5 with correct nodal counts.", "1d")
    add("1D excited states", "subspace Ritz", "not_applicable", "Not needed for final 1D benchmark.")
    add("1D excited states", "symmetry sectors", "not_applicable", "No geometric degeneracy in 1D.")

    # Square
    add("square degeneracy", "residual PINN", "failure", "State-by-state residual/Rayleigh formulations are ambiguous in a degenerate eigenspace.")
    add("square degeneracy", "orthogonality penalty", "partial", "Can separate functions but does not naturally identify the degenerate subspace.")
    add("square degeneracy", "orthogonal projection", "partial", "Projection helps ordering but treats a subspace as individual states.")
    add("square degeneracy", "subspace Ritz", "success", "Recovered span{psi_12, psi_21}.", "square")
    add("square degeneracy", "symmetry sectors", "not_applicable", "Subspace formulation is the natural representation.")

    # Rectangle
    add("rectangle splitting", "residual PINN", "missing", "Not used as final method.")
    add("rectangle splitting", "orthogonality penalty", "missing", "Not used as final method.")
    add("rectangle splitting", "orthogonal projection", "partial", "May track branches but is less natural near eta=1.")
    add("rectangle splitting", "subspace Ritz", "success", "Recovered analytic E12/E21 splitting and sign reversal.", "rectangle")
    add("rectangle splitting", "symmetry sectors", "not_applicable", "Branch tracking via overlaps/subspace is sufficient.")

    # Disk
    add("disk m=1 doublet", "residual PINN", "missing", "Not used as final method.")
    add("disk m=1 doublet", "orthogonality penalty", "partial", "Degenerate angular modes should be interpreted as an eigenspace.")
    add("disk m=1 doublet", "orthogonal projection", "partial", "Projection imposes an arbitrary basis choice in the degenerate doublet.")
    add("disk m=1 doublet", "subspace Ritz", "success", "Recovered the rotational m=1 doublet.", "disk")
    add("disk m=1 doublet", "symmetry sectors", "not_applicable", "Full rotational subspace is more natural for the disk.")

    # Ellipse
    add("ellipse splitting", "residual PINN", "missing", "Not used as final method.")
    add("ellipse splitting", "orthogonality penalty", "failure", "Unstable for the split doublet.")
    add("ellipse splitting", "orthogonal projection", "failure", "Generic projection/subspace attempts produced high-energy spurious branches.")
    add("ellipse splitting", "subspace Ritz", "partial", "Generic subspace becomes unstable after degeneracy is lifted.")
    add("ellipse splitting", "symmetry sectors", "success", "Recovered x-like/y-like splitting versus FD.", "ellipse")
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def make_method_matrix(entries: pd.DataFrame, fig_dir: Path) -> None:
    cases = ["1D excited states", "square degeneracy", "rectangle splitting", "disk m=1 doublet", "ellipse splitting"]
    methods = ["residual PINN", "orthogonality penalty", "orthogonal projection", "subspace Ritz", "symmetry sectors"]
    mat = entries.pivot(index="case", columns="method", values="score").reindex(cases)[methods].to_numpy(dtype=float)

    cmap = ListedColormap(["#f2f2f2", "#d9d9d9", "#d73027", "#fee08b", "#1a9850"])
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(7.05, 3.35))
    ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(cases)))
    ax.set_xticklabels(["residual\nPINN", "orthogonality\npenalty", "orthogonal\nprojection", "subspace\nRitz", "symmetry\nsectors"])
    ax.set_yticklabels(["1D excited\nstates", "square\ndegeneracy", "rectangle\nsplitting", "disk $m=1$\ndoublet", "ellipse\nsplitting"])
    ax.set_title("methodological hierarchy for degenerate quantum eigenspaces")

    symbols = {-2: "—", -1: "?", 0: "×", 1: "±", 2: "✓"}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = int(mat[i, j])
            color = "white" if val in [0, 2] else "black"
            ax.text(j, i, symbols[val], ha="center", va="center", fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(np.arange(-0.5, len(methods), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cases), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    handles = [
        Patch(facecolor="#1a9850", label="success"),
        Patch(facecolor="#fee08b", label="partial/unstable"),
        Patch(facecolor="#d73027", label="failure"),
        Patch(facecolor="#d9d9d9", label="not run/missing"),
        Patch(facecolor="#f2f2f2", edgecolor="0.7", label="not applicable"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=5, frameon=False, fontsize=7)
    save_figure(fig, fig_dir / "Fig_method_ablation_matrix")


def make_method_hierarchy(fig_dir: Path) -> None:
    levels = [
        ("Residual\nPINN", "PDE residual alone", "#d73027"),
        ("Penalty\northogonality", "soft constraints", "#fee08b"),
        ("Projection", "hard orthogonalization", "#91cf60"),
        ("Subspace\nRitz", "degenerate eigenspaces", "#1a9850"),
        ("Symmetry\nsectors", "split branches", "#1a9850"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 1.75))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(levels))
    y = 0.58
    for i, (title, subtitle, color) in enumerate(levels):
        ax.scatter(xs[i], y, s=850, color=color, edgecolor="0.2", linewidth=0.8, zorder=3)
        ax.text(xs[i], y, str(i+1), ha="center", va="center", fontsize=10, fontweight="bold", color="white" if i in [0, 3, 4] else "black")
        ax.text(xs[i], 0.24, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(xs[i], 0.07, subtitle, ha="center", va="center", fontsize=7, color="0.25")
        if i < len(levels) - 1:
            ax.annotate("", xy=(xs[i+1]-0.055, y), xytext=(xs[i]+0.055, y), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))
    ax.text(0.5, 0.93, "from naive residual fitting to symmetry-aware variational eigensolvers", ha="center", va="center", fontsize=9)
    save_figure(fig, fig_dir / "Fig_method_hierarchy")


def make_physics_chain(fig_dir: Path) -> None:
    stages = [
        ("1D well", "nodes and\nexcited states", r"$\epsilon_n=n^2\pi^2$"),
        ("square", "discrete\nsymmetry", r"$\epsilon_{12}=\epsilon_{21}$"),
        ("rectangle", "discrete\nsplitting", r"$L_x\neq L_y$"),
        ("disk", "rotational\ndoublet", r"$m=1$"),
        ("ellipse", "angular\nsplitting", r"$x$-/ $y$-like"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.05))
    ax.axis("off")
    xs = np.linspace(0.08, 0.92, len(stages))
    y = 0.58
    for i, (name, concept, formula) in enumerate(stages):
        ax.scatter(xs[i], y, s=720, color="#f7f7f7", edgecolor="#2c7fb8", linewidth=1.2)
        ax.text(xs[i], y + 0.01, name, ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(xs[i], 0.30, concept, ha="center", va="center", fontsize=7.5)
        ax.text(xs[i], 0.12, formula, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate("", xy=(xs[i+1]-0.055, y), xytext=(xs[i]+0.055, y), arrowprops=dict(arrowstyle="->", lw=1.0, color="0.25"))
    ax.text(0.5, 0.93, "spectral reorganization under geometric symmetry breaking", ha="center", va="center", fontsize=9)
    save_figure(fig, fig_dir / "Fig_physics_chain_summary")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Create method-ablation and physics-summary figures")
    p.add_argument("--root", type=str, default=None, help="Project root. Default inferred from script location.")
    p.add_argument("--usetex", action="store_true", help="Use LaTeX rendering if available.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    root = Path(args.root).resolve() if args.root else project_root_from_script()
    setup_publication_style(args.usetex)

    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    print("Method/failure-mode summary")
    print(f"  root: {root}")

    physics_rows: List[dict] = []
    sources: Dict[str, Optional[str]] = {}
    for key, fn in [
        ("1d", extract_1d),
        ("square", extract_square),
        ("rectangle", extract_rectangle),
        ("disk", extract_disk),
        ("ellipse", extract_ellipse),
    ]:
        rows, source = fn(root)
        physics_rows.extend(rows)
        sources[key] = source

    physics_df = pd.DataFrame(physics_rows)
    method_df = build_method_entries(sources)

    physics_path = data_dir / "method_physics_summary.csv"
    method_path = data_dir / "method_ablation_summary.csv"
    physics_df.to_csv(physics_path, index=False)
    method_df.to_csv(method_path, index=False)

    make_method_matrix(method_df, fig_dir)
    make_method_hierarchy(fig_dir)
    make_physics_chain(fig_dir)

    print("\nSources detected:")
    for key, src in sources.items():
        print(f"  {key:10s}: {src or 'missing'}")

    print("\nSaved outputs:")
    for p in [
        physics_path,
        method_path,
        fig_dir / "Fig_method_ablation_matrix.png",
        fig_dir / "Fig_method_ablation_matrix.pdf",
        fig_dir / "Fig_method_hierarchy.png",
        fig_dir / "Fig_method_hierarchy.pdf",
        fig_dir / "Fig_physics_chain_summary.png",
        fig_dir / "Fig_physics_chain_summary.pdf",
    ]:
        if p.exists():
            print(f"  {p.relative_to(root)}")

    if not physics_df.empty:
        print("\nPhysics metrics:")
        for _, row in physics_df.iterrows():
            val = row["value"]
            try:
                val_s = f"{float(val):.4g}"
            except Exception:
                val_s = str(val)
            print(f"  {row['case']:10s} | {row['quantity']:34s} = {val_s} {row['unit']} [{row['status']}]")

    print("\nInterpretation:")
    print("  The central methodological message is that degenerate quantum eigenspaces")
    print("  are not reliably recovered by naive state-by-state PINNs. Projection,")
    print("  subspace Rayleigh--Ritz training, and symmetry-adapted sectors are the")
    print("  stabilizing ingredients needed to follow geometry-induced degeneracy lifting.")


if __name__ == "__main__":
    main()
