#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ablation_native_scripts.py

Run representative ablation using the *native project scripts* that already
produced the validated manuscript results.

This script calls the
existing scripts with subprocess, captures their stdout, parses the reported
summaries, and writes statistical ablation tables.

Place in:
    PINN/scripts/run_ablation_native_scripts.py

Run quick test from project root:
    cd PINN
    python scripts/run_ablation_native_scripts.py --root . --seeds 2 --dry-run
    python scripts/run_ablation_native_scripts.py --root . --seeds 2

Run from scripts:
    cd PINN/scripts
    python run_ablation_native_scripts.py --root .. --seeds 2

Full run:
    python scripts/run_ablation_native_scripts.py --root . --seeds 20

Outputs:
    data/ablation_native_runs.csv
    data/ablation_native_summary.csv
    data/ablation_native_summary_table.tex
    data/ablation_native_logs/*.log
    figures/Fig5_ablation_native_success_rates.pdf
    figures/Fig5_ablation_native_success_rates.png

Important:
    If a native script does not expose a --seed argument, this wrapper will run
    it without --seed and record a warning in the CSV. For a real statistical
    ablation, each native training script should ideally accept --seed and call:

        import random, numpy as np, torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    If the scripts do not expose --seed yet, we can patch them next.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Case specification
# -----------------------------------------------------------------------------

@dataclass
class NativeCase:
    case: str
    method: str
    script: str
    args: List[str]
    parser: str
    note: str = ""


def native_cases() -> List[NativeCase]:
    """Representative ablation cases using existing scripts.

    Adjust args here if a native script uses slightly different flags.
    The wrapper auto-detects whether --seed is accepted.
    """
    return [
        NativeCase(
            case="1D n=5",
            method="baseline state-by-state",
            script="train_pinn_1d.py",
            args=["--states", "5"],
            parser="parse_1d",
            note="native baseline 1D script",
        ),
        NativeCase(
            case="1D n=5",
            method="projected Rayleigh",
            script="train_pinn_1d_projected_v2.py",
            args=["--states", "5"],
            parser="parse_1d",
            note="native projected 1D script",
        ),
        NativeCase(
            case="square doublet",
            method="subspace Ritz",
            script="train_pinn_square_2d_subspace.py",
            args=["--states", "4"],
            parser="parse_square",
            note="native square subspace script",
        ),
        NativeCase(
            case="disk m=1",
            method="subspace Ritz",
            script="train_pinn_disk_2d_subspace.py",
            args=[],
            parser="parse_disk",
            note="native disk subspace script",
        ),
        NativeCase(
            case="ellipse split",
            method="generic subspace",
            script="train_pinn_ellipse_2d_splitting.py",
            args=[],
            parser="parse_ellipse_generic",
            note="native generic ellipse script",
        ),
        NativeCase(
            case="ellipse split",
            method="symmetry sectors",
            script="train_pinn_ellipse_2d_symmetry_v2.py",
            args=[],
            parser="parse_ellipse_sectors",
            note="native symmetry-sector ellipse script",
        ),
    ]


# -----------------------------------------------------------------------------
# Style / reporting
# -----------------------------------------------------------------------------

def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8.0,
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.7,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
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


def relerr(pred: float, exact: float) -> float:
    return 100.0 * abs(pred - exact) / max(abs(exact), 1e-14)


def script_accepts_seed(script_path: Path) -> bool:
    try:
        text = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "--seed" in text or "add_argument(\"--seed\"" in text or "add_argument('--seed'" in text


def run_command(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    timeout: Optional[int],
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------

def parse_1d(stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    """Parse rows like:
       5      246.74011003     246.73645888       0.0015%      4/4
    """
    rows = []
    pat = re.compile(
        r"^\s*(?P<n>\d+)\s+"
        r"(?P<exact>[-+0-9.eE]+)\s+"
        r"(?P<pinn>[-+0-9.eE]+)\s+"
        r"(?P<err>[-+0-9.eE]+)%"
        r"(?:\s+(?P<nodes>\d+)\s*/\s*(?P<nodes_exp>\d+))?",
        re.MULTILINE,
    )
    for m in pat.finditer(stdout):
        rows.append({
            "n": int(m.group("n")),
            "exact": float(m.group("exact")),
            "pinn": float(m.group("pinn")),
            "err": float(m.group("err")),
            "nodes": int(m.group("nodes")) if m.group("nodes") is not None else np.nan,
            "nodes_exp": int(m.group("nodes_exp")) if m.group("nodes_exp") is not None else np.nan,
        })

    if not rows:
        return failure_metrics("could not parse 1D summary")

    # Use n=5 if available, otherwise highest n.
    rows = sorted(rows, key=lambda r: r["n"])
    target = [r for r in rows if r["n"] == 5]
    r = target[0] if target else rows[-1]

    nodes_ok = True
    if not np.isnan(r["nodes"]):
        nodes_ok = int(r["nodes"]) == int(r["nodes_exp"])
    success = (r["err"] < thresholds.success_1d_error) and nodes_ok
    return {
        "success": bool(success),
        "rel_error_percent": r["err"],
        "max_rel_error_percent": max(rr["err"] for rr in rows),
        "subspace_weight": np.nan,
        "parity_score": np.nan,
        "nodes": r["nodes"],
        "nodes_expected": r["nodes_exp"],
        "failure_mode": "stable" if success else ("wrong nodes" if not nodes_ok else "energy error"),
        "parsed_rows": len(rows),
    }


def parse_square(stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    """Parse square summary rows with rel.err and deg.weight.

    Example:
      2      psi_12       49.34802201       49.525719    0.360      0.90
    """
    rows = []
    pat = re.compile(
        r"^\s*(?P<i>\d+)\s+"
        r"(?P<label>\S+)\s+"
        r"(?P<exact>[-+0-9.eE]+)\s+"
        r"(?P<pinn>[-+0-9.eE]+)\s+"
        r"(?P<err>[-+0-9.eE]+)%?\s+"
        r"(?P<weight>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    for m in pat.finditer(stdout):
        rows.append({
            "i": int(m.group("i")),
            "label": m.group("label"),
            "exact": float(m.group("exact")),
            "pinn": float(m.group("pinn")),
            "err": float(m.group("err")),
            "weight": float(m.group("weight")),
        })

    if not rows:
        return failure_metrics("could not parse square summary")

    # Prefer states 2 and 3 as degenerate doublet if present.
    doublet = [r for r in rows if r["i"] in (2, 3)]
    use = doublet if len(doublet) >= 2 else rows
    mean_err = float(np.mean([r["err"] for r in use]))
    max_err = float(np.max([r["err"] for r in use]))
    mean_w = float(np.mean([r["weight"] for r in use]))
    success = (max_err < thresholds.success_square_error) and (mean_w > thresholds.success_subspace_weight)
    return {
        "success": bool(success),
        "rel_error_percent": mean_err,
        "max_rel_error_percent": max_err,
        "subspace_weight": mean_w,
        "parity_score": np.nan,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "failure_mode": "stable" if success else ("low doublet weight" if mean_w <= thresholds.success_subspace_weight else "energy error"),
        "parsed_rows": len(rows),
    }


def parse_disk(stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    """Parse disk summary rows:
      2  psi_11^c  14.68197064  14.68202623  0.0004%  1.0000
    """
    rows = []
    pat = re.compile(
        r"^\s*(?P<i>\d+)\s+"
        r"(?P<label>.+?)\s+"
        r"(?P<exact>[-+0-9.eE]+)\s+"
        r"(?P<pinn>[-+0-9.eE]+)\s+"
        r"(?P<err>[-+0-9.eE]+)%\s+"
        r"(?P<w>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    for m in pat.finditer(stdout):
        label = m.group("label").strip()
        # Avoid matching headers accidentally.
        if "psi" not in label and "\\psi" not in label and "$" not in label:
            continue
        rows.append({
            "i": int(m.group("i")),
            "label": label,
            "exact": float(m.group("exact")),
            "pinn": float(m.group("pinn")),
            "err": float(m.group("err")),
            "w": float(m.group("w")),
        })

    if not rows:
        return failure_metrics("could not parse disk summary")

    doublet = [r for r in rows if r["i"] in (2, 3)]
    use = doublet if len(doublet) >= 2 else rows
    mean_err = float(np.mean([r["err"] for r in use]))
    max_err = float(np.max([r["err"] for r in use]))
    min_w = float(np.min([r["w"] for r in use]))
    success = (max_err < thresholds.success_disk_error) and (min_w > thresholds.success_subspace_weight)
    return {
        "success": bool(success),
        "rel_error_percent": mean_err,
        "max_rel_error_percent": max_err,
        "subspace_weight": min_w,
        "parity_score": np.nan,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "failure_mode": "stable" if success else ("low m=1 weight" if min_w <= thresholds.success_subspace_weight else "energy error"),
        "parsed_rows": len(rows),
    }


def parse_ellipse_sectors(stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    """Parse symmetry-sector ellipse summary rows:
      0.80  E_x_PINN E_y_PINN Delta_PINN E2_FD E3_FD Delta_FD lower
    """
    rows = []
    pat = re.compile(
        r"^\s*(?P<eta>0\.\d+|1\.0+|1\.\d+)\s+"
        r"(?P<ex>[-+0-9.eE]+)\s+"
        r"(?P<ey>[-+0-9.eE]+)\s+"
        r"(?P<delta>[-+0-9.eE]+)\s+"
        r"(?P<fd2>[-+0-9.eE]+)\s+"
        r"(?P<fd3>[-+0-9.eE]+)\s+"
        r"(?P<dfd>[-+0-9.eE]+)\s+"
        r"(?P<lower>[xy])",
        re.MULTILINE,
    )
    for m in pat.finditer(stdout):
        rows.append({
            "eta": float(m.group("eta")),
            "ex": float(m.group("ex")),
            "ey": float(m.group("ey")),
            "fd2": float(m.group("fd2")),
            "fd3": float(m.group("fd3")),
            "lower": m.group("lower"),
        })
    if not rows:
        return failure_metrics("could not parse ellipse-sector summary")

    errors = []
    ordering = []
    for r in rows:
        pred = sorted([r["ex"], r["ey"]])
        exact = [r["fd2"], r["fd3"]]
        errors.extend([relerr(pred[0], exact[0]), relerr(pred[1], exact[1])])
        if r["eta"] < 1.0:
            ordering.append(r["ey"] > r["ex"])
        elif r["eta"] > 1.0:
            ordering.append(r["ey"] < r["ex"])
        else:
            ordering.append(abs(r["ey"] - r["ex"]) < thresholds.success_ellipse_degen_abs)

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    parity_score = float(np.mean(ordering))
    success = (mean_err < thresholds.success_ellipse_error_mean) and all(ordering)
    return {
        "success": bool(success),
        "rel_error_percent": mean_err,
        "max_rel_error_percent": max_err,
        "subspace_weight": np.nan,
        "parity_score": parity_score,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "failure_mode": "stable branch tracking" if success else "branch drift/order or energy error",
        "parsed_rows": len(rows),
    }


def parse_ellipse_generic(stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    """Parse generic ellipse summary rows:
      eta E2_PINN E3_PINN Delta_PINN E2_FD E3_FD Delta_FD
    """
    rows = []
    pat = re.compile(
        r"^\s*(?P<eta>0\.\d+|1\.0+|1\.\d+)\s+"
        r"(?P<e2>[-+0-9.eE]+)\s+"
        r"(?P<e3>[-+0-9.eE]+)\s+"
        r"(?P<delta>[-+0-9.eE]+)\s+"
        r"(?P<fd2>[-+0-9.eE]+)\s+"
        r"(?P<fd3>[-+0-9.eE]+)\s+"
        r"(?P<dfd>[-+0-9.eE]+)",
        re.MULTILINE,
    )
    for m in pat.finditer(stdout):
        rows.append({
            "eta": float(m.group("eta")),
            "e2": float(m.group("e2")),
            "e3": float(m.group("e3")),
            "fd2": float(m.group("fd2")),
            "fd3": float(m.group("fd3")),
        })

    if not rows:
        return failure_metrics("could not parse generic ellipse summary")

    errors = []
    for r in rows:
        pred = sorted([r["e2"], r["e3"]])
        exact = [r["fd2"], r["fd3"]]
        errors.extend([relerr(pred[0], exact[0]), relerr(pred[1], exact[1])])

    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))
    success = mean_err < thresholds.success_ellipse_error_mean
    return {
        "success": bool(success),
        "rel_error_percent": mean_err,
        "max_rel_error_percent": max_err,
        "subspace_weight": np.nan,
        "parity_score": 0.0,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "failure_mode": "stable" if success else "branch drift/generic subspace instability",
        "parsed_rows": len(rows),
    }


def failure_metrics(reason: str) -> Dict[str, object]:
    return {
        "success": False,
        "rel_error_percent": np.inf,
        "max_rel_error_percent": np.inf,
        "subspace_weight": np.nan,
        "parity_score": np.nan,
        "nodes": np.nan,
        "nodes_expected": np.nan,
        "failure_mode": reason,
        "parsed_rows": 0,
    }


def parse_by_name(parser_name: str, stdout: str, thresholds: argparse.Namespace) -> Dict[str, object]:
    parsers = {
        "parse_1d": parse_1d,
        "parse_square": parse_square,
        "parse_disk": parse_disk,
        "parse_ellipse_generic": parse_ellipse_generic,
        "parse_ellipse_sectors": parse_ellipse_sectors,
    }
    return parsers[parser_name](stdout, thresholds)


# -----------------------------------------------------------------------------
# Summary / plotting
# -----------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case, method), g in df.groupby(["case", "method"], sort=False):
        rows.append({
            "case": case,
            "method": method,
            "n_seeds": int(g["seed"].nunique()),
            "success_rate_percent": 100.0 * float(g["success"].mean()),
            "mean_rel_error_percent": float(pd.to_numeric(g["rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()),
            "std_rel_error_percent": float(pd.to_numeric(g["rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).std(ddof=1)),
            "max_rel_error_percent": float(pd.to_numeric(g["max_rel_error_percent"], errors="coerce").replace([np.inf, -np.inf], np.nan).max()),
            "mean_subspace_weight": float(pd.to_numeric(g["subspace_weight"], errors="coerce").mean()),
            "mean_parity_score": float(pd.to_numeric(g["parity_score"], errors="coerce").mean()),
            "mean_runtime_s": float(pd.to_numeric(g["runtime_s"], errors="coerce").mean()),
            "seed_argument_available": bool(g["seed_argument_available"].all()),
            "typical_failure_mode": g["failure_mode"].mode().iloc[0] if len(g["failure_mode"].mode()) else "",
        })
    return pd.DataFrame(rows)


def save_latex_table(summary: pd.DataFrame, out_path: Path) -> None:
    cols = [
        "case",
        "method",
        "n_seeds",
        "success_rate_percent",
        "mean_rel_error_percent",
        "max_rel_error_percent",
        "mean_subspace_weight",
        "mean_parity_score",
        "mean_runtime_s",
        "typical_failure_mode",
    ]
    s = summary[cols].copy()
    for c in ["success_rate_percent", "mean_rel_error_percent", "max_rel_error_percent",
              "mean_subspace_weight", "mean_parity_score", "mean_runtime_s"]:
        s[c] = s[c].map(lambda x: "--" if pd.isna(x) else f"{x:.3g}")
    out_path.write_text(s.to_latex(index=False, escape=False), encoding="utf-8")


def plot_success(summary: pd.DataFrame, fig_dir: Path) -> None:
    setup_style()
    labels = [f"{r.case}\n{r.method}" for _, r in summary.iterrows()]
    vals = summary["success_rate_percent"].fillna(0).to_numpy()
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(vals))
    ax.bar(x, vals, color="0.58", edgecolor="0.15", linewidth=0.7)
    ax.set_ylim(0, 105)
    ax.set_ylabel("success rate (%)")
    ax.set_title("Native-script failure-mode ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.axhline(100, color="0.15", lw=0.8)
    ax.axhline(50, color="0.65", ls="--", lw=0.6)
    fig.savefig(fig_dir / "Fig5_ablation_native_success_rates.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(fig_dir / "Fig5_ablation_native_success_rates.png", dpi=600, bbox_inches="tight", pad_inches=0.025)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=".")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--dry-run", action="store_true", help="Print commands but do not run them.")
    p.add_argument("--timeout", type=int, default=None, help="Per-run timeout in seconds.")
    p.add_argument("--continue-existing", action="store_true", help="Skip runs already present in CSV.")
    p.add_argument("--python", type=str, default=sys.executable)

    # Thresholds.
    p.add_argument("--success_1d_error", type=float, default=0.1)
    p.add_argument("--success_square_error", type=float, default=1.0)
    p.add_argument("--success_disk_error", type=float, default=0.2)
    p.add_argument("--success_subspace_weight", type=float, default=0.90)
    p.add_argument("--success_ellipse_error_mean", type=float, default=3.0)
    p.add_argument("--success_ellipse_degen_abs", type=float, default=0.5)
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    scripts_dir = root / "scripts"
    data_dir = root / "data"
    fig_dir = root / "figures"
    logs_dir = data_dir / "ablation_native_logs"
    data_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)
    logs_dir.mkdir(exist_ok=True, parents=True)

    out_runs = data_dir / "ablation_native_runs.csv"
    out_summary = data_dir / "ablation_native_summary.csv"
    out_latex = data_dir / "ablation_native_summary_table.tex"

    existing = pd.DataFrame()
    if args.continue_existing and out_runs.exists():
        existing = pd.read_csv(out_runs)

    rows: List[Dict[str, object]] = []
    if not existing.empty:
        rows = existing.to_dict("records")

    print("Native-script ablation")
    print(f"  root: {root}")
    print(f"  scripts_dir: {scripts_dir}")
    print(f"  seeds: {args.seeds}")
    print(f"  dry_run: {args.dry_run}")

    for seed in range(args.seeds):
        for spec in native_cases():
            script_path = scripts_dir / spec.script
            if not script_path.exists():
                print(f"WARNING: missing script {script_path}; skipping")
                continue

            if not existing.empty:
                mask = (
                    (existing["case"] == spec.case)
                    & (existing["method"] == spec.method)
                    & (existing["seed"] == seed)
                )
                if mask.any():
                    print(f"skip existing: {spec.case} | {spec.method} | seed={seed}")
                    continue

            accepts_seed = script_accepts_seed(script_path)
            cmd = [args.python, spec.script] + list(spec.args)
            if accepts_seed:
                cmd += ["--seed", str(seed)]

            env = os.environ.copy()
            env["PYTHONHASHSEED"] = str(seed)
            env["PINN_ABLATION_SEED"] = str(seed)

            log_name = f"{spec.case.replace(' ', '_').replace('=', '')}__{spec.method.replace(' ', '_')}__seed{seed}.log"
            log_path = logs_dir / log_name

            print("RUN:", " ".join(cmd), f"(cwd={scripts_dir})")
            if args.dry_run:
                continue

            t0 = time.time()
            try:
                cp = run_command(cmd, scripts_dir, env, args.timeout)
                stdout = cp.stdout
                returncode = cp.returncode
            except subprocess.TimeoutExpired as exc:
                stdout = (exc.stdout or "") + f"\nTIMEOUT after {args.timeout} s\n"
                returncode = -999

            runtime_s = time.time() - t0
            log_path.write_text(stdout, encoding="utf-8", errors="ignore")

            if returncode == 0:
                metrics = parse_by_name(spec.parser, stdout, args)
            else:
                metrics = failure_metrics(f"nonzero return code {returncode}")

            row = {
                "case": spec.case,
                "method": spec.method,
                "script": spec.script,
                "seed": seed,
                "seed_argument_available": bool(accepts_seed),
                "returncode": returncode,
                "runtime_s": runtime_s,
                "log_path": str(log_path.relative_to(root)),
                "note": spec.note,
            }
            row.update(metrics)
            rows.append(row)

            pd.DataFrame(rows).to_csv(out_runs, index=False)

    if args.dry_run:
        print("Dry run finished.")
        return

    runs = pd.DataFrame(rows)
    runs.to_csv(out_runs, index=False)
    summary = summarize(runs)
    summary.to_csv(out_summary, index=False)
    save_latex_table(summary, out_latex)
    plot_success(summary, fig_dir)

    print("\nSaved:")
    print(f"  {out_runs}")
    print(f"  {out_summary}")
    print(f"  {out_latex}")
    print(f"  {fig_dir / 'Fig5_ablation_native_success_rates.pdf'}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    if not summary["seed_argument_available"].all():
        print("\nWARNING:")
        print("At least one native script did not expose --seed, so repeated runs may not be statistically independent.")
        print("If needed, patch the native scripts to accept --seed and call random/np/torch seed setters.")


if __name__ == "__main__":
    main()
