#!/usr/bin/env python3
"""Square-doublet 50-seed analysis with non-overlapping restart seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_square_basis_selection as base  # noqa: E402
import train_pinn_square_2d_subspace as sq  # noqa: E402


OUT_DATA = ROOT / "data"
OUT_FIG = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=50)
    p.add_argument("--seed0", type=int, default=1000)
    p.add_argument("--restart-seed-stride", type=int, default=1000)
    p.add_argument("--states", type=int, default=3)
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
    p.add_argument("--log-every", type=int, default=1000000)
    p.add_argument("--check-every", type=int, default=100)
    return p.parse_args()


def make_base_args(args: argparse.Namespace, training_seed: int) -> argparse.Namespace:
    shim = argparse.Namespace(
        states=args.states,
        epochs=args.epochs,
        restarts=args.restarts,
        n_quad=args.n_quad,
        n_validate=args.n_validate,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        fourier_features=args.fourier_features,
        lr=args.lr,
        lbfgs_steps=args.lbfgs_steps,
        dtype=args.dtype,
        device=args.device,
        threads=args.threads,
        log_every=args.log_every,
        check_every=args.check_every,
    )
    return base.make_square_args(shim, training_seed)


def main() -> None:
    args = parse_args()
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    all_mode_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    print("Square basis selection v2")
    print(f"  nominal seeds: {args.seed0} ... {args.seed0 + args.seeds - 1}")
    print(f"  restart seed stride: {args.restart_seed_stride}")

    for k in range(args.seeds):
        nominal_seed = args.seed0 + k
        training_seed = nominal_seed * args.restart_seed_stride
        print(f"\n=== seed {k + 1}/{args.seeds}: nominal={nominal_seed}, training_seed={training_seed} ===")

        sq_args = make_base_args(args, training_seed)
        sq.set_seed(training_seed)
        model = sq.train(sq_args, device, dtype)
        rows_modes, doublet, _, svals, max_angle_deg, theta_mod_pi = base.compute_doublet_diagnostics(
            model, sq_args, device, dtype
        )

        selected_indices = [doublet[0]["mode_index"], doublet[1]["mode_index"]]
        for row in rows_modes:
            r = dict(row)
            r["seed_id"] = k
            r["seed"] = nominal_seed
            r["training_seed"] = training_seed
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
            "seed": nominal_seed,
            "training_seed": training_seed,
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
        print(f"  mean w_S={mean_w:.6f}, max angle={max_angle_deg:.4f} deg")
        pd.DataFrame(all_mode_rows).to_csv(OUT_DATA / "square_basis_selection_modes_v2.csv", index=False)
        pd.DataFrame(seed_rows).to_csv(OUT_DATA / "square_basis_selection_summary_v2.csv", index=False)

    df_modes = pd.DataFrame(all_mode_rows)
    df_seed = pd.DataFrame(seed_rows)
    modes_path = OUT_DATA / "square_basis_selection_modes_v2.csv"
    seed_path = OUT_DATA / "square_basis_selection_summary_v2.csv"
    df_modes.to_csv(modes_path, index=False)
    df_seed.to_csv(seed_path, index=False)
    base.make_figure(df_modes, df_seed, OUT_FIG)

    numeric_cols = [c for c in df_seed.columns if c not in ("seed_id", "seed", "training_seed")]
    unique_rows = len(df_seed[numeric_cols].drop_duplicates())
    print("\nSaved:")
    print(f"  {modes_path}")
    print(f"  {seed_path}")
    print(f"  unique numerical rows: {unique_rows}/{len(df_seed)}")


if __name__ == "__main__":
    main()
