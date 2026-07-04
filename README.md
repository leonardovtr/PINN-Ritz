# Rayleigh--Ritz PINNs for quantum eigenspaces

This repository contains the code, data, and figures used for the manuscript

**Learning quantum eigenspaces under geometric symmetry breaking with Rayleigh--Ritz physics-informed neural networks**.

The project studies ideal infinite-well quantum eigenproblems as controlled tests for neural eigensolvers. The examples cover ordered excited states, degenerate eigenspaces, symmetry-split branches, residual-baseline controls, and modal tracking near avoided crossings. Analytical spectra or independent numerical references are used only for validation, not as supervised training targets.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts are written for CPU execution by default. Some training runs are slow, especially the multi-seed ablations.

## Repository Structure

```text
data/                  CSV outputs used by the manuscript figures and tables
figures/               manuscript figures and supporting figures
quantum_dots/          analytical spectra and plotting helpers
scripts/               training, analysis, ablation, and figure scripts
paper_final_clean.tex  current clean manuscript
references.bib         bibliography used by the manuscript
```

## Main Reproduction Map

Run commands from the repository root.

| Manuscript item | Output file(s) | Script / command |
| --- | --- | --- |
| Table 1, hyperparameters | `paper_final_clean.tex` | Reported directly in the manuscript from the production settings. |
| Fig. 3, main spectral tests | `figures/Fig3_results_summary.pdf` | `python scripts/make_main_results_figure.py` |
| Fig. 4, signed splittings | `figures/Fig4_splitting_deltas.pdf` | `python scripts/make_main_results_and_splitting_figures.py --overwrite` |
| Table 2, benchmark summary | `data/pinn_1d_summary_n1_n5_projected_v2.csv`, `data/final_rectangle_2d_splitting.csv`, `data/final_disk_2d_subspace_summary.csv`, `data/final_ellipse_2d_sector_results.csv` | Produced by the corresponding training scripts listed below. |
| Table 3, statistical ablation | `data/ablation_native_summary.csv`, `data/ablation_native_summary_table.tex` | `python scripts/run_ablation_native_scripts.py --root . --seeds 20` |
| Fig. 7, residual initialization control | `figures/Fig7_residual_baseline_control.pdf` | `python scripts/train_pinn_1d_residual_baseline.py --all-init-regimes` |
| Fig. 2, formulation hierarchy | `figures/Fig2_method_ablation.pdf` | `python scripts/make_method_ablation_summary.py --root .` |
| Fig. 5, avoided crossing with neural corrections | `figures/Fig5_avoided_crossing_alpha005_h1.pdf` | `python scripts/make_avoided_crossing_alpha005_h1.py` |
| Fig. 6, geometric avoided crossing | `figures/Fig6_avoided_crossing_geometric.pdf` | `python scripts/train_pinn_avoided_crossing_geom_v1.py --epochs 120 --restarts 1 --nq 36 --nq-val 52 --nq-ref 72 --nmax-ref 10` |
| Ellipse FD convergence check | `data/ellipse_fd_grid_convergence.csv`, `figures/Fig_ellipse_fd_convergence.pdf` | `python scripts/ellipse_fd_grid_convergence.py` |
| Square doublet orientation test | `data/square_basis_selection_summary_v2.csv`, `figures/square_basis_selection_v2.pdf` | `python scripts/analyze_square_basis_selection_v2.py --epochs 1000` |

## Main Training Scripts

These scripts regenerate the principal numerical data used by the figures and tables:

```bash
python scripts/train_pinn_1d_projected_v2.py
python scripts/train_pinn_square_2d_subspace.py
python scripts/train_pinn_rectangle_2d_splitting.py
python scripts/train_pinn_disk_2d_subspace.py
python scripts/train_pinn_ellipse_2d_splitting.py
python scripts/train_pinn_avoided_crossing_2d_v6.py --root . --alpha 0.05 --lambda_corr 1e-2 --lambda_h1corr 100.0 --epochs 1200 --nq 64 --nmax_ref 10 --restarts 3 --device cpu
```

## Large Data Files

Three grid-level CSV files are about 31 MB each:

- `data/final_ellipse_2d_modes_grid.csv`
- `data/pinn_ellipse_2d_symmetry_modes_grid.csv`
- `data/pinn_ellipse_2d_symmetry_v2_modes_grid.csv`

Before publishing the repository, choose one of two routes:

1. keep these files with Git LFS; or
2. exclude them from Git and document regeneration through `python scripts/train_pinn_ellipse_2d_splitting.py`, keeping only the smaller summary CSVs in the repository.

They are not ignored in the current `.gitignore`; the choice is intentionally left open.

## Notes

- The manuscript uses dimensionless eigenvalues for the Hamiltonian `H = -nabla^2`.
- Exact analytical spectra are used for validation in the 1D well, rectangle, square, and disk.
- The elliptic well is validated against an independent finite-difference reference.
- The avoided-crossing references are sine-basis spectral calculations for the coupled Hamiltonian.
