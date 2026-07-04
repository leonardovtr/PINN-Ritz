# Rayleigh-Ritz PINNs for quantum eigenspaces

Code and data for the manuscript

**Learning quantum eigenspaces under geometric symmetry breaking with Rayleigh--Ritz physics-informed neural networks**.

Preprint link: to be added after the arXiv submission.

## Scope

This repository contains the final scripts, data files, and manuscript figures used for the submitted version of the work. The examples study ordered excited states, degenerate eigenspaces, symmetry-split branches, residual-baseline controls, and modal tracking near avoided crossings in ideal quantum wells.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts run on CPU by default.

## Structure

```text
data/                  CSV files used by the manuscript figures and tables
figures/               final manuscript figures
quantum_dots/          analytical helper routines
scripts/               final training and reproduction scripts
paper_final_clean.tex  clean manuscript source
references.bib         bibliography
```

## Final Manuscript Figures

The repository includes only the final figure files used in the manuscript:

- `figures/Fig2_method_ablation.pdf`
- `figures/Fig3_results_summary.pdf`
- `figures/Fig4_splitting_deltas.pdf`
- `figures/Fig5_avoided_crossing_modal_tracking.pdf`
- `figures/Fig6_avoided_crossing_geometric.pdf`
- `figures/Fig7_residual_baseline_control.pdf`

PNG copies are included for convenient preview.

## Main Scripts

Core training and analysis scripts:

```bash
python scripts/train_pinn_1d_projected.py
python scripts/train_pinn_square_2d_subspace.py
python scripts/train_pinn_rectangle_2d_splitting.py
python scripts/train_pinn_disk_2d_subspace.py
python scripts/train_pinn_ellipse_2d_splitting.py
python scripts/train_pinn_avoided_crossing_carrier.py
python scripts/train_pinn_1d_residual_baseline.py --all-init-regimes
python scripts/train_pinn_avoided_crossing_geometric.py
```

Final figure and diagnostic scripts:

```bash
python scripts/make_main_results_and_splitting_figures.py --overwrite
python scripts/make_method_ablation_summary.py --root .
python scripts/make_avoided_crossing_modal_tracking.py
python scripts/ellipse_fd_grid_convergence.py
python scripts/analyze_square_basis_selection.py --epochs 1000
```

## Large Data Files

Three grid-level CSV files are stored with Git LFS:

- `data/final_ellipse_2d_modes_grid.csv`
- `data/pinn_ellipse_2d_symmetry_modes_grid.csv`
- `data/ellipse_symmetry_modes_grid.csv`

After cloning, install Git LFS and run:

```bash
git lfs pull
```

## Notes

- Eigenvalues are dimensionless for the Hamiltonian `H = -nabla^2`.
- Exact analytical spectra are used for validation in the 1D well, rectangle, square, and disk.
- The elliptic well is compared with an independent finite-difference reference.
- The avoided-crossing references are sine-basis spectral calculations for the coupled Hamiltonian.
