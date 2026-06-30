# PINN_funcs

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![CI](https://github.com/AeroTUBerlin/PINN_funcs/actions/workflows/ci.yml/badge.svg)](https://github.com/AeroTUBerlin/PINN_funcs/actions/workflows/ci.yml)

Utility package for **Physics-Informed Neural Networks (PINNs)** in experimental aerodynamics research using the TensorFlow / Keras backend.

This repository is intended to:
- provide a practical starting point for PINN research on inverse problems in experimental aerodynamics,
- enable reproduction of the results for the rhombus airfoil case from Rohlfs et al. 2026

## Installation

### Recommended (pinned, reproducible)

```bash
conda env create -f environment.yml
conda activate pinns_tf
python -m pip install -e .
```

### Lightweight / development

```bash
python -m pip install -e .[dev]
pre-commit install  # optional: run lint + tests on every local commit
```

### Tested stack

- Python 3.12
- TensorFlow 2.17.1
- Keras 3.13.2
- CUDA 12.3
- cuDNN 8.9.2

## Data

The data for the Rhombus example can be fetched from the project's Release using:

```bash
python helpers/fetch_data.py
```

This downloads `examples/data/rhombus/rhombus_euler_bundle_v1.npz`.

## Reproduce the Rhombus Airfoil case (paper)

Run all three configurations from the article (Sim. BOS; PIQS (no BC); PIQS):

```bash
python examples/Rhombus/run_rhombus_repro.py
```

Trained models, plots, and metrics are saved under `examples/Rhombus/results/<case_name>/<timestamp>/`.

For a single, hand-editable run (hyperparameters inline rather than loaded from a config file):

```bash
python examples/Rhombus/Rhombus_Euler_rxy_cflearn.py
```

## Notes for new problems

- Use editable installs (`pip install -e .`) to iterate quickly while preserving package imports.
- Keep case-specific logic inside `examples/`; keep reusable pieces in `PINN_funcs/`.
- Start from the Rhombus example to understand the end-to-end workflow for density-gradient assimilation.
- See `docs/architecture.md` for a module-by-module overview.

## Development

```bash
python -m pip install -e .[dev]
pytest -q
ruff check .
```

## License

This project is licensed under **GNU GPL v3.0**. See `LICENSE`.

## Citation

If you use this software, please cite our work:
```bash
Rohlfs, L., Weiss, J. 
Quantitative Schlieren with Physics-Informed Neural Networks.
Exp Fluids 67, 94 (2026).
https://doi.org/10.1007/s00348-026-04268-1
```
