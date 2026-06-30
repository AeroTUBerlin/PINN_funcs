# PINN_funcs

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![CI](https://github.com/AeroTUBerlin/PINN_funcs/actions/workflows/ci.yml/badge.svg)](https://github.com/AeroTUBerlin/PINN_funcs/actions/workflows/ci.yml)

Utility package for **Physics-Informed Neural Networks (PINNs)** in experimental aerodynamics research.

This repository is intended to:
- provide a practical starting point for PINN research on inverse problems in experimental aerodynamics,
- reproduce results for the Rhombus Euler case from the associated article

## Scope

- **Backend:** TensorFlow / Keras.
- **Distribution:** GitHub repository (no PyPI release planned at this stage).

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

## Data

The Rhombus Euler example can be fetched from the project's GitHub Release using:

```bash
python helpers/fetch_data.py
```

This downloads `examples/data/rhombus/rhombus_euler_bundle_v1.npz`.

## Reproduce the Rhombus Euler case (paper)

Run all three configurations from the article:

```bash
python examples/Rhombus/run_rhombus_repro.py
```

This trains `rhombus_euler_paper_baseline`, `rhombus_euler_paper_PIQS_off`, and
`rhombus_euler_paper_PIQS_on` (configs in `examples/configs/`), writing trained models, plots,
and metrics under `examples/Rhombus/results/<case_name>/<timestamp>/`.

For a single, hand-editable run (hyperparameters inline rather than loaded from a config file):

```bash
python examples/Rhombus/Rhombus_Euler_rxy_cflearn.py
```

## Notes for new problems

- Use editable installs (`pip install -e .`) to iterate quickly while preserving package imports.
- Keep case-specific logic inside `examples/`; keep reusable pieces in `PINN_funcs/`.
- Start from the Rhombus example to understand the end-to-end workflow for density-gradient assimilation.
- See `docs/architecture.md` for a module-by-module overview.

## Public API guidance

Prefer explicit module imports in new code, e.g.:

- `from PINN_funcs.data_handling import DataHandler`
- `from PINN_funcs.training import train_model`

Top-level `import PINN_funcs` re-exports convenience symbols, but module-level imports are
clearer and more stable long-term.

## Development

```bash
python -m pip install -e .[dev]
pytest -q
ruff check .
```

## License

This project is licensed under **GNU GPL v3.0**. See `LICENSE`.

## Citation

If you use this software in academic work, please cite it using `CITATION.cff`.
