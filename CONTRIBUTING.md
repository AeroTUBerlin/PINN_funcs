# Contributing

## Setup

```bash
git clone https://github.com/AeroTUBerlin/PINN_funcs.git
cd PINN_funcs
python -m pip install -e .[dev]
pre-commit install
```

## Workflow

- Run `pytest -q` and `ruff check .` before opening a PR; both also run in CI.
- Keep reusable, backend-agnostic logic in `PINN_funcs/`; keep case-specific scripts and configs in `examples/`.

## Tests

Tests synthesize their own small `.npz` bundles (see `tests/test_integration_smoke.py`), so no
external data download is required to run the suite.
