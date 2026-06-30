# Architecture Overview

## High-level flow

1. `examples/` script defines the case and training parameters.
2. `DataHandler` in `PINN_funcs/data_handling.py` loads NPZ bundles, prepares train/residual/validation data.
3. Model builders in `PINN_funcs/models.py` construct Keras models.
4. PINN logic in `PINN_funcs/PINNs.py` defines training/test behavior and physics-aware losses.
5. PDE residual operators in `PINN_funcs/PDEs.py` provide equation constraints.
6. `PINN_funcs/training.py` runs fit loops with optional resampling/weight updates.
7. Callbacks/utilities/logging support evaluation and experiment tracking.

## Main modules

- `PINN_funcs/data_handling.py`: bundle loading, sampling, data preparation, optimizer/model setup.
- `PINN_funcs/models.py`: architecture factories (`vanilla`, `norm`, `fourier`, etc.).
- `PINN_funcs/PINNs.py`: base PINN classes and TF-specific train/test logic.
- `PINN_funcs/PDEs.py`: PDE operators and derived physical quantities.
- `PINN_funcs/training.py`: training orchestration loop.
- `PINN_funcs/callbacks.py`: custom logging and adaptive weighting callbacks.
- `PINN_funcs/optimizers.py`: custom optimizer implementations and LR schedules.
- `PINN_funcs/utils.py`: logging helpers, utility math, reproducibility helpers.

## Data contracts

NPZ bundles should include:

- `X_data`, `U_data`, `X_val`
- `X_BC__*` and `U_BC__*` pairs
- `val__*` arrays for validation variables

Validation and consistency checks are implemented in `load_case_bundle`.

## Public API guidance

Prefer importing from module paths directly in scripts, e.g.:

- `from PINN_funcs.data_handling import DataHandler`
- `from PINN_funcs.training import train_model`

Top-level `import PINN_funcs` re-exports many names for convenience but module-level imports are clearer and more stable for long-term maintenance.
