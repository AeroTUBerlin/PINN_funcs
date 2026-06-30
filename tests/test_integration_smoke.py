import os

import numpy as np

from PINN_funcs.data_handling import DataHandler
from PINN_funcs.PINNs import PINN_tf
from PINN_funcs.training import train_model


def _write_smoke_bundle(path):
    rng = np.random.default_rng(42)

    X_data = rng.random((16, 2), dtype=np.float32)
    U_data = rng.normal(loc=0.0, scale=0.1, size=(16, 5)).astype(np.float32)

    X_val = rng.random((8, 2), dtype=np.float32)
    val_u = np.linspace(0.1, 0.8, 8, dtype=np.float32)
    val_v = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
    val_p = np.linspace(0.9, 1.1, 8, dtype=np.float32)
    val_r = np.linspace(0.95, 1.05, 8, dtype=np.float32)
    val_E = np.linspace(1.5, 2.0, 8, dtype=np.float32)
    U_val = {
        "u": val_u,
        "v": val_v,
        "p": val_p,
        "r": val_r,
        "E": val_E,
    }

    X_BC_inlet = np.stack([np.zeros(6, dtype=np.float32), np.linspace(0.0, 1.0, 6, dtype=np.float32)], axis=1)
    U_BC_inlet = np.zeros((6, 5), dtype=np.float32)

    X_BC = {"inlet": X_BC_inlet}
    U_BC = {"inlet": U_BC_inlet}

    np.savez(
        path,
        X_data=X_data,
        U_data=U_data,
        X_val=X_val,
        U_val=U_val,
        X_BC=X_BC,
        U_BC=U_BC,
    )


def test_smoke_end_to_end_training_and_eval(tmp_path):
    bundle_path = tmp_path / "smoke_bundle.npz"
    _write_smoke_bundle(bundle_path)

    params_data = {
        "name": "smoke_case",
        "bundle_path": str(bundle_path),
        'save_path': os.path.dirname(__file__)
    }

    params_training = {
        "val_vars": ["u", "v", "p", "r", "E"],
        "batch_size": 8,
        "NN_type": "get_model_norm",
        "loss_weights": {"lambda_data": 1.0, "lambda_BC": 0.1, "lambda_PDE": 0.01},
        "lr_schedule": "constant",
        "lr_start": 1e-3,
        "optimizer": "adam",
        "steps_per_execution": 1,
        "jit_compile": False,
        "auto_scale_loss": False,
        "epochs": [1, 1],
    }

    params_nn = {
        "input_shape": (2,),
        "output_shape": 5,
        "n_layers": 1,
        "n_neurons": 8,
        "activation": "tanh",
        "dtype": "float32",
    }

    handler = DataHandler(params_data, params_training, params_nn, dtype="float32")
    dataset = handler.prepare_training(residual=True)
    model = handler.prepare_NN(PINN_tf)

    history = train_model(
        model,
        dataset,
        callbacks=[],
        epochs=1,
        iterations=1,
        resample=False,
    )

    metrics = model.val_step(model.validation_data, get_dict=False)

    assert "training_time" in history
    assert "mae_u" in metrics
    assert float(metrics["mae_u"]) >= 0.0
