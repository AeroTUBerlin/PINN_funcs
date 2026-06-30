import numpy as np
from keras import ops

import PINN_funcs.data_handling as data_handling
from PINN_funcs.data_handling import DataHandler


class _LossWeightsStub:
    def __init__(self):
        self.last_update = None

    def update_state(self, values):
        self.last_update = values


class _ModelStub:
    def __init__(self):
        self.loss_weights = _LossWeightsStub()
        self.validation_data = None
        self.dtype = None
        self.bc_call = None
        self.compile_kwargs = None
        self.summary_called = False

    def set_BC(self, X_BC, U_BC):
        self.bc_call = (X_BC, U_BC)

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs

    def summary(self):
        self.summary_called = True


def test_prepare_training_builds_residual_and_updates_samples(monkeypatch):
    handler = DataHandler.__new__(DataHandler)
    handler.dtype = "float32"
    handler.params_training = {"batch_size": 4}
    handler.X_data = ops.convert_to_tensor(np.zeros((8, 2), dtype=np.float32))
    handler.U_data = ops.convert_to_tensor(np.zeros((8, 1), dtype=np.float32))
    handler.lb = np.array([0.0, 0.0], dtype=np.float32)
    handler.ub = np.array([1.0, 1.0], dtype=np.float32)
    handler.sdf_func = None
    handler.X_res = None

    residual_call = {}

    def fake_get_residual_sample(lb, ub, batches, batchsize, dim, dtype, sdf_func):
        residual_call["args"] = {
            "lb": lb,
            "ub": ub,
            "batches": batches,
            "batchsize": batchsize,
            "dim": dim,
            "dtype": dtype,
            "sdf_func": sdf_func,
        }
        return [np.ones((batchsize, dim), dtype=np.float32)]

    dataset_call = {}

    def fake_create_dataset(X_data, U_data, X_res, batch_size):
        dataset_call["args"] = {
            "X_data": X_data,
            "U_data": U_data,
            "X_res": X_res,
            "batch_size": batch_size[1],
        }
        return "dataset_marker"

    monkeypatch.setattr(data_handling, "get_residual_sample", fake_get_residual_sample)
    monkeypatch.setattr(data_handling, "create_dataset", fake_create_dataset)

    out = handler.prepare_training(residual=True)

    assert out == "dataset_marker"
    assert handler.params_training["samples"] == [2, 4]
    assert tuple(handler.X_res.shape) == (8, 2)
    assert residual_call["args"]["batchsize"] == 8, \
    f"Expected batch size of 8 for residual sampling, found {residual_call['args']['batchsize']}"
    assert dataset_call["args"]["batch_size"] == 4, \
    f"Expected batch size of 4, found {dataset_call['args']['batch_size']}"


def test_prepare_nn_sets_normalization_bc_and_compile(monkeypatch):
    handler = DataHandler.__new__(DataHandler)
    handler.dtype = "float32"
    handler.X_data = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)
    handler.X_BC = {"inlet": ops.convert_to_tensor(np.zeros((2, 2), dtype=np.float32))}
    handler.U_BC = {"inlet": ops.convert_to_tensor(np.zeros((2, 1), dtype=np.float32))}
    handler.X_val = ops.convert_to_tensor(np.zeros((2, 2), dtype=np.float32))
    handler.U_val = {"u": ops.convert_to_tensor(np.zeros((2,), dtype=np.float32))}
    handler.params_NN = {"input_shape": (2,), "output_shape": 1}
    handler.params_training = {
        "NN_type": "test_model_factory",
        "loss_weights": {"lambda_data": 1.0, "lambda_BC": 0.5, "lambda_PDE": 0.1},
        "lr_schedule": "constant",
        "lr_start": 1e-3,
        "optimizer": "adam",
        "opti_beta_1": 0.9,
        "opti_beta_2": 0.999,
        "adam_eps": 1e-8,
        "adam_WD": 0.0,
        "samples": [3, 16],
        "epochs": [1, 1],
        "steps_per_execution": 8,
        "jit_compile": False,
        "auto_scale_loss": False,
    }

    factory_calls = {}

    def fake_model_factory(model_name, **kwargs):
        factory_calls["model_name"] = model_name
        factory_calls["kwargs"] = kwargs
        return _ModelStub()

    def fake_adam(**kwargs):
        return {"optimizer": "adam", "kwargs": kwargs}

    monkeypatch.setattr(data_handling.models, "test_model_factory", fake_model_factory, raising=False)
    monkeypatch.setattr(data_handling.keras.optimizers, "Adam", fake_adam)

    model = handler.prepare_NN(model_name="ModelClass")

    assert factory_calls["model_name"] == "ModelClass"
    assert handler.params_NN["mean"] == [1.0, 4.0]
    assert handler.params_NN["std"] == [1.0, 2.0]

    assert model.loss_weights.last_update == handler.params_training["loss_weights"]
    assert model.bc_call == (handler.X_BC, handler.U_BC)
    assert model.validation_data == (handler.X_val, handler.U_val)
    assert model.summary_called is True

    assert model.compile_kwargs["steps_per_execution"] == 3
    assert model.compile_kwargs["jit_compile"] is False
    assert model.compile_kwargs["auto_scale_loss"] is False
