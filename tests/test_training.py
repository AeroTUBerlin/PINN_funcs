from PINN_funcs.training import train_model


class _LambdaValue:
    def __init__(self, value):
        self.value = value


class _LossWeights:
    def __init__(self, value=1.0):
        self.lambda_data = _LambdaValue(value)

    def update_state(self, values):
        self.lambda_data.value = values["lambda_data"]


class _FitHistory:
    def __init__(self, loss_value, epoch_start):
        self.history = {"loss": [loss_value]}
        self.epoch = [epoch_start]


class _ModelStub:
    def __init__(self):
        self.loss_weights = _LossWeights(1.0)
        self.fit_calls = []
        self.data = None

    def fit(self, dataset, epochs, verbose, callbacks, initial_epoch, **kwargs):
        self.fit_calls.append(
            {
                "epochs": epochs,
                "initial_epoch": initial_epoch,
                "callbacks": callbacks,
                "dataset": dataset,
            }
        )
        return _FitHistory(loss_value=0.5 + initial_epoch, epoch_start=initial_epoch)


class _DataHandlerStub:
    def __init__(self, replacement_dataset):
        self.calls = 0
        self.replacement_dataset = replacement_dataset

    def prepare_training(self, dataset, residual, decouple_batching):
        self.calls += 1
        assert dataset is True
        assert residual is True
        return self.replacement_dataset


def test_train_model_accumulates_history_and_applies_decay():
    initial_dataset = [("x0", "u0", "r0")]
    resampled_dataset = [("x1", "u1", "r1")]
    model = _ModelStub()
    data_handler = _DataHandlerStub(replacement_dataset=resampled_dataset)

    history = train_model(
        model,
        initial_dataset,
        callbacks=["cb"],
        epochs=1,
        iterations=2,
        resample=True,
        Data_Handler=data_handler,
        lw_decay=0.5,
        decouple_batching=True,
    )

    assert len(model.fit_calls) == 2
    assert model.fit_calls[0]["initial_epoch"] == 0
    assert model.fit_calls[1]["initial_epoch"] == 1
    assert data_handler.calls == 1

    assert history["epoch"] == [0, 1]
    assert len(history["loss"]) == 2
    assert "training_time" in history

    assert model.loss_weights.lambda_data.value == 0.25
