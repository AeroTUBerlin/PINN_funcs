import logging
import timeit

logger = logging.getLogger(__name__)


def train_model(model, dataset, callbacks=None, epochs=1, iterations=1, 
               resample=False, Data_Handler=None, lw_decay=None, decouple_batching=False, **kwargs):
    """
    Train a model using a tf dataset with optional resampling and learning rate decay.
    
    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained.
    dataset : tf.data.Dataset
        The dataset to be used for training.
    callbacks : list, optional
        List of callbacks to be applied during training.
    epochs : int, optional
        Number of epochs to train the model for.
    iterations : int, optional
        Number of iterations to train the model for.
    resample : bool, optional
        Whether to resample the dataset after each iteration.
    Data_Handler : object, optional
        Data handler object to prepare training data. (Required if resample is True)
    lw_decay : float, optional
        Decay factor for the data loss weight after each iteration. (Default is None)
    decouple_batching : bool, optional
        Use decoupled batching strategy when resampling. (Default is False)
    
    Returns
    -------
    history : dict
        History of the training process.
    """

    start_time = timeit.default_timer()

    for i in range(iterations):
        start_epoch = i * epochs
        end_epoch = start_epoch + epochs
        if resample and (i > 0):
            logger.info("Resampling data for iteration %d...", i)
            dataset = Data_Handler.prepare_training(dataset=True, residual=True,
                                                    decouple_batching=decouple_batching)

        batches = [batch for batch in dataset]
        model.data = batches[0]
        logger.info("Training iteration %d...", i)

        h = model.fit(dataset, epochs=end_epoch, verbose=0,
                     callbacks=callbacks, initial_epoch=start_epoch, **kwargs)

        h.history["epoch"] = h.epoch
        if i == 0:
            history = h.history
        else:
            for key in h.history.keys():
                history[key] += h.history[key]

        if lw_decay is not None:
            old_lambda_data = model.loss_weights.lambda_data.value
            model.loss_weights.update_state({'lambda_data': old_lambda_data * lw_decay})
            logger.info("Updated lambda_data: %s", model.loss_weights.lambda_data.value)

    training_time = timeit.default_timer() - start_time
    logger.info("Training took %.2f seconds", training_time)
    history["training_time"] = training_time

    return history