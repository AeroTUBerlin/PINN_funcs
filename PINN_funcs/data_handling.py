import logging
import os
import warnings
from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras import ops
from matplotlib import pyplot as plt
from pyDOE import lhs

from . import models
from .callbacks import custom_callback, tb_custom
from .optimizers import SOAP, ExponentialDecayWithWarmup
from .utils import log_params, log_params_global

logger = logging.getLogger(__name__)


def load_npz_bundle(bundle_path: str, dtype: str = 'float32', BC_names: tuple = ('inlet',)) -> dict:
    """
    Load a case bundle from an `.npz` file.

    The bundle must contain `X_data`, `U_data`, `X_val`, `U_val`, `X_BC`, `U_BC` (the latter
    three are pickled dicts keyed by boundary-condition name such as "inlet" or "wall").

    Parameters
    ----------
    bundle_path : str
        Path to the `.npz` bundle file.
    dtype : str, optional
        Dtype to cast all loaded arrays/tensors to.
    BC_names : tuple, optional
        Boundary-condition names expected to be present in `X_BC`/`U_BC`; a `UserWarning` is
        raised if any are missing.

    Returns
    -------
    dict
        Keys `X_data`, `U_data`, `X_val` (tensors), `U_val`, `X_BC`, `U_BC` (dicts of tensors),
        and `lb`/`ub` (per-dimension min/max of `X_data`, as numpy arrays).
    """
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")
    with np.load(bundle_path, allow_pickle=True) as bundle:
        X_data = bundle['X_data'].astype(dtype)
        U_data = bundle['U_data'].astype(dtype)
        logger.info("Loaded data from bundle: X_data shape %s, U_data shape %s", X_data.shape, U_data.shape)

        X_BC = bundle['X_BC'].item()
        U_BC = bundle['U_BC'].item()
        logger.info(
            "Loaded BC data from bundle: X_BC keys %s, X_BC shapes %s",
            list(X_BC.keys()), [value.shape for value in X_BC.values()],
        )

        if not all(key in X_BC for key in BC_names) or not all(key in U_BC for key in BC_names):
            warnings.warn(f"Bundle BC dictionaries do not contain all specified BC names: {BC_names}", stacklevel=2)

        X_val  = bundle['X_val'].astype(dtype)
        U_val = bundle['U_val'].item()
        logger.info(
            "Loaded validation data from bundle: X_val shape %s, U_val keys %s", X_val.shape, list(U_val.keys()),
        )

    lb = X_data.min(axis=0).astype(dtype)
    ub = X_data.max(axis=0).astype(dtype)

    return {
        'X_data': ops.convert_to_tensor(X_data, dtype=dtype),
        'U_data': ops.convert_to_tensor(U_data, dtype=dtype),
        'X_val': ops.convert_to_tensor(X_val, dtype=dtype),
        'U_val': {key: ops.convert_to_tensor(value, dtype=dtype) for key, value in U_val.items()},
        'X_BC': {key: ops.convert_to_tensor(value, dtype=dtype) for key, value in X_BC.items()},
        'U_BC': {key: ops.convert_to_tensor(value, dtype=dtype) for key, value in U_BC.items()},
        'lb': lb,
        'ub': ub,
    }

class DataHandler:
    """
    Loads an NPZ case bundle and prepares training/validation data, model, and callbacks for
    a single PINN run.

    Subclass and override `load_data`, `prepare_data`, `prepare_NN`, `prepare_callbacks`, or `evaluate` to
    customize behavior for a specific case (see e.g `examples/Rhombus/Rhombus_Euler_rxy_cflearn.py`).

    Parameters
    ----------
    params_data : dict
        Case/data configuration. Must include `bundle_path` (or `dname`), `name`, and
        `save_path`.
    params_training : dict
        Training hyperparameters (optimizer, learning-rate schedule, loss weights, batch
        size, etc.). See `examples/configs/` for the expected keys.
    params_NN : dict
        Keyword arguments forwarded to the model factory selected via
        `params_training['NN_type']`.
    protocol : str, optional
        CSV file path that summary metrics for every run are appended to.
    dtype : str, optional
        Compute dtype for loaded data and the model.
    """
    def __init__(self, params_data: dict, params_training: dict, params_NN: dict,
                 protocol: str = 'results/' + 'Protokoll.csv',
                 dtype: str = 'float32', **kwargs):
        self.dtype = dtype
        self.params_data = params_data
        self.params_training = params_training
        self.params_NN = params_NN
        self.protocol = protocol

        self.load_data()

        self.params_data['case_name'] = os.path.join(self.params_data['name'], datetime.now().strftime('%Y%m%d-%H%M'))
        self.spath = os.path.join(self.params_data['save_path'],'results', self.params_data['case_name'], '')
        os.makedirs(self.spath, exist_ok=True)

    def load_data(self) -> None:
        """Load the NPZ bundle referenced by `params_data['bundle_path']` (or `'dname'`)."""
        bundle_path = self.params_data.get('bundle_path', self.params_data.get('dname'))
        if bundle_path is None or not str(bundle_path).endswith('.npz'):
            raise ValueError(
                "NPZ-first mode requires params_data['bundle_path'] (or params_data['dname']) to point to a .npz bundle file."  # noqa: E501
            )

        self.bundle_path = bundle_path
        self.bundle_mode = True
        self.sdf_func = None

        bundle = load_npz_bundle(
            self.bundle_path,
            dtype=self.dtype,
            BC_names=self.params_training.get('BC_weights', {}).keys()
        )

        self.X_BC, self.U_BC = bundle['X_BC'], bundle['U_BC']
        self.X_data, self.U_data = bundle['X_data'], bundle['U_data']
        self.X_res = None
        self.X_val = bundle['X_val']
        self.U_val = bundle['U_val']

        self.lb = bundle['lb']
        self.ub = bundle['ub']

    def log_params(self, comment: str | None = None) -> str:
        """Write run parameters to a per-run log file and append a summary row to `protocol`."""
        self.logfile = log_params(
            {**self.params_data, **self.params_training, **self.params_NN},
            os.path.join(self.spath, 'training'),
            comments=comment,
        )
        log_params_global(
            {**self.params_data, **self.params_training, **self.params_NN},
            self.protocol,
            comments=comment,
        )
        return self.logfile

    def prepare_callbacks(self) -> list:
        """Build the default callback list: TensorBoard logging, NaN termination, and metric logging."""
        logs = 'logs/' + self.params_data['case_name']

        tb_callback = tb_custom(
            log_dir=logs,
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch',
            record_freq=self.params_training['print_freq'],
        )
        term = keras.callbacks.TerminateOnNaN()

        loss_cb = custom_callback(
            log_dir=logs,
            eval_freq=self.params_training['eval_freq'],
            print_freq=self.params_training['print_freq'],
            save_freq=self.params_training['save_freq'],
        )

        return [tb_callback, term, loss_cb]

    def prepare_data(self) -> None:
        """Record dataset metadata (`bundle_path`, `input_dim`, `n_data`) onto `params_data`."""
        self.params_data['bundle_path'] = self.bundle_path
        self.params_data['input_dim'] = int(self.X_data.shape[1])
        self.params_data['n_data'] = int(self.X_data.shape[0])

    def prepare_training(self, residual: bool = False) -> tf.data.Dataset:
        """
        Build the training `tf.data.Dataset`, optionally sampling fresh PDE collocation points.

        Parameters
        ----------
        residual : bool, optional
            If True, (re-)sample `X_res` collocation points via latin hypercube sampling
            (filtered by `sdf_func` if set) before building the dataset.

        Returns
        -------
        tf.data.Dataset
            Yields `(X_data_batch, U_data_batch, X_res_batch)` tuples.
        """
        logger.info("Preparing training data...")

        if isinstance(self.params_training['batch_size'], int):
            n_batches = int(np.maximum(1.0, np.floor(len(self.X_data) / self.params_training['batch_size'])))
            samples = [n_batches, self.params_training['batch_size']]
        
        else:
            n_batches = int(np.maximum(1.0, np.floor(len(self.X_data) / self.params_training['batch_size'][0])))
            samples = [n_batches] + self.params_training['batch_size']

        self.params_training['samples'] = samples

        if residual:
            n_res_total = samples[0] * samples[-1]
            self.X_res = get_residual_sample(
                self.lb,
                self.ub,
                batches=1,
                batchsize=n_res_total,
                dim=len(self.lb),
                dtype=self.dtype,
                sdf_func=self.sdf_func,
            )[0]
            self.X_res = ops.convert_to_tensor(self.X_res)

        return create_dataset(self.X_data, self.U_data, self.X_res, batch_size=samples)

    def prepare_validation(self, distance_th: float = 0.0, **kwargs) -> None:
        """
        Snapshot the full validation set as `X_val_full`/`U_val_full` for plotting, then optionally restrict
        `X_val`/`U_val` to points at or beyond `distance_th` from the SDF boundary (if `sdf_func`
        is set).
        """
        self.X_val_full = self.X_val
        self.U_val_full = self.U_val.copy()

        if self.sdf_func is not None:
            distance = self.sdf_func(self.X_val)
            mask = distance >= distance_th
            self.X_val = self.X_val[mask]
            for key in self.U_val:
                self.U_val[key] = self.U_val[key][mask]

    def prepare_NN(self, model_name: type[keras.Model]) -> keras.Model:
        """
        Build, configure, and compile a PINN model from `params_training`/`params_NN`.

        Computes input normalization statistics from the training data, creates the
        model via the name in `params_training['NN_type']` (see `PINN_funcs.models` for options),
        attaches loss weights and boundary conditions, and compiles it with the configured
        learning-rate schedule and optimizer (`'adam'` or `'soap'`).

        Parameters
        ----------
        model_name : type[keras.Model]
            PINN model class to instantiate (forwarded to the model factory as `model_class`).

        Returns
        -------
        keras.Model
            A compiled model ready for `train_model`.
        """
        # Define parameters based on training data to ensure coordinates are between -1 and 1 after normalization.
        self.params_NN['mean'] = np.mean(self.X_data, axis=0).tolist()
        self.params_NN['std'] = np.max(np.abs(self.X_data - self.params_NN['mean']), axis=0).tolist()

        for i, std_value in enumerate(self.params_NN['std']):
            if std_value == 0:
                self.params_NN['std'][i] = 1.0

        model_func = getattr(models, self.params_training['NN_type'])
        model = model_func(model_name, **self.params_NN)

        try:
            model.dtype = self.dtype
        except AttributeError:
            pass
        model.loss_weights.update_state(self.params_training['loss_weights'])
        model.set_BC(self.X_BC, self.U_BC)
        model.validation_data = (self.X_val, self.U_val)

        if self.params_training['lr_schedule'] == 'CosineDecay':
            lr_schedule = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=self.params_training['lr_start'],
                decay_steps=(
                    self.params_training['samples'][0]
                    * self.params_training['epochs'][0]
                    * self.params_training['epochs'][1]
                ),
                alpha=self.params_training['lr_decay'],
                warmup_target=self.params_training['lr_warmup'],
                warmup_steps=self.params_training.get('lr_warmup_steps', 1000),
            )
        elif self.params_training['lr_schedule'] == 'ExponentialDecay':
            decay_steps = (
                self.params_training['samples'][0]
                * self.params_training['epochs'][0]
                * self.params_training['epochs'][1]
                * np.log(self.params_training['lr_decay_rate'])
                / np.log(self.params_training['lr_decay'])
            )
            lr_schedule = ExponentialDecayWithWarmup(
                initial_learning_rate=self.params_training['lr_start'],
                decay_steps=decay_steps,
                decay_rate=self.params_training['lr_decay_rate'],
                warmup_target=self.params_training['lr_warmup'],
                warmup_steps=self.params_training.get('lr_warmup_steps', 1000),
            )
        else:
            logger.warning(
                "Unsupported learning rate schedule: %s, using constant learning rate",
                self.params_training['lr_schedule'],
            )
            lr_schedule = self.params_training['lr_start']

        if self.params_training['optimizer'] == 'adam':
            optim = keras.optimizers.Adam(
                learning_rate=lr_schedule,
                beta_1=self.params_training.get('opti_beta_1', 0.9),
                beta_2=self.params_training.get('opti_beta_2', 0.999),
                epsilon=self.params_training.get('adam_eps', 1e-8),
                amsgrad=False,
                weight_decay=self.params_training.get('adam_WD', 0.0),
                name='Adam',
            )
        elif self.params_training['optimizer'] == 'soap':
            optim = SOAP(
                learning_rate=lr_schedule,
                beta_1=self.params_training.get('opti_beta_1', 0.95),
                beta_2=self.params_training.get('opti_beta_2', 0.95),
                precondition_frequency=self.params_training.get('soap_precondition_frequency', 10),
                epsilon=self.params_training.get('adam_eps', 1e-8),
                weight_decay=self.params_training.get('adam_WD', 0.0),
                name='SOAP',
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.params_training['optimizer']}")
        
        optim = keras.optimizers.LossScaleOptimizer(optim)

        model.compile(
            optimizer=optim,
            steps_per_execution=np.minimum(self.params_training['samples'][0], 
                                           self.params_training.get('steps_per_execution', 16)),
            jit_compile=self.params_training.get('jit_compile', True),
            auto_scale_loss=self.params_training.get('auto_scale_loss', False),
            run_eagerly=self.params_training.get('run_eagerly', False)
        )
        model.summary()
        logger.info("Model setup completed. Selected optimizer: %s", self.params_training['optimizer'])
        return model

    def plot_sampling(
        self, dataset: tf.data.Dataset, batch: int = 0, show: bool = True,
        BC_colors: tuple = ('r', 'g', 'c', 'm', 'y', 'orange'),
    ):
        """
        Scatter-plot one batch's data points, PDE collocation points, and boundary-condition
        points (2D inputs only).

        Parameters
        ----------
        dataset : tf.data.Dataset
            Dataset as returned by `prepare_training`.
        batch : int, optional
            Index of the batch to plot.
        show : bool, optional
            If True, display the figure interactively; otherwise close it after building.
        BC_colors : tuple, optional
            Colors cycled across boundary-condition groups.

        Returns
        -------
        matplotlib.figure.Figure or None
            The created figure, or None if the input dimensionality isn't 2.
        """
        batches = [batch_item for batch_item in dataset]
        input_dim = batches[batch][0].shape[-1] if batches[batch][0] is not None else 0

        if input_dim == 2:
            fig_sampling = plt.figure()
            ax = fig_sampling.add_subplot()
            ax.scatter(batches[batch][0][:, 0], batches[batch][0][:, 1], c='k', s=0.25, label='Data')

            if len(batches[batch]) > 2 and self.params_training['loss_weights']['lambda_PDE'] > 0:
                ax.scatter(batches[batch][2][:, 0], batches[batch][2][:, 1], c='b', s=0.25, label='PDE Points')

            if self.X_BC is not None:
                for i, (key, value) in enumerate(self.X_BC.items()):
                    ax.scatter(value[:, 0], value[:, 1], c=BC_colors[i % len(BC_colors)], s=2, label=f'BC ({key})')

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend(loc='best')
        else:
            logger.warning("Cannot visualize data with dimensionality %d", input_dim)
            return None

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        if show:
            plt.show()
        else:
            plt.close()

        return fig_sampling

def get_residual_sample(
    lb: np.ndarray,
    ub: np.ndarray,
    batches: int = 32,
    batchsize: int = 8192,
    dtype: str = 'float32',
    dim: int = 2,
    sampling_method: str = 'lhs',
    **kwargs,
) -> list:
    """
    Generate random latin hypercube sample in the domain defined by lb and ub.
    Supersampling is used to ensure enough points outside the SDF filtering region.

    Parameters
    ----------
    lb, ub : np.ndarray
        Per-dimension lower/upper bounds of the sampling domain.
    batches : int, optional
        Number of batches to split the final sample into.
    batchsize : int, optional
        Number of points per batch.
    dtype : str, optional
        Dtype of the returned arrays.
    dim : int, optional
        Number of dimensions to sample.
    sampling_method : str, optional
        Currently only latin hypercube sampling is implemented; this parameter is unused.
    **kwargs
        `sdf_func`, if provided and not None, filters out points with negative signed distance.

    Returns
    -------
    list[np.ndarray]
        `batches` arrays of shape `(batchsize, dim)`.
    """
    # Generate a supersampled latin hypercube sample in the domain defined by lb and ub. 
    X_res = (lb + (ub - lb) * lhs(dim, int(2 * batchsize * batches), criterion='center')).astype(dtype)
    np.random.shuffle(X_res, )

    # Apply SDF filtering if provided in kwargs
    if 'sdf_func' in kwargs and kwargs['sdf_func'] is not None:
        distance = kwargs['sdf_func'](X_res)
        mask = distance > 0
        X_res = X_res[mask]
    
    idx = np.arange(batchsize * batches)
    return np.split(X_res[idx], batches)


def create_dataset(
    X_data: tf.Tensor,
    U_data: tf.Tensor,
    X_res: tf.Tensor,
    batch_size: list,
) -> tf.data.Dataset:
    """
    Create a combined tf.data.Dataset where the smaller dataset is repeated to match the larger one.

    Args:
        X_data: Labeled data coordinates, shape (n_data, n_dims)
        U_data: Labeled data values, shape (n_data, n_vars)
        X_res: Collocation points, shape (n_res, n_dims)
        batch_size: Fixed batch size for all components

    Returns:
        combined_dataset: Dataset yielding (X_data_batch, U_data_batch, X_res_batch)
    """
    n_data = len(X_data)
    n_res = len(X_res)
    
    # Create dataset for labeled data points
    data_dataset = tf.data.Dataset.from_tensor_slices((X_data, U_data))
    data_dataset = data_dataset.cache()  # Cache the raw tensors
    data_dataset = data_dataset.shuffle(buffer_size=n_data, reshuffle_each_iteration=False)
    
    # Create dataset for collocation points
    res_dataset = tf.data.Dataset.from_tensor_slices(X_res)
    res_dataset = res_dataset.cache()  # Cache the raw tensors
    
    # Determine which dataset should repeat
    if n_data < n_res:
        # More collocation points - data repeats
        logger.info("Repeating data dataset: %d data points < %d collocation points", n_data, n_res)
        data_dataset = data_dataset.repeat()
        data_dataset = data_dataset.batch(batch_size[1], drop_remainder=True)
        res_dataset = res_dataset.batch(batch_size[-1], drop_remainder=False)
    else:
        # More data points - collocation points repeat
        logger.info("Repeating residual dataset: %d collocation points < %d data points", n_res, n_data)
        res_dataset = res_dataset.repeat()
        data_dataset = data_dataset.batch(batch_size[1], drop_remainder=False)
        res_dataset = res_dataset.batch(batch_size[-1], drop_remainder=True)

    # Zip them together
    combined_dataset = tf.data.Dataset.zip((data_dataset, res_dataset))

    # Restructure to match expected format: (X_data, U_data, X_res)
    combined_dataset = combined_dataset.map(
        lambda data_batch, res_batch: (data_batch[0], data_batch[1], res_batch)
    )
    combined_dataset = combined_dataset.cache()

    logger.info("Combined dataset created with %d batches of size %s", len(list(combined_dataset)), batch_size)
    return combined_dataset
