import keras
from keras import ops

from .layers import FourierMapping, RWFDense


def get_model_vanilla(
    model_class: type[keras.Model],
    input_shape: tuple[int, ...] = (2,),
    output_shape: int = 4,
    output_scaler: float = 1.0,
    n_layers: int = 8,
    n_neurons: int = 64,
    activation: str = 'swish',
    dtype: str = 'float32',
) -> keras.Model:
    """
    Build a vanilla fully-connected PINN model.

    Parameters:
    ----------
    model_class : type[keras.Model]
        Functional-API model subclass to instantiate (e.g. a `PINN_tf` variant).
    input_shape : tuple[int, ...], optional
        Shape of a single input sample, excluding the batch dimension.
    output_shape : int, optional
        Number of model outputs.
    output_scaler : float, optional
        Constant factor applied to the final linear output layer.
    n_layers : int, optional
        Number of hidden Dense layers.
    n_neurons : int, optional
        Width of each hidden Dense layer.
    activation : str, optional
        Keras activation name used in hidden layers.
    dtype : str, optional
        Compute dtype for the output layer and `output_scaler`.

    Returns
    -------
    keras.Model
        An uncompiled instance of `model_class`.
    """
    output_scaler = ops.convert_to_tensor(output_scaler, dtype=dtype)
    inputs = keras.Input(shape=input_shape)
    X = inputs
    for _ in range(n_layers):
        X = keras.layers.Dense(n_neurons,
                               activation=keras.activations.get(activation))(X)
    outputs = keras.layers.Dense(output_shape, dtype=dtype, activation='linear')(X) * output_scaler
    return model_class(inputs=inputs, outputs=outputs, name='PINN_vanilla')

def get_model_norm(
    model_class: type[keras.Model],
    input_shape: tuple[int, ...] = (2,),
    output_shape: int = 4,
    output_scaler: float = 1.0,
    n_layers: int = 8,
    n_neurons: int = 64,
    mean: float = 0.0,
    std: float = 1.0,
    RWF: bool = False,
    activation: str = 'swish',
    dtype: str = 'float32',
) -> keras.Model:
    """
    Build a PINN model with input normalization, optionally using random weight
    factorization (RWF, Wang et al. 2021) in the dense layers.

    Parameters
    ----------
    model_class : type[keras.Model]
        Functional-API model subclass to instantiate (e.g. a `PINN_tf` variant).
    input_shape : tuple[int, ...], optional
        Shape of a single input sample, excluding the batch dimension.
    output_shape : int, optional
        Number of model outputs.
    output_scaler : float, optional
        Constant factor applied to the final linear output layer.
    n_layers : int, optional
        Number of hidden Dense (or RWFDense) layers.
    n_neurons : int, optional
        Width of each hidden layer.
    mean, std : float, optional
        Per-feature input normalization statistics.
    RWF : bool, optional
        If True, hidden layers use `RWFDense` instead of `keras.layers.Dense`.
    activation : str, optional
        Keras activation name used in hidden layers.
    dtype : str, optional
        Compute dtype for the output layer and `output_scaler`.

    Returns
    -------
    keras.Model
        An uncompiled instance of `model_class`.
    """
    output_scaler = ops.convert_to_tensor(output_scaler, dtype=dtype)
    mean = ops.convert_to_tensor(mean, dtype=dtype)
    std = ops.convert_to_tensor(std, dtype=dtype)

    inputs = keras.Input(shape=input_shape)
    X = keras.layers.Normalization(axis=1, mean=mean, variance=std ** 2)(inputs)
    for _ in range(n_layers):
        if RWF:
            X = RWFDense(n_neurons, activation=keras.activations.get(activation))(X)
        else:
            X = keras.layers.Dense(n_neurons,
                               activation=keras.activations.get(activation))(X)
    outputs = keras.layers.Dense(output_shape, dtype=dtype, activation='linear', use_bias=False)(X) * output_scaler
    return model_class(inputs=inputs, outputs=outputs, name='PINN_norm')

def get_model_fourier(
    model_class: type[keras.Model],
    input_shape: tuple[int, ...] = (2,),
    output_shape: int = 4,
    output_scaler: float = 1.0,
    n_layers: int = 8,
    n_neurons: int = 64,
    sigma: float = 1.0,
    ff_scale: int = 128,
    activation: str = 'swish',
    dtype: str = 'float32',
) -> keras.Model:
    """
    Build a PINN model with a Fourier feature mapping layer (Tancik et al. 2020) followed
    by stacked Dense layers. No input normalization.

    Parameters
    ----------
    model_class : type[keras.Model]
        Functional-API model subclass to instantiate (e.g. a `PINN_tf` variant).
    input_shape : tuple[int, ...], optional
        Shape of a single input sample, excluding the batch dimension.
    output_shape : int, optional
        Number of model outputs.
    output_scaler : float, optional
        Constant factor applied to the final linear output layer.
    n_layers : int, optional
        Number of hidden Dense layers after the Fourier mapping.
    n_neurons : int, optional
        Width of each hidden Dense layer.
    sigma : float, optional
        Standard deviation of the random Fourier feature frequencies.
    ff_scale : int, optional
        Number of Fourier features.
    activation : str, optional
        Keras activation name used in hidden layers.
    dtype : str, optional
        Compute dtype for the output layer and `output_scaler`.

    Returns
    -------
    keras.Model
        An uncompiled instance of `model_class`.
    """
    inputs = keras.Input(shape=input_shape)
    X = FourierMapping(n_features=ff_scale, sigma=sigma)(inputs)

    for _ in range(n_layers):
        X = keras.layers.Dense(n_neurons,
                               activation=keras.activations.get(activation))(X)

    output_scaler = ops.convert_to_tensor(output_scaler, dtype=dtype)
    outputs = keras.layers.Dense(output_shape, dtype=dtype, activation='linear', use_bias=False)(X) * output_scaler
    return model_class(inputs=inputs, outputs=outputs, name='PINN_fourier')

def get_model_fourier_norm(
    model_class: type[keras.Model],
    input_shape: tuple[int, ...] = (2,),
    output_shape: int = 4,
    output_scaler: float = 1.0,
    n_layers: int = 8,
    n_neurons: int = 64,
    sigma: float = 1.0,
    mean: float = 0.0,
    std: float = 1.0,
    ff_scale: int = 128,
    activation: str = 'swish',
    dtype: str = 'float32',
    RWF: bool = False,
) -> keras.Model:
    """
    Build a PINN model with input normalization, a Fourier feature mapping layer, and
    stacked Dense (or RWFDense) layers.

    Parameters
    ----------
    model_class : type[keras.Model]
        Functional-API model subclass to instantiate (e.g. a `PINN_tf` variant).
    input_shape : tuple[int, ...], optional
        Shape of a single input sample, excluding the batch dimension.
    output_shape : int, optional
        Number of model outputs.
    output_scaler : float, optional
        Constant factor applied to the final linear output layer.
    n_layers : int, optional
        Number of hidden layers after the Fourier mapping.
    n_neurons : int, optional
        Width of each hidden layer.
    sigma : float, optional
        Standard deviation of the random Fourier feature frequencies.
    mean, std : float, optional
        Per-feature input normalization statistics.
    ff_scale : int, optional
        Number of Fourier features.
    activation : str, optional
        Keras activation name used in hidden layers.
    dtype : str, optional
        Compute dtype for the output layer and `output_scaler`.
    RWF : bool, optional
        If True, hidden layers use `RWFDense` instead of `keras.layers.Dense`.

    Returns
    -------
    keras.Model
        An uncompiled instance of `model_class`.
    """
    mean = ops.convert_to_tensor(mean, dtype=dtype)
    std = ops.convert_to_tensor(std, dtype=dtype)

    inputs = keras.Input(shape=input_shape)
    X = keras.layers.Normalization(axis=1, mean=mean, variance=std ** 2)(inputs)
    ## Fourier Mapping
    X = FourierMapping(n_features=ff_scale, sigma=sigma)(X)
    for _ in range(n_layers):
        if RWF:
            X = RWFDense(n_neurons, activation=keras.activations.get(activation))(X)
        else:
            X = keras.layers.Dense(n_neurons,
                               activation=keras.activations.get(activation))(X)

    output_scaler = ops.convert_to_tensor(output_scaler, dtype=dtype)
    outputs = keras.layers.Dense(output_shape, dtype=dtype, activation='linear', use_bias=False)(X) * output_scaler
    return model_class(inputs=inputs, outputs=outputs, name='PINN_fourier')
