import keras
import numpy as np
from keras import ops
from keras.layers import InputSpec


class FourierMapping(keras.layers.Layer):
    """
    A layer that applies a Fourier feature mapping to the input data, 
    as described in Tancik et al. 2020 (http://arxiv.org/abs/2006.10739).
    """
    def __init__(self, n_features=128, sigma=1,**kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features

        if isinstance(sigma, (int, float)):
            self.sigma = sigma
        elif isinstance(sigma, (list, np.ndarray)):
            self.sigma = ops.convert_to_tensor(sigma, dtype=self.dtype)
        else:
            raise ValueError(
                "sigma must be either a single number or a list of numbers with the same length as the input shape"
                )

    def build(self, input_shape):
        self.B = self.add_weight(
            shape=(input_shape[-1], self.n_features // 2),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=1.0),
            trainable=False, dtype=self.dtype
        ) * ops.reshape(self.sigma, (-1, 1))

    def call(self, x):
        B = ops.cast(self.B, self.compute_dtype)
        ff = ops.stack([ops.sin(2*np.pi*x @ B), ops.cos(2*np.pi*x @ B)], axis=-1)

        return ops.reshape(ff, (-1, self.n_features))

class RWFDense(keras.layers.Layer):  # Inherit from Layer, not Dense
    """
    A dense layer with random weight factorization (RWF).

    Implements the RWF method from Wang & Perdikaris 2022 (http://arxiv.org/abs/2210.01274) where weights are
    factorized as W = g * v, with g being learnable scaling factors.
    """
    def __init__(self, units, activation=None, use_bias=True,
                 kernel_initializer="glorot_uniform", bias_initializer="zeros",
                 mean=1.0, std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.mean = mean
        self.std = std
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Step 1: Create the base weight matrix W using standard initialization
        w_base = self.kernel_initializer((input_dim, self.units), dtype=self.dtype)

        # Step 2: Create g values (scaling factors) from log-normal distribution
        g_random = keras.random.normal((self.units,), 
                                       mean=self.mean, 
                                       stddev=self.std, 
                                       dtype=self.dtype, seed=keras.random.SeedGenerator(seed=42)
                                       )
        g_values = ops.exp(g_random)

        # Step 3: Initialize g as trainable parameter
        self.g = self.add_weight(
            name="g",
            shape=(self.units,),
            initializer=lambda shape, dtype: ops.cast(g_values, dtype),
            trainable=True
        )

        # Step 4: Initialize v = W / g (using the SAME g values)
        v_values = w_base / g_values  # Broadcasting: (input_dim, units) / (units,)

        self.v = self.add_weight(
            name="v",
            shape=(input_dim, self.units),
            initializer=lambda shape, dtype: ops.cast(v_values, dtype),
            trainable=True  # CRITICAL: v must be trainable
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.bias_initializer,
                trainable=True
            )
        else:
            self.bias = None

        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        super().build(input_shape)

    def call(self, inputs):
        # Compute effective kernel: W = g * v
        # Broadcasting: (input_dim, units) * (units,) -> (input_dim, units)
        kernel = ops.cast(self.v * self.g, self.compute_dtype)

        # Standard dense computation
        outputs = ops.matmul(inputs, kernel)

        if self.bias is not None:
            outputs = ops.add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'mean': self.mean,
            'std': self.std
        }
        base_config = super().get_config()
        return {**base_config, **config}
    
