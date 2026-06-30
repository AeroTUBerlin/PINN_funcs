import keras
import tensorflow as tf
from keras import ops

from .metrics import loss_weight_dict, single_Value
from .PDEs import get_Total
from .utils import MAE, MSE, MSE_zero


@keras.saving.register_keras_serializable()
class PINN_universal(keras.Model):
    '''
    A universal PINN class that can be extended for different backends (e.g., TensorFlow, PyTorch).
    This class defines the common structure and methods for PINN models, including loss calculations 
    and BC handling. For usage, extend this class and implement the train_step and test_step methods 
    for the specific backend and include a PDE function
    '''
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.X_BC = None
        self.U_BC = None
        self.PDE = None

        self.loss_total = single_Value(name='loss')
        self.loss_data = single_Value(name='loss_data')
        self.loss_res = single_Value(name='loss_res')
        self.loss_BC = {'total': single_Value(name='loss_BC')}

        self.learning_rate = single_Value(name='learning_rate')

        self.loss_weights = loss_weight_dict(name='lambda', dtype=self.dtype)

        self.hist = []

        self.indices = 0
  
    def get_loss_BC(self, U_BC: dict, U_BC_pred: dict, BC_weights: dict) -> tf.Tensor:
        """Sum of weighted per-boundary-condition MSE losses; also updates per-BC loss metrics."""
        loss = []
        for name_BC, value_BC in U_BC.items():
            err = MSE(U_BC_pred[name_BC], value_BC, axis=0)
            loss.append(ops.sum(err * BC_weights[name_BC]))
            self.loss_BC[name_BC].update_state(ops.sum(err))
        return ops.sum(loss)

    @staticmethod
    def get_loss_data(U_data: tf.Tensor, U_pred: tf.Tensor) -> tf.Tensor:
        """Per-output-channel MSE between labeled data and model predictions."""
        return MSE(U_data, U_pred, axis=0)

    @staticmethod
    def get_loss_PDE(U_pred: tf.Tensor) -> tf.Tensor:
        """Per-output-channel MSE-to-zero of the PDE residuals."""
        return MSE_zero(U_pred, axis=0)

    def get_loss(self, data: tuple) -> tf.Tensor:
        """
        Compute the total weighted loss (data + BC + PDE) for one batch and update the
        running loss metrics.

        Parameters
        ----------
        data : tuple
            `(X, U, X_res)`: labeled data coordinates, labeled data values, and PDE
            collocation points.

        Returns
        -------
        tf.Tensor
            Scalar total loss, weighted by `self.loss_weights`.
        """
        X, U, X_res = data
        U_pred_data = self.get_pred_data(X)
        loss_data = ops.sum(self.get_loss_data(U, U_pred_data))
        self.loss_data.update_state(loss_data)

        U_pred_PDE = self.get_pred_PDE(X_res)
        loss_res = ops.sum(self.get_loss_PDE(U_pred_PDE))
        self.loss_res.update_state(loss_res)

        if self.X_BC is not None:
            U_pred_BC = self.get_pred_BC(self.X_BC)
            loss_BC = self.get_loss_BC(self.U_BC, U_pred_BC, self.BC_weights)
            self.loss_BC['total'].update_state(loss_BC)
        else:
            loss_BC = ops.zeros(())

        self.loss_weights.update_state(self.loss_weights.result())
        
        return loss_data * self.loss_weights.lambda_data + \
            loss_BC * self.loss_weights.lambda_BC + \
            loss_res * self.loss_weights.lambda_PDE

    def get_pred_data(self, X_data: tf.Tensor) -> tf.Tensor:
        """Model prediction at labeled data coordinates. Override to predict derived quantities."""
        U_pred = self(X_data)
        return U_pred

    def get_pred_PDE(self, X_res: tf.Tensor) -> tf.Tensor:
        """PDE residuals at the collocation points, via `self.PDE`, or zeros if none is set."""
        if self.PDE is None:
            return ops.zeros_like(X_res)
        else:
            return self.PDE(self, X_res)

    def get_pred_BC(self, X_BC_dict: dict) -> dict:
        """Model prediction at each boundary condition's coordinates, keyed the same as `X_BC_dict`."""
        U_BC_pred = {}
        for key, value in X_BC_dict.items():
            U_BC_pred[key] = self(value)
        return U_BC_pred

    def set_BC(self, X_BC: dict, U_BC: dict, BC_weights: dict | None = None) -> None:
        """
        Register boundary-condition coordinates/targets and create a per-BC loss metric.

        Parameters
        ----------
        X_BC : dict
            Boundary-condition coordinates, keyed by BC name.
        U_BC : dict
            Boundary-condition target values, keyed by BC name.
        BC_weights : dict, optional
            Per-output-channel weight for each BC; defaults to all-ones.
        """

        self.X_BC = {key: ops.cast(value, self.dtype) for key, value in X_BC.items()}
        self.U_BC = {key: ops.cast(value, self.dtype) for key, value in U_BC.items()}
        if BC_weights is None:
            self.BC_weights = {key: ops.ones(value.shape[1]) for key,value in U_BC.items()}
        else:
            self.BC_weights = BC_weights
        
        for key in X_BC.keys():
            self.loss_BC[key] = single_Value(name='loss_BC_' + key)

    def train_step(self, *args, **kwargs):
        raise NotImplementedError("Implement train_step in a backend specific subclass")

    def test_step(self, *args, **kwargs):
        raise NotImplementedError("Implement test_step in a backend specific subclass")

@keras.saving.register_keras_serializable()
class PINN_tf(PINN_universal):
    """
    A basic PINN implementation using TensorFlow. This class extends the universal PINN class 
    and implements the train_step and test_step methods for TensorFlow. It also includes methods 
    for calculating the gradients of the data loss, BC loss, and PDE loss with respect to the 
    trainable parameters, which can be used for loss balancing
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val_scale = 1.0

        self.idx = None

    def _build_idx(self):
        """Build the index mapping for dynamic_stitch if not already built."""
        if self.idx is None:
            self.idx = []
            count = 0
            for i, var in enumerate(self.trainable_variables):
                shape = tf.shape(var)
                n = tf.reduce_prod(shape)
                self.idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape))
                count += n

    def _get_gradient_data(self):
        """
        Calculate the gradients of the weighted data loss with respect to the trainable parameters.
        """
        X, U, _ = [ops.cast(i, self.dtype) for i in self.data]
        with tf.GradientTape() as tape:
            U_pred = self.get_pred_data(X)
            loss_data = ops.sum(self.get_loss_data(U, U_pred)) * self.loss_weights.lambda_data

        grads = tape.gradient(loss_data, self.trainable_variables)
        return tf.dynamic_stitch(self.idx, grads)

    def _get_gradient_BC(self):
        """
        Calculate the gradients of the weighted BC loss with respect to the trainable parameters.
        """
        with tf.GradientTape() as tape:
            U_pred_BC = self.get_pred_BC(self.X_BC)
            loss_BC = self.get_loss_BC(self.U_BC, U_pred_BC, self.BC_weights) * self.loss_weights.lambda_BC

        grads = tape.gradient(loss_BC, self.trainable_variables)
        # If any entry of grads is none, set it to zero
        grads = [g if g is not None else tf.zeros_like(var) for g, var in zip(grads, self.trainable_variables)]

        return tf.dynamic_stitch(self.idx, grads)

    def _get_gradient_PDE(self):
        """
        Calculate the gradients of the weighted PDE loss with respect to the trainable parameters.
        """
        _, _, X_res = [ops.cast(i, self.dtype) for i in self.data]
        with tf.GradientTape() as tape:
            U_pred = self.get_pred_data(X_res)
            loss_PDE_weighted = self.get_loss_PDE(U_pred) * self.loss_weights.lambda_PDE
        grads = tape.gradient(loss_PDE_weighted, self.trainable_variables)
        # If any entry of grads is none, set it to zero
        grads = [g if g is not None else tf.zeros_like(var) for g, var in zip(grads, self.trainable_variables)]

        return tf.dynamic_stitch(self.idx, grads)
    
    @tf.function
    def call_dict(self, X: tf.Tensor) -> dict:
        """Model prediction at `X`, decoded into named physical quantities (u, v, p, r, E, muT)."""
        Y = self(X)

        u = Y[:, 0]
        v = Y[:, 1]
        E = ops.exp(Y[:, 2])
        r = ops.exp(Y[:, 3])
        muT = Y[:, 4]
        p = (E - 0.5 * (u ** 2 + v ** 2)) * r * 0.4

        return {'u': u, 'v': v,
                'p': p, 'r': r,
                'E': E, 'muT': muT}


    def test_step(self, data: tuple) -> dict:
        """Keras test-step override: compute the loss and return current metric values."""
        self.loss = self.get_loss(data)

        self.loss_total.update_state(self.loss)

        for metric in self.metrics:
            if metric.name == 'learning_rate':
                metric.update_state(self.optimizer.learning_rate)

        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data: tuple) -> dict:
        """Keras train-step override: one gradient-descent step on the weighted PINN loss."""
        with tf.GradientTape() as tape:
            self.loss = self.get_loss(data)
            loss_scaled = self.optimizer.scale_loss(self.loss)

        self.loss_total.update_state(self.loss)
        # Compute gradients
        trainables = self.trainable_variables
        grads = tape.gradient(loss_scaled, trainables)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == 'learning_rate':
                metric.update_state(self.optimizer.learning_rate)

        return {m.name: m.result() for m in self.metrics}

    def val_step(self, data: tuple, get_dict: bool = False):
        """
        Evaluate the model against a validation set and compute per-output error metrics
        (MAE, RMSE, MAPE, NRMSE).

        Parameters
        ----------
        data : tuple
            `(X_data, val_dict)`: validation coordinates and a dict of target values keyed
            by variable name (as produced by `call_dict`).
        get_dict : bool, optional
            If True, also return the prediction and error dicts.

        Returns
        -------
        dict or tuple
            `metrics` dict if `get_dict` is False, else `(pred_dict, err_dict, metrics)`.
        """
        X_data, val_dict = data

        pred_dict = self.call_dict(ops.cast(X_data, self.dtype))

        err_dict = {}
        metrics = {}

        for key, value in val_dict.items():
            value = ops.cast(value / self.val_scale, self.dtype)
            pred_dict[key] = ops.ravel(pred_dict[key] / self.val_scale)

            mae = ops.mean(abs(value - pred_dict[key]))
            rmse = (ops.mean((value - pred_dict[key]) ** 2)) ** 0.5
            nrmse = rmse / (ops.max(value) - ops.min(value))
            mape = ops.mean(abs((value - pred_dict[key])/(value + 1e-6))) * 100
            err_dict['err_' + key] = value - pred_dict[key]
            err_dict['rel_err_' + key] = (value - pred_dict[key]) / (1+value)
            metrics['mae_' + key] = mae
            metrics['rmse_' + key] = rmse
            metrics['mape_' + key] = mape
            metrics['nrmse_' + key] = nrmse

        if get_dict:
            return pred_dict, err_dict, metrics
        else:
            return metrics

@keras.saving.register_keras_serializable()
class PINN_tf_rxy(PINN_tf):
    """
    A variant of the PINN_tf class that used the density gradients r_x and r_y 
    as data for use with calibrated Schlieren-type measurements.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_pred_data(self, X_data: tf.Tensor) -> tf.Tensor:
        """Density-gradient prediction `(r_x, r_y)` at `X_data`, used as the training target."""
        U_pred = self.get_r_grad(X_data)
        return U_pred

    def get_r_grad(self, X: tf.Tensor) -> tf.Tensor:
        """Spatial gradient of density `(r_x, r_y)` at `X`, via automatic differentiation."""
        x = X[:, 0:1]
        y = X[:, 1:2]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(y)
            out = self((tf.stack([x[:, 0], y[:, 0]], axis=1)))
            r = ops.exp(out[:, 3])
        r_x = tape.gradient(r, x)
        r_y = tape.gradient(r, y)
        return tf.concat([r_x, r_y], axis=1)

    def get_pred_BC(self, X_BC_dict: dict) -> dict:
        """Model prediction at each BC's coordinates; `'wall'` additionally returns viscous stress terms."""
        U_BC_pred = {}
        for key, value in X_BC_dict.items():
            if key == 'wall':
                x = value[:, 0:1]
                y = value[:, 1:2]
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(y)
                    out = self((tf.stack([x[:, 0], y[:, 0]], axis=1)))
                    u = out[:, 0:1]
                v = out[:, 1:2]
                E = ops.exp(out[:, 2:3])
                r = ops.exp(out[:, 3:4])
                p = (E - 0.5 * (u ** 2 + v ** 2)) * r * 0.4
                p = tf.math.maximum(p, 1e-6)
                mu_T = out[:, 4:5] * self.mu_T

                U_BC_pred[key] = tf.concat([u, v, E, r, mu_T, p], axis=1)
            else:
                U_BC_pred[key] = self(value)
        return U_BC_pred

    def call_dict(self, X: tf.Tensor) -> dict:
        """Model prediction at `X`, decoded into named physical quantities plus (rx, ry, T0, p0, a, Ma)."""
        Y = self(X)

        u = Y[:, 0]
        v = Y[:, 1]
        E = ops.exp(Y[:, 2])
        r = ops.exp(Y[:, 3])

        U = (u ** 2 + v ** 2) ** 0.5  # Magnitude of velocity
        muT = Y[:, 4] * self.mu_T
        p = (E - 0.5 * (u ** 2 + v ** 2)) * r * 0.4

        rxy = self.get_r_grad(X)

        T0, p0, a, Ma = get_Total(U, p, r, dim=False, Ma_infty=self.Ma)

        return {'u': u, 'v': v,
                'p': p, 'r': r,
                'E': E, 'muT': muT,
                'rx': rxy[:, 0], 'ry': rxy[:, 1],
                'T0': T0, 'p0': p0, 'a': a, 'Ma': Ma}
    
@keras.saving.register_keras_serializable()
class PINN_tf_rxy_cflearn(PINN_tf_rxy):
    """
    A variant of the PINN_tf_rxy class that uses the density gradients r_x and r_y as training data 
    for use with uncalibrated Schlieren-type measurements.
    Therefore it includes an additional calibration parameter that is trained together with the PINN.
    To resolve the scale ambiguity, an additional BC is inclded, 
    that enforces the Rankine-Hugoniot jump conditions across a shock wave.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cal_exp = self.add_weight((), initializer='zeros', name='cal_exp', trainable=True, dtype=self.dtype)
        self.cal_base = 2.0
        self.initial_guess = tf.Variable(1.0, trainable=False, dtype=self.dtype)

        self.cal_fac_metric = single_Value(name='cal_fac')

    def get_loss_data(self, U_data: tf.Tensor, U_pred: tf.Tensor) -> tf.Tensor:
        """Per-output-channel MAE between `U_data` and `U_pred`, after applying the calibration factor."""
        cal_fac = self.cal_base**self.cal_exp * self.initial_guess
        U_data = ops.multiply(U_data, cal_fac)
        max = ops.max(U_data, axis=0)
        self.cal_fac_metric.update_state(cal_fac)
        return MAE(U_data/max, U_pred/max, axis=0)
    
    def get_pred_BC(self, X_BC_dict: dict) -> dict:
        """Model prediction at each BC's coordinates; `'shock'` returns Rankine-Hugoniot jump ratios."""
        U_BC_pred = {}
        for key, X in X_BC_dict.items():
            if key == 'shock':
                out = self(X) # X is (x,y) on both sides of the shock, stacked as [[x1,x2],[y1,y2]]
                u = ops.reshape(out[:, 0], (2, -1))
                v = ops.reshape(out[:, 1], (2, -1))
                E = ops.exp(ops.reshape(out[:, 2], (2, -1)))
                r = ops.exp(ops.reshape(out[:, 3], (2, -1)))
                p = (E - 0.5 * (u ** 2 + v ** 2)) * r * 0.4
                p = tf.math.maximum(p, 1e-3)
                Ma = ops.sqrt(u ** 2 + v ** 2) / ops.sqrt(1.4 * p / r)
                
                M21 = ops.divide_no_nan(Ma[1, :], Ma[0, :])
                p21 = ops.divide_no_nan(p[1, :], p[0, :])
                r21 = ops.divide_no_nan(r[1, :], r[0, :])
                U_BC_pred[key] = tf.stack([M21, p21, r21], axis=-1)
            else:
                U_BC_pred[key] = self(X)
        return U_BC_pred

    def call_dict(self, X: tf.Tensor) -> dict:
        """Model prediction at `X`, decoded into named physical quantities plus (rx, ry, T0, p0, a, Ma)."""
        Y = self(X)

        u = Y[:, 0]
        v = Y[:, 1]
        E = ops.exp(Y[:, 2])
        r = ops.exp(Y[:, 3])

        U = (u ** 2 + v ** 2) ** 0.5  # Magnitude of velocity
        p = (E - 0.5 * (u ** 2 + v ** 2)) * r * 0.4

        rxy = self.get_r_grad(X)

        T0, p0, a, Ma = get_Total(U, p, r, dim=False, Ma_infty=self.Ma)

        return {'u': u, 'v': v,
                'p': p, 'r': r,
                'E': E,
                'rx': rxy[:, 0], 'ry': rxy[:, 1],
                'T0': T0, 'p0': p0, 'a': a, 'Ma': Ma}

