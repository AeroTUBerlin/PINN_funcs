from itertools import chain

import keras
import tensorflow as tf
from keras import ops, saving
from keras.optimizers import Optimizer


def merge_small_dims(shape_to_merge, max_dim):
    r"""Merge small dimensions.

        If there are some small dimensions, we collapse them
            e.g. [1, 2, 512, 1, 2048, 1, 3, 4] --> [1024, 2048, 12] if max_dim = 1024
            [1, 2, 768, 1, 2048] --> [2, 768, 2048].

    :param shape_to_merge: Union[List[int], torch.Size]. Shape to merge small dimensions.
    :param max_dim: int. Maximal dimension of output shape used in merging.
    """
    merged_shape = []

    product = 1
    for dim in shape_to_merge:
        product *= dim
        if product > max_dim:
            merged_shape.append(product // dim)
            product = dim

    merged_shape.append(product)

    return merged_shape if len(merged_shape) > 1 else [1]

@saving.register_keras_serializable()
class SOAP(Optimizer):
    """ 
    Tensorflow implementation of the SOAP optimizer following the paper https://arxiv.org/abs/2409.11321 
    and the code in https://github.com/nikhilvyas/SOAP. Based on the Keras Optimizer class, 
    with custom state variables for the Shampoo preconditioner and custom update rules.
    """
    def __init__(
            self,
            learning_rate=3e-3,
            beta_1=0.95,
            beta_2=0.95,
            epsilon=1e-8,
            weight_decay=1e-2,
            shampoo_beta=None,
            precondition_frequency=10,
            max_precondition_dim=10000,
            merge_dims=False,
            precondition_1d=False,
            correct_bias=True,
            normalize_gradient=False,
            data_format='channels_last',
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            loss_scale_factor=None,
            gradient_accumulation_steps=None,
            name="soap",
            **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            loss_scale_factor=loss_scale_factor,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **kwargs,
        )
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.shampoo_beta = shampoo_beta if shampoo_beta is not None else beta_2
        self.precondition_frequency = precondition_frequency
        self.max_precondition_dim = max_precondition_dim
        self.merge_dims = merge_dims
        self.precondition_1d = precondition_1d
        self.correct_bias = correct_bias
        self.normalize_gradient = normalize_gradient
        self.data_format = data_format

    # Update the property accessors to use the tracked variables
    @property
    def GG(self):
        """Access GG variables as nested list structure."""
        return self._gg_variables

    @property
    def Q(self):
        """Access Q variables as nested list structure."""
        return self._q_variables

    @property
    def matrices(self):
        """Access matrices variables as nested list structure."""
        return self._matrices_variables

    def build(self, var_list):
        """Initialize optimizer state for all trainable variables."""
        if self.built:
            return
        super().build(var_list)

        self.exp_avg = []
        self.exp_avg_sq = []
        self._gg_variables = []
        self._q_variables = []
        self._matrices_variables = []

        for i, var in enumerate(var_list):
            # Add momentum variables
            self.exp_avg.append(self.add_variable_from_reference(
                reference_variable=var, name=f"exp_avg_{i}"
            ))
            self.exp_avg_sq.append(self.add_variable_from_reference(
                reference_variable=var, name=f"exp_avg_sq_{i}"
            ))

            # Initialize preconditioner variables
            gg_vars, q_vars, matrices_vars = self._init_preconditioner_variables(var, i)
            self._gg_variables.append(gg_vars)
            self._q_variables.append(q_vars)
            self._matrices_variables.append(matrices_vars)

    def _init_preconditioner_variables(self, var, var_index):
        """Initialize preconditioner variables for a single parameter."""
        gg_vars = []
        q_vars = []
        matrices_vars = []

        if len(var.shape) == 1:
            if not self.precondition_1d or var.shape[0] > self.max_precondition_dim:
                # Create scalar placeholder variables
                gg_vars.append(self.add_variable(
                    shape=(), initializer="ones", name=f"gg_{var_index}_0"
                ))
                q_vars.append(self.add_variable(
                    shape=(), initializer="ones", name=f"q_{var_index}_0"
                ))
                matrices_vars.append(self.add_variable(
                    shape=(), initializer="ones", name=f"matrices_{var_index}_0"
                ))
            else:
                # Create matrix variables
                gg_vars.append(self.add_variable(
                    shape=(var.shape[0], var.shape[0]),
                    initializer="zeros",
                    dtype=var.dtype,
                    name=f"gg_{var_index}_0"
                ))
                q_vars.append(self.add_variable(
                    shape=(var.shape[0], var.shape[0]),
                    initializer="zeros",
                    dtype=var.dtype,
                    name=f"q_{var_index}_0"
                ))
                matrices_vars.append(self.add_variable(
                    shape=(var.shape[0], var.shape[0]),
                    initializer="zeros",
                    dtype=var.dtype,
                    name=f"matrices_{var_index}_0"
                ))
        else:
            shape_to_use = var.shape
            if self.merge_dims:
                shape_to_use = merge_small_dims(var.shape, self.max_precondition_dim)

            for dim_idx, sh in enumerate(shape_to_use):
                if sh > self.max_precondition_dim:
                    # Scalar placeholders
                    gg_vars.append(self.add_variable(
                        shape=(), initializer="ones", name=f"gg_{var_index}_{dim_idx}"
                    ))
                    q_vars.append(self.add_variable(
                        shape=(), initializer="ones", name=f"q_{var_index}_{dim_idx}"
                    ))
                    matrices_vars.append(self.add_variable(
                        shape=(), initializer="ones", name=f"matrices_{var_index}_{dim_idx}"
                    ))
                else:
                    # Matrix variables
                    gg_vars.append(self.add_variable(
                        shape=(sh, sh),
                        initializer="zeros",
                        dtype=var.dtype,
                        name=f"gg_{var_index}_{dim_idx}"
                    ))
                    q_vars.append(self.add_variable(
                        shape=(sh, sh),
                        initializer="zeros",
                        dtype=var.dtype,
                        name=f"q_{var_index}_{dim_idx}"
                    ))
                    matrices_vars.append(self.add_variable(
                        shape=(sh, sh),
                        initializer="zeros",
                        dtype=var.dtype,
                        name=f"matrices_{var_index}_{dim_idx}"
                    ))

        return gg_vars, q_vars, matrices_vars

    def _project_jit(self, grad, q_vars, project_type='forward'):
        """Project gradient through orthogonal matrices Q."""
        for mat in q_vars:
            if mat.shape.ndims > 0:
                grad = tf.tensordot(grad, mat, axes=[[0], [0 if project_type == 'forward' else 1]])
            else:
                grad = tf.transpose(grad, ([*list(range(1, len(grad.shape))), 0]))
        return grad
    
    def project(
            self,
            param,
            grad,
            merge_dims=False,
            max_precondition_dim=10000,
            project_type='forward',
    ):
        """Project gradient through Q matrices with optional dimension merging."""
        original_shape = grad.shape

        if merge_dims:
            if self.data_format == 'channels_first' and grad.dim() == 4:
                permuted_shape = tf.transpose(grad, (0, 2, 3, 1)).shape

            grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

        grad = self._project_jit(grad, self.Q[self._get_variable_index(param)], project_type)

        if merge_dims:
            if self.data_format == 'channels_first' and len(original_shape) == 4:
                grad = tf.transpose(tf.reshape(grad, permuted_shape), (0, 3, 1, 2))
            else:
                grad = tf.reshape(grad, original_shape)

        return grad

    def _compute_eigenvectors_gpu(self, m):
        """Compute eigenvectors via eigendecomposition on GPU with float32 stability."""
        m_reg = m + 1e-30 * tf.eye(m.shape[0], dtype=m.dtype)
        
        compute_dtype = tf.float32 if m.dtype == tf.float16 else m.dtype
        m_compute = tf.cast(m_reg, compute_dtype)
        
        # Eigendecomposition on GPU
        eigenvalues, eigenvectors = tf.linalg.eigh(m_compute)
        
        # Sort in descending order
        eigenvectors = tf.reverse(eigenvectors, axis=[1])
        
        # Cast back to original dtype
        return tf.cast(eigenvectors, m.dtype)

    def get_orthogonal_matrix(self, param, mat):
        """Compute orthogonal Q matrices from gradient Gram matrix via eigendecomposition."""
        param_idx = self._get_variable_index(param)
        
        for i, m in enumerate(mat):
            if m.shape.ndims == 0:
                continue
            
            q = self._compute_eigenvectors_gpu(m)
            self.matrices[param_idx][i].assign(q)

    def _compute_qr_gpu(self, m, o):
        """Compute QR decomposition via power iteration on GPU with float32 stability."""
        est_eig = tf.linalg.diag_part(tf.matmul(tf.transpose(o), tf.matmul(m, o)))
        sort_idx = tf.argsort(est_eig, direction='DESCENDING')
        
        power_iter = tf.matmul(m, tf.gather(o, sort_idx, axis=1))
        
        compute_dtype = tf.float32 if power_iter.dtype == tf.float16 else power_iter.dtype
        power_compute = tf.cast(power_iter, compute_dtype)
        q, _ = tf.linalg.qr(power_compute)
        
        return tf.cast(q, power_iter.dtype), sort_idx

    def get_orthogonal_matrix_qr(self, param, max_precondition_dim=10000, merge_dims=False):
        r"""Compute the eigen-bases of the pre-conditioner using one round of power iteration."""
        param_idx = self._get_variable_index(param)
        orig_shape = self.exp_avg_sq[param_idx].shape
        if self.data_format == 'channels_first' and len(orig_shape) == 4:
            permuted_shape = tf.transpose(self.exp_avg_sq[param_idx], (0, 2, 3, 1)).shape

        exp_avg_sq = self.exp_avg_sq[param_idx]
        if merge_dims:
            exp_avg_sq = tf.reshape(exp_avg_sq, merge_small_dims(exp_avg_sq.shape, max_precondition_dim))

        for ind, (m, o) in enumerate(zip(self.GG[param_idx], self.Q[param_idx])):
            if m.shape.ndims == 0:
                continue
            
            q, sort_idx = self._compute_qr_gpu(m, o)
            exp_avg_sq = tf.gather(exp_avg_sq, sort_idx, axis=ind)
            self.matrices[param_idx][ind].assign(q)

        if merge_dims:
            if self.data_format == 'channels_first' and len(orig_shape) == 4:
                exp_avg_sq.assign(tf.transpose(tf.reshape(exp_avg_sq, permuted_shape), (0, 3, 1, 2)))
            else:
                exp_avg_sq.assign(tf.reshape(exp_avg_sq, orig_shape))

    def update_pre_conditioner(
            self,
            param,
            grad,
            step,
            max_precondition_dim=10000,
            precondition_1d=False,
            merge_dims=False,
    ):
        """Update Shampoo preconditioner: Gram matrix GG and orthogonal bases Q.
        
        Called every precondition_frequency steps. Computes gradient outer products,
        updates GG with exponential moving average, and recomputes Q matrices via
        eigendecomposition (first iteration) or QR iteration (subsequent updates).
        """
        param_idx = self._get_variable_index(param)
        
        if len(grad.shape) == 1:
            if precondition_1d and grad.shape[0] <= max_precondition_dim:
                outer_product = tf.tensordot(
                    tf.expand_dims(grad, 1),
                    tf.expand_dims(grad, 0),
                    axes=1
                )
                GG = self.GG[param_idx][0]
                GG.assign(GG * self.shampoo_beta + tf.cast(outer_product, GG.dtype) * (1.0 - self.shampoo_beta))
        else:
            if merge_dims:
                grad = tf.reshape(grad, merge_small_dims(grad.shape, max_precondition_dim))

            for idx, dim in enumerate(grad.shape):
                if dim <= max_precondition_dim:
                    outer_product = tf.tensordot(
                        grad,
                        grad,
                        axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
                    )

                    GG = self.GG[param_idx][idx]
                    GG.assign(
                        GG * self.shampoo_beta + tf.cast(outer_product, GG.dtype) * (1.0 - self.shampoo_beta)
                    )

        def update_q_first():
            self.get_orthogonal_matrix(param, self.GG[param_idx])
            for i in range(len(self.Q)):
                if isinstance(self.Q[i], (list, tuple)):
                    self.Q[i] = self.matrices[i]
                else:
                    self.Q[i].assign(self.matrices[i])
        
        def update_q_periodic():
            self.get_orthogonal_matrix_qr(param, max_precondition_dim, merge_dims)
            for i in range(len(self.Q)):
                if isinstance(self.Q[i], (list, tuple)):
                    self.Q[i] = self.matrices[i]
                else:
                    self.Q[i].assign(self.matrices[i])
        
        def no_op():
            pass
        
        tf.cond(tf.equal(self.iterations, 0), update_q_first, no_op)
        tf.cond(
            tf.logical_and(
                tf.greater(self.iterations, 0),
                tf.equal(tf.math.floormod(self.iterations + 1, self.precondition_frequency), 0)
            ),
            update_q_periodic,
            no_op
        )


    def _update_step_jit(self, p, g, exp_avg_var, exp_avg_sq_var, q_vars, step, lr):
        """Core SOAP update: project gradients, update moments, apply preconditioned step."""
        grad_projected = self._project_jit(g, q_vars, 'forward')
        
        exp_avg_new = exp_avg_var * self.beta_1 + g * (1.0 - self.beta_1)
        exp_avg_sq_new = exp_avg_sq_var * self.beta_2 + tf.square(grad_projected) * (1.0 - self.beta_2)
        
        exp_avg_projected = self._project_jit(exp_avg_new, q_vars, 'forward')
        de_nom = tf.sqrt(exp_avg_sq_new) + self.epsilon
        norm_grad = self._project_jit(exp_avg_projected / de_nom, q_vars, 'backward')
        
        if self.normalize_gradient:
            norm_grad = tf.sqrt(norm_grad / tf.reduce_mean(tf.square(norm_grad))) + self.epsilon
        
        step_size = lr
        if self.correct_bias:
            bias_correction1 = 1.0 - tf.pow(self.beta_1, step)
            bias_correction2_sq = tf.sqrt(1.0 - tf.pow(self.beta_2, step))
            step_size = step_size * bias_correction2_sq / bias_correction1
        
        p_new = p - norm_grad * step_size
        
        if self.weight_decay > 0.0:
            p_new = p_new * (1.0 - self.weight_decay * lr)
        
        return p_new, exp_avg_new, exp_avg_sq_new
    
    def _backend_update_step(self, grads, trainable_variables, learning_rate):
        """Collective update_step that can be overridden by the backend.

        It is overridden by torch for performance reasons, and
        by TF to support tf.distribute.
        """
        self.update_step(grads, trainable_variables, learning_rate)

    def update_step(self, grads, trainable_variables, learning_rate):
        """Apply SOAP optimizer step: conditionally update preconditioner, then apply update."""
        for i, (p, g) in enumerate(zip(trainable_variables, grads)):
            if tf.keras.backend.is_sparse(g):
                raise RuntimeError('SOAP does not support sparse gradients')
            
            step = tf.cast(self.iterations + 1, p.dtype)
            lr = tf.cast(learning_rate, p.dtype)
            
            def should_update_preconditioner():
                self.update_pre_conditioner(
                    p, g,
                    step=step,
                    max_precondition_dim=self.max_precondition_dim,
                    precondition_1d=self.precondition_1d,
                    merge_dims=self.merge_dims,
                )
                
            def skip_preconditioner():
                pass

            tf.cond(
                tf.logical_or(
                    tf.equal(self.iterations, 0),
                    tf.equal(tf.math.floormod(self.iterations + 1, self.precondition_frequency), 0)
                ),
                should_update_preconditioner,
                skip_preconditioner
            )
            
            p_new, exp_avg_new, exp_avg_sq_new = self._update_step_jit(
                p, g,
                self.exp_avg[i],
                self.exp_avg_sq[i],
                self.Q[i],
                step,
                lr
            )
            
            p.assign(p_new)
            self.exp_avg[i].assign(exp_avg_new)
            self.exp_avg_sq[i].assign(exp_avg_sq_new)

    def get_config(self):
        config = super().get_config()
        config.update({
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "epsilon": self.epsilon,
            "shampoo_beta": self.shampoo_beta,
            "precondition_frequency": self.precondition_frequency,
            "max_precondition_dim": self.max_precondition_dim,
            "merge_dims": self.merge_dims,
            "precondition_1d": self.precondition_1d,
            "correct_bias": self.correct_bias,
            "normalize_gradient": self.normalize_gradient,
            "data_format": self.data_format,
        })
        return config

    def _apply_weight_decay(self, variables):
        pass


@saving.register_keras_serializable()
class ExponentialDecayWithWarmup(keras.optimizers.schedules.LearningRateSchedule):
    """A `LearningRateSchedule` that uses exponential decay with optional warmup.

    This schedule combines the exponential decay functionality from 
    `ExponentialDecay` with the linear warmup functionality from `CosineDecay`.

    Args:
        initial_learning_rate: A Python float. The initial learning rate.
        decay_steps: A Python integer. Must be positive. See the decay
            computation above.
        decay_rate: A Python float. The decay rate.
        staircase: Boolean. If `True` decay the learning rate at discrete
            intervals.
        warmup_target: A Python float. The target learning rate for our
            warmup phase. Will cast to the `initial_learning_rate` datatype.
            Setting to `None` will skip warmup and begins decay phase from
            `initial_learning_rate`. Otherwise scheduler will warmup from
            `initial_learning_rate` to `warmup_target`.
        warmup_steps: A Python int. Number of steps to warmup over.
        name: String. Optional name of the operation. Defaults to
            `"ExponentialDecayWithWarmup"`.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar tensor of the
        same type as `initial_learning_rate`.
    """

    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=False,
        warmup_target=None,
        warmup_steps=0,
        name="ExponentialDecayWithWarmup",
    ):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.name = name

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `decay_steps` must be > 0. "
                f"Received: decay_steps={self.decay_steps}"
            )

    def _decay_function(self, step, decay_steps, decay_rate, decay_from_lr, dtype):
        """Exponential decay function."""
        decay_steps = ops.cast(decay_steps, dtype)
        decay_rate = ops.cast(decay_rate, dtype)
        
        p = step / decay_steps
        if self.staircase:
            p = ops.floor(p)
        return ops.multiply(decay_from_lr, ops.power(decay_rate, p))

    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        """Linear warmup function."""
        completed_fraction = step / warmup_steps
        total_step_delta = warmup_target - initial_learning_rate
        return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        initial_learning_rate = ops.convert_to_tensor(
            self.initial_learning_rate
        )
        dtype = initial_learning_rate.dtype
        decay_steps = ops.cast(self.decay_steps, dtype)
        global_step_recomp = ops.cast(step, dtype)

        if self.warmup_target is None:
            # No warmup, apply exponential decay directly
            return self._decay_function(
                global_step_recomp,
                decay_steps,
                self.decay_rate,
                initial_learning_rate,
                dtype,
            )

        # With warmup
        warmup_target = ops.cast(self.warmup_target, dtype)
        warmup_steps = ops.cast(self.warmup_steps, dtype)

        return ops.cond(
            global_step_recomp < warmup_steps,
            lambda: self._warmup_function(
                global_step_recomp,
                warmup_steps,
                warmup_target,
                initial_learning_rate,
            ),
            lambda: self._decay_function(
                global_step_recomp - warmup_steps,
                decay_steps,
                self.decay_rate,
                warmup_target,
                dtype,
            ),
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "warmup_target": self.warmup_target,
            "warmup_steps": self.warmup_steps,
            "name": self.name,
        }
