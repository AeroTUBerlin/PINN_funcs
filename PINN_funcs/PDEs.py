import keras
import tensorflow as tf
from keras import ops


def sutherland(T: tf.Tensor, T_0: float = 273, mu_0: float = 1.716 * 10 ** -5, dim: bool = True) -> tf.Tensor:
    """
    Sutherland's law for dynamic viscosity.

    Parameters
    ----------
    T : tf.Tensor
        Temperature. Dimensional (Kelvin) if `dim` is True, else nondimensionalized by `T_0`.
    T_0 : float, optional
        Reference temperature used for nondimensionalization (Kelvin).
    mu_0 : float, optional
        Reference dynamic viscosity at `T_0` (used only when `dim` is True).
    dim : bool, optional
        If True, return dimensional viscosity `mu`. If False, return the nondimensional
        ratio `mu / mu_0`.

    Returns
    -------
    tf.Tensor
        Dynamic viscosity (or viscosity ratio if `dim` is False).
    """
    if dim:
        C = 110.4

        mu = mu_0 * (T / T_0 + 1e-8) ** 1.5 * (T_0 + C) / (C + T)
    else:
        C = 110.4
        mu = (T + 1e-8) ** 1.5 * (1 + C / T_0) / (T + C / T_0)  # mu/mu0
    return mu

def get_Total(
    U: tf.Tensor, p: tf.Tensor, r: tf.Tensor, dim: bool = True, Ma_infty: float = 2.28
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute total temperature and pressure, local speed of sound, and Mach
    number for air (k = 1.4).

    Parameters
    ----------
    U : tf.Tensor
        Velocity magnitude.
    p : tf.Tensor
        Static pressure (dimensional if `dim` is True, else nondimensionalized by `Ma_infty`).
    r : tf.Tensor
        Density (dimensional if `dim` is True, else nondimensionalized).
    dim : bool, optional
        If True, return dimensional total temperature and pressure. If False, return
        nondimensionalized values.
    Ma_infty : float, optional
        Freestream Mach number, used only when `dim` is False.

    Returns
    -------
    T_0 : tf.Tensor
        Total temperature.
    p_0 : tf.Tensor
        Total pressure.
    a : tf.Tensor
        Local speed of sound.
    Ma : tf.Tensor
        Local Mach number.
    """
    k = 1.4
    R = 287.15

    a = tf.sqrt(tf.math.maximum(k * p / (r + 1e-8), 1e-9))

    Ma = U / a

    T0_T = 1 + (k - 1) / 2 * Ma**2

    if dim:
        T_0 = p / (R * r) * T0_T

    else:
        T_0 = p / r * (k * Ma_infty**2) * T0_T

    p_0 = p * (T0_T ** (k / (k - 1)))

    return T_0, p_0, a, Ma

def safe_positive(x: tf.Tensor, k: float = 20, eps: float = 1e-6) -> tf.Tensor:
    """Smooth, differentiable approximation of `max(x, eps)` via a scaled softplus."""
    return tf.math.softplus(k * (x - eps)) / k + eps

def Euler_2D_conservative_E(model: keras.Model, X: tf.Tensor) -> tf.Tensor:
    '''
    This function evaluates the 2D, compressible, nondimensional Euler equations in their conservative form
    The model needs to have 4 outputs that correspond to (u,v,E,rho).
    The outputs for E and rho are forced to be positive by wrapping them in an exponential function.

    Parameters
    ----------
    model : keras.Model
        PINN model exposing `Ma` and `T0` attributes (`T0=None` disables the total-temperature
        constraint).
    X : tf.Tensor
        Input coordinates of shape (N, 2): columns are (x, y).

    Returns
    -------
    tf.Tensor
        PDE residuals of shape (N, 6): the four conservative Euler residuals, the optional
        total-temperature constraint, and an irrotationality condition.
    '''
    k = 1.4

    x = X[:, 0:1]
    y = X[:, 1:2]
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(y)

        out = model((tf.stack([x[:, 0], y[:, 0]], axis=1)))
        u = out[:, 0:1]
        v = out[:, 1:2]
        E = tf.exp(out[:, 2:3])
        r = tf.exp(out[:, 3:4])

        U = (u ** 2 + v ** 2) ** 0.5  # Magnitude of velocity
        p = r * (E - 0.5 * U**2) * (k - 1)
        p = tf.math.maximum(p, 1e-4)

        Y_x = [r * u, p + r * u ** 2, r * u * v, (r * E + p) * u]
        Y_y = [r * v, r * u * v, p + r * v ** 2, (r * E + p) * v]

    e1_x = tape.gradient(Y_x[0], x)
    e2_x = tape.gradient(Y_x[1], x)
    e3_x = tape.gradient(Y_x[2], x)
    e4_x = tape.gradient(Y_x[3], x)

    e1_y = tape.gradient(Y_y[0], y)
    e2_y = tape.gradient(Y_y[1], y)
    e3_y = tape.gradient(Y_y[2], y)
    e4_y = tape.gradient(Y_y[3], y)

    e1 = e1_x + e1_y
    e2 = e2_x + e2_y
    e3 = e3_x + e3_y
    e4 = e4_x + e4_y

    u_y = tape.gradient(u, y)
    v_x = tape.gradient(v, x)
    e6 = u_y - v_x

    if model.T0 is not None:
        T0, p0, a, M = get_Total(U, p, r, dim=False, Ma_infty=model.Ma)  # T0,p0
        e5 = T0 - tf.reduce_mean(T0)
    else:
        e5 = tf.zeros_like(e1)

    del tape

    return tf.concat([e1, e2, e3, e4, e5, e6], axis=-1)

def Euler_AS_conservative_E(model: keras.Model, X: tf.Tensor) -> tf.Tensor:
    '''
    This function evaluates the 2D, axissymmetric compressible, nondimensional Euler equations in conservative form
    The model needs to have 4 outputs that correspond to (u,v,E,rho).
    The outputs for E and rho are forced to be positive by wrapping them in an exponential function.

    Parameters
    ----------
    model : keras.Model
        PINN model exposing `Ma` and `T0` attributes (`T0=None` disables the total-temperature
        constraint).
    X : tf.Tensor
        Input coordinates of shape (N, 2): columns are (x, r) in axisymmetric coordinates.

    Returns
    -------
    tf.Tensor
        PDE residuals of shape (N, 6): the four conservative Euler residuals, the optional
        total-temperature constraint, and an irrotationality condition.
    '''

    k = 1.4

    x = X[:, 0:1]
    r = X[:, 1:2]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        tape.watch(r)

        out = model((tf.stack([x[:, 0], r[:, 0]], axis=1)))
        u = tf.exp(out[:, 0:1])
        v = out[:, 1:2]
        E = tf.exp(out[:, 2:3])
        rho = tf.exp(out[:, 3:4])

        U = (u ** 2 + v ** 2) ** 0.5  # Magnitude of velocity
        p = rho * (E - 0.5 * U**2) * (k - 1)
        p = tf.math.maximum(p, 1e-4)

        # Conservative fluxes in x and r
        flux_x = [rho * u, rho * u**2 + p, rho * u * v, (rho * E + p) * u]
        flux_r = [r * rho * v, r * rho * u * v, r * (rho * v ** 2 + p), r * (rho * E + p) * v]

    # Compute derivatives
    fx_x = [tape.gradient(f, x) for f in flux_x]
    fx_r = [tape.gradient(f, r) for f in flux_r]

    # Axisymmetric divergence (steady form): ∂/∂x(flux_x) + (1/r)*∂/∂r(r*flux_r)
    r_reg = tf.maximum(abs(r), 1e-4)
    e1 = fx_x[0] + (1 / r_reg) * fx_r[0]
    e2 = fx_x[1] + (1 / r_reg) * fx_r[1]
    e3 = fx_x[2] + (1 / r_reg) * fx_r[2] - p/r_reg
    e4 = fx_x[3] + (1 / r_reg) * fx_r[3]

    # Irrotationality condition
    u_r = tape.gradient(u, r)
    v_x = tape.gradient(v, x)
    e6 = u_r - v_x

    # Optionally enforce constant Total Temperature or Pressure
    if model.T0 is not None:
        T0, p0, a, M = get_Total(U, p, rho, dim=False, Ma_infty=model.Ma)  # T0,p0
        e5 = 1.0*(T0 - tf.reduce_mean(T0))
    else:
        e5 = tf.zeros_like(e1)
    del tape

    return tf.concat([e1, e2, e3, e4, e5, e6], axis=-1)

def RANS_2D_E(model: keras.Model, X: tf.Tensor) -> tf.Tensor:
    '''
    This function evaluates the 2D, compressible, nondimensional RANS equations in conservative form
    The model needs to have 5 outputs that correspond to (u,v,E,rho,mu_T).
    The outputs for E and rho are forced to be positive by wrapping them in an exponential function.

    Parameters
    ----------
    model : keras.Model
        PINN model exposing `Re`, `Ma`, `T0`, `mu_0`, `mu_T`, and `X_BC['inlet']` attributes.
    X : tf.Tensor
        Input coordinates of shape (N, 2): columns are (x, y).

    Returns
    -------
    tf.Tensor
        PDE residuals of shape (N, 5): the four conservative RANS residuals plus the
        constant-total-temperature constraint above the inlet boundary layer.
    '''
    gamma = 1.4

    Re = model.Re
    Ma = model.Ma

    Pr = 0.72
    Pr_T = 0.9

    x = X[:, 0:1]
    y = X[:, 1:2]

    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        tape.watch(y)

        out = model((tf.stack([x[:, 0], y[:, 0]], axis=1)))
        u = out[:, 0:1]
        v = out[:, 1:2]
        E = tf.exp(out[:, 2:3])
        r = tf.exp(out[:, 3:4])
        mu_T = out[:, 4:5] * model.mu_T

        U = (u ** 2 + v ** 2 + 1e-8) ** 0.5  # Magnitude of velocity
        T = gamma * (gamma - 1) * Ma ** 2 * (E - 0.5 * (u ** 2 + v ** 2))
        T = safe_positive(T)  # Ensure positive temperature
        p = r * T / (gamma * Ma ** 2)

        mu = sutherland(T, T_0=model.T0, mu_0=model.mu_0,
                        dim=False)

        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        T_x = tape.gradient(T, x)
        T_y = tape.gradient(T, y)

        S_xx = u_x - 1 / 3 * (u_x + v_y)
        S_xy = 0.5 * (u_y + v_x)
        S_yy = v_y - 1 / 3 * (u_x + v_y)

        tau_xx = 2 * (mu + mu_T) * S_xx 
        tau_xy = 2 * (mu + mu_T) * S_xy
        tau_yy = 2 * (mu + mu_T) * S_yy 

        q_x = (mu / Pr + mu_T / Pr_T) * T_x
        q_y = (mu / Pr + mu_T / Pr_T) * T_y

        Y_x = [r * u, p + r * u ** 2 - tau_xx / Re, r * u * v - tau_xy / Re,
               (r * E + p) * u - (u * tau_xx + v * tau_xy) / Re - q_x / ((gamma - 1) * Ma ** 2 * Re)]
        Y_y = [r * v, r * u * v - tau_xy / Re, p + r * v ** 2 - tau_yy / Re,
               (r * E + p) * v - (v * tau_yy + u * tau_xy) / Re - q_y / ((gamma - 1) * Ma ** 2 * Re)]

    e1_x = tape.gradient(Y_x[0], x)
    e2_x = tape.gradient(Y_x[1], x)
    e3_x = tape.gradient(Y_x[2], x)
    e4_x = tape.gradient(Y_x[3], x)

    e1_y = tape.gradient(Y_y[0], y)
    e2_y = tape.gradient(Y_y[1], y)
    e3_y = tape.gradient(Y_y[2], y)
    e4_y = tape.gradient(Y_y[3], y)


    e1 = e1_x + e1_y
    e2 = e2_x + e2_y
    e3 = e3_x + e3_y
    e4 = e4_x + e4_y

    del tape

    # Enforce constant Total Temperature at the inlet above the Boundary Layer
    T0 = get_Total(U, p, r, dim=False, Ma_infty=model.Ma)[0]

    idx = tf.where(X[:,1] > tf.reduce_min(model.X_BC['inlet'][:,1]))
    mask = tf.cast(X[:, 1:2] > tf.reduce_min(model.X_BC['inlet'][:,1]), dtype=X.dtype)
    e5 = (T0 - tf.reduce_mean(tf.gather(T0, idx))) * mask

    return tf.concat([e1, e2, e3, e4, e5], axis=-1)

def NS_3D_IC(model: keras.Model, X: tf.Tensor) -> tf.Tensor:
    '''
    This function evaluates the 3D, incompressible, dimensional Navier-Stokes equations
    The model needs to have 4 outputs that correspond to (u,v,w,p)
    The viscosity is computed as the inverse of the Reynolds number from the model with U = 1 and L = 1

    Parameters
    ----------
    model : keras.Model
        PINN model exposing `Re` and a 4-element `scaler` attribute (per-output scale factors).
    X : tf.Tensor
        Input coordinates of shape (N, 4): columns are (x, y, z, t).

    Returns
    -------
    tf.Tensor
        PDE residuals of shape (N, 4): continuity and the three momentum equations.
    '''
    # Constants
    nu = 1 / model.Re  # Reynolds number
    rho = 1000  # Density in kg/m3
    t_scale = 1.0   # Time scale L/U
    x = X[:, 0:1]
    y = X[:, 1:2]
    z = X[:, 2:3]
    t = X[:, 3:4]
    
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(x)
        tape.watch(y)
        tape.watch(z)
        tape.watch(t)
        
        out = model((tf.stack([x[:, 0], y[:, 0], z[:, 0], t[:,0]], axis=1)))
        u = out[:, 0:1] * model.scaler[0]# / 0.2    # x-velocity
        v = out[:, 1:2] * model.scaler[1]# / 0.2    # y-velocity
        w = out[:, 2:3] * model.scaler[2]# / 0.2    # z-velocity
        p = out[:, 3:4] * model.scaler[3]    # pressure
        
        # Velocity gradients
        u_x = tape.gradient(u, x)
        u_y = tape.gradient(u, y)
        u_z = tape.gradient(u, z)
        v_x = tape.gradient(v, x)
        v_y = tape.gradient(v, y)
        v_z = tape.gradient(v, z)
        w_x = tape.gradient(w, x)
        w_y = tape.gradient(w, y)
        w_z = tape.gradient(w, z)
        
    # Continuity equation (incompressibility constraint)
    e1 = u_x + v_y + w_z
    
    # Momentum equations
    # X-momentum

    lapl_u = ops.nan_to_num(tape.gradient(u_x, x) + tape.gradient(u_y, y) + tape.gradient(u_z, z), 0.0, 1e5, -1e5)
    e2 = (t_scale*tape.gradient(u, t) + 
            u * u_x + v * u_y + w * u_z + 
            tape.gradient(p, x) / rho - 
            nu * lapl_u)

    # Y-momentum
    lapl_v = ops.nan_to_num(tape.gradient(v_x, x) + tape.gradient(v_y, y) + tape.gradient(v_z, z), 0.0, 1e5, -1e5)
    e3 = (t_scale*tape.gradient(v, t) + 
            u * v_x + v * v_y + w * v_z + 
            tape.gradient(p, y) / rho -
            nu * lapl_v)

    # Z-momentum
    lapl_w = ops.nan_to_num(tape.gradient(w_x, x) + tape.gradient(w_y, y) + tape.gradient(w_z, z), 0.0, 1e5, -1e5)
    e4 = (t_scale*tape.gradient(w, t) + 
           u * w_x + v * w_y + w * w_z + 
            tape.gradient(p, z) / rho -
            nu * lapl_w)

    del tape
    return tf.concat([e1, e2, e3, e4], axis=-1)