import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import keras
import numpy as np
import tensorflow as tf
from keras import ops
from matplotlib.backends.backend_pdf import PdfPages

from PINN_funcs.callbacks import Autoweight_cb, cb_calfac, tb_custom
from PINN_funcs.data_handling import DataHandler
from PINN_funcs.PDEs import Euler_2D_conservative_E
from PINN_funcs.PINNs import PINN_tf_rxy_cflearn
from PINN_funcs.plotting import plot_results
from PINN_funcs.training import train_model
from PINN_funcs.utils import append_to_log, print_metrics, reset_random_seeds


class Rhombus:
    """
    A class that defines the lower boundary of a rhombus-shaped computational domain.
    
    The domain has a rhombus protrusion between x=0 and x=0.6096, with the peak at 
    x=0.3048, y=0.081670913853. Outside this range, the boundary is flat at y=0.
    
    Attributes:
        x1 (float): Left edge of rhombus (0.0)
        x_peak (float): Peak x-coordinate (0.3048)
        x2 (float): Right edge of rhombus (0.6096)
        y_peak (float): Peak y-coordinate (0.081670913853)
    """
    
    def __init__(self, L=1.0):

        # Define the rhombus geometry based on the provided vertices
        offset = 0.0 #0.0008
        self.x1 = -offset / L #0.0                    # Left edge of rhombus
        self.x_peak = 0.3048 / L            # Peak x-coordinate
        self.x2 = (0.6096 + offset) / L                # Right edge of rhombus
        self.y_peak = 0.081670913853 / L #0.081670913853    # Peak y-coordinate

    def __call__(self, x):
        """
        Return the y-coordinate of the lower boundary for given x-coordinates.
        
        Args:
            x (array-like): Array of x-coordinates
            
        Returns:
            numpy.ndarray: Corresponding y-coordinates of the lower boundary
        """
        x = np.asarray(x)
        y = np.zeros_like(x)
        
        # Rising section: linear interpolation from (0, 0) to (x_peak, y_peak)
        rising_mask = (x >= self.x1) & (x <= self.x_peak)
        y[rising_mask] = (x[rising_mask] - self.x1) * self.y_peak / (self.x_peak - self.x1)
        
        # Falling section: linear interpolation from (x_peak, y_peak) to (x2, 0)
        falling_mask = (x > self.x_peak) & (x <= self.x2)
        y[falling_mask] = (self.x2 - x[falling_mask]) * self.y_peak / (self.x2 - self.x_peak)
        
        # Flat sections (x < 0 or x > 0.6096) remain at y = 0 (already initialized)
        
        return y

    def sample_points(self, num_points, lb = -0.15242, ub = 0.9139):
        """
        Sample points along the lower boundary of the rhombus.

        Args:
            num_points (int): Number of points to sample

        Returns:
            tuple: Arrays of x and y coordinates of sampled points
        """

        x_samples = np.linspace(lb, ub, num_points)  # Sample beyond the rhombus edges
        y_samples = self.__call__(x_samples)

        return x_samples, y_samples
    
    def signed_distance(self, xy, mask_tip=0.0, mask_shock=0.0):
        """
        Compute the signed distance from points (x, y) to the rhombus boundary.

        Args:
            xy (array-like): Array of shape (N, 2) containing x and y coordinates
        
        Returns:
            numpy.ndarray: Array of signed distances (positive above the boundary, negative below)
        """
        x = np.asarray(xy[:, 0])
        y = np.asarray(xy[:, 1])

        y_boundary = self.__call__(x)
        distance = y - y_boundary  # Positive above the boundary, negative below

        if mask_tip > 0.0:
            tip_mask = (abs(x-self.x_peak) < mask_tip) & (abs(y - self.y_peak) < mask_tip)
            distance[tip_mask] = -1.0  # Inside the tip region
        
        if mask_shock > 0.0:
            shock_mask = ((abs(x - self.x1) < mask_shock) | (abs(x - self.x2) < mask_shock)) & (abs(y) < mask_shock)
            distance[shock_mask] = -1.0  # Inside the shock region

        return distance
    
    def get_angle(self, xy):
        """
        Compute the angle of the rhombus boundary normal at given points (x, y).

        Args:
            xy (array-like): Array of shape (N, 2) containing x and y coordinates
        
        Returns:
            numpy.ndarray: Array of angles in radians
        """
        x = np.asarray(xy[:, 0])
        angles = np.zeros_like(x)

        # Rising section
        rising_mask = (x >= self.x1) & (x <= self.x_peak)
        angles[rising_mask] = np.arctan2(self.y_peak, self.x_peak - self.x1)

        # Falling section
        falling_mask = (x > self.x_peak) & (x <= self.x2)
        angles[falling_mask] = np.arctan2(-self.y_peak, self.x2 - self.x_peak)

        # Flat sections remain at angle 0 (already initialized)

        return angles

class PINN_Rhombus(PINN_tf_rxy_cflearn):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.PDE = Euler_2D_conservative_E
            
    def get_pred_BC(self, X_BC_dict):
        U_BC_pred = {}
        for key, value in X_BC_dict.items():
            if key == 'wall':
                out = self(value)
                u = out[:, 0:1]
                v = out[:, 1:2]

                # Compute unit vectors of velocity
                vel_magnitude = ops.sqrt(u**2 + v**2) + 1e-8
                ux = u / vel_magnitude
                uy = v / vel_magnitude

                U_BC_pred[key] = ops.concatenate([ux, uy], axis=1)

            elif key == 'shock':
                out = self(value)
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
                U_BC_pred[key] = tf.stack([ops.mean(M21), ops.mean(p21), ops.mean(r21)], axis=-1)

            else:
                U_BC_pred[key] = self(value)
        return U_BC_pred

class DH_Rhombus(DataHandler):
    def __init__(self, params_data, params_training, params_NN,
                 protocol = 'results/' + 'Protokoll.csv',
                 **kwargs):
        super().__init__(params_data, params_training, params_NN,
                         protocol = protocol, **kwargs)   
        self.Ma = params_data['Ma']

    def evaluate(self, model, history, sdf_func=None):
        # Log final metrics and save history
        metrics = model.val_step(model.validation_data, get_dict=False)
        metrics = {key: value.numpy() for key, value in metrics.items()}

        if isinstance(history, dict) and "training_time" in history:
            metrics['training_time'] = history["training_time"]

        print_metrics(metrics, self.logfile + '.log')
        append_to_log(metrics, self.protocol, append_last_row=True)
        self.metrics = metrics

        try:
            with open(self.spath + 'history.json', 'w', encoding='utf-8') as history_file:
                json.dump(history, history_file)
        except (TypeError, ValueError, OSError):
            pass
        
        # Evaluate the model on the full validation dataset and plot results
        X_val_full = self.X_val_full
        val_dict_full = self.U_val_full

        pred_dict, err_dict, _ = model.val_step((X_val_full, val_dict_full), get_dict=True)
        ref_dict = {key+'_ref': value.numpy() for key, value in val_dict_full.items()}
        pred_dict = {key: value.numpy() for key, value in pred_dict.items()}
        err_dict = {key+'_err': value.numpy() for key, value in err_dict.items()}

        x, y = X_val_full.numpy().T

        if sdf_func is not None:
            sdf = sdf_func(X_val_full.numpy())
            mask = sdf >= 0.0  # Keep points above or on the boundary
            x = np.ma.array(x, mask=~mask)
            y = np.ma.array(y, mask=~mask)
            for dict_ in [pred_dict, ref_dict, err_dict]:
                dict_ = {key: np.ma.array(value, mask=~mask) for key, value in dict_.items()}
            mask = None

        plot_results(x, y, pred_dict | err_dict | ref_dict, spath=self.spath, levels=50, cmap='viridis', mask=mask)

        # Save the trained model
        model.save(self.spath + 'model.keras')

    def prepare_callbacks(self):
        logs = os.path.join(self.params_data['save_path'], 'logs', self.params_data['case_name'])

        tb_callback = tb_custom(
            log_dir=logs,
            histogram_freq=0,
            write_graph=False,
            update_freq='epoch',
            record_freq=self.params_training['print_freq'],
        )
        term = keras.callbacks.TerminateOnNaN()

        loss_cb = cb_calfac(log_dir=logs,
                            eval_freq=self.params_training['eval_freq'],
                            print_freq=self.params_training['print_freq'],
                            save_freq=self.params_training['save_freq'],
                            )
        
        autoweight = Autoweight_cb(update_freq=self.params_training['update_freq'])
        
        return [tb_callback, term, loss_cb, autoweight]

    def prepare_data(self):
        super().prepare_data()
        self.T0_ref = 1 + 0.2 * self.Ma ** 2

    def prepare_NN(self, model_name):
        model = super().prepare_NN(model_name)
        if self.params_training['BC_weights']:
            for key, value in self.params_training['BC_weights'].items():
                model.BC_weights[key] = ops.convert_to_tensor(value, dtype=self.dtype)

        model.T0 = self.T0_ref
        model.Ma = self.Ma

        model.cal_exp.trainable = self.params_training['cal_exp_trainable']
        model.initial_guess = tf.Variable(self.params_training['cal_initial_guess'], dtype=self.dtype)
        model.cal_base = self.params_training['cal_base']
        return model


def main(data, nn_training, nn_args,
         comment=None, protocol='results/' + 'Log_Rhombus.csv'):

    """
    main function to train a PINN model on the simulated BOS dataset for the Rhombus airfoil.
    """
    #limit_memory_growth()

    keras.mixed_precision.set_global_policy(nn_training['data_policy'])
    reset_random_seeds(nn_training['random_seed'])
    dtype = 'float32'

    ################################################################################################################
    ## Load Data ##

    DF = DH_Rhombus(data, nn_training, nn_args, protocol=protocol, dtype=dtype)

    rhombus = Rhombus(L=data['L'])
    DF.sdf_func = rhombus.signed_distance

    DF.prepare_data()
    DF.prepare_validation()
    
    dataset = DF.prepare_training(residual=True)
    
    # Plot the created sampling points
    with PdfPages(DF.spath + 'sampling.pdf') as pp:
        fig = DF.plot_sampling(dataset, batch=0, show=False)
        pp.savefig(fig)

    ################################################################################################################
    ## Prepare Model ##
    DF.params_training['epochs'] = [int(nn_training['steps'][0]/DF.params_training['samples'][0]), 
                                    nn_training['steps'][1]]
    
    DF.model = DF.prepare_NN(PINN_Rhombus)

    callbacks = DF.prepare_callbacks()
    DF.log_params(comment=comment)
    
    
    ################################################################################################################
    ## Training ##
    DF.history = train_model(DF.model, dataset,
                         callbacks=callbacks,
                         epochs=DF.params_training['epochs'][0],
                         iterations=DF.params_training['epochs'][1],
                         resample=True,
                         Data_Handler=DF,
                         lw_decay=1.0)

    ################################################################################################################
    ## Evaluate ##
    DF.evaluate(DF.model, DF.history, sdf_func=rhombus.signed_distance)

    return DF

if __name__ == '__main__':
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    ## Get Defaults ##
    nn_training = json.load(open(os.path.join(repo_root, 'configs', 'nn_training_default.json')))

    ## Define Parameters ##
    comment = 'Euler Rhombus Simulated BOS without BCs and disabled cflearn'

    data = {
                'name': 'PIQS/temp',
                'bundle_path': os.path.join(
                    repo_root,
                    'data',
                    'rhombus',
                    'rhombus_euler_bundle_v1.npz'
                ),
                'save_path': os.path.dirname(__file__),
                'script_name': os.path.basename(__file__),
                'xi': [0, 1],
                'Ma': 3.0,
                'L': 0.081670913853 #0.6096 # length scale to nondim lengths
            }
    
    nn_training.update({
                'NN_type': 'get_model_fourier_norm',
                'optimizer': 'soap',
                'soap_precondition_frequency': 32,
                'train_vars': ['rx', 'ry'],
                'val_vars': ['u', 'v', 'p', 'r', 'E'],
                'lr_schedule': 'CosineDecay',
                'data_policy': 'mixed_float16',
                'loss_weights': {'lambda_data': 1.0, 'lambda_BC': 1.0, 'lambda_PDE': 0.01},
                'BC_weights': {'wall': 0.01, 'inlet': 1.00,
                               'shock': [0.0, 0.0, 0.0] #['M21', 'p21', 'r21']
                               }, 
                'batch_size': 4096,
                'steps': [1000, 1],
                'random_seed': 42,
                'print_freq': 10,
                'eval_freq': 25,
                'save_freq': 200,
                'cal_exp_trainable': False,
                'cal_initial_guess': 1.0,
                'cal_base': 2.0,
                'update_freq': 0,#100,  # add number of steps for autoweight callback
            })

    nn_args = {
            'input_shape': (2,),
            'output_shape': 4,
            'n_layers': 4,
            'n_neurons': 128,
            'activation':'swish',
            'sigma': 0.1,
            'ff_scale': 256,
            'RWF': True
    }

    DF = main(
        data,
        nn_training,
        nn_args,
        comment=comment,
        protocol=os.path.join(repo_root, 'Rhombus', 'Log_Rhombus.csv')
    )
