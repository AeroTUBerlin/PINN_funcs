import csv
import json
import os

import keras
import numpy as np
import tensorflow as tf
from keras import ops


###########################################################################################################################
# Utility functions for logging and printing parameters and metrics
###########################################################################################################################
def append_to_log(params, logfile, append_last_row=False):
    # Read existing data
    with open(logfile, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        existing_data = list(reader)
        existing_header = reader.fieldnames

    updated_header = existing_header + [key for key in params if key not in existing_header]

    if append_last_row:
        existing_data[-1].update(params)
    else:
        existing_data.append(params)

    # Write the updated data back
    with open(logfile, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=updated_header, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in existing_data:
            writer.writerow(row)

def log_params(params, logfile, comments = None):
    """
        Function to log the parameters to a file.
        Args:
            params (dict): The dictionary of parameters.
            log_dir (str): The directory to save the log file.
    """
    with open(logfile + '.log', 'a') as f:
        if comments:
            f.write(f'\n{comments}\n')
            
        f.write('Starting Training with the following parameters:\n')
        for key, value in params.items():
            f.write(f'{key}: {value}\n')

    with open(logfile + '.json', 'w') as f:
        json.dump(params, f)
    
    return logfile

def log_params_global(params, logfile, comments=None):
    """
    Function to log the parameters to a file.
    Args:
        params (dict): The dictionary of parameters.
        logfile (str): The file to save the log.
    """

    params['comments'] = comments
    # Check if the file exists
    if os.path.exists(logfile):
        append_to_log(params, logfile)
    else:
        with open(logfile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=params.keys(), delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerow(params)


def print_metrics(metrics, logfile=None, tb_writer=None, epoch=None):
    '''
    Function to print the metrics in a nice format.
    Args:
        metrics (dict): The dictionary of metrics.
    '''

    for key, value in metrics.items():
        if key.startswith('mape'):
            f_value = "{:s} = {:6.2f}% \n".format(key, value)
        else:
            try:
                f_value = "{:s} = {:8.6f}, ".format(key, value)
            except TypeError:
                continue
        print(f_value, end="")

        if logfile:
            with open(logfile, 'a') as f:
                f.write(f_value + '\n')

        if tb_writer:
            with tb_writer.as_default(step=epoch):
                tf.summary.scalar(key, value)

def read_logfile(logfile):
    """
    Function to read the contents of a logfile back into a dictionary.
    Args:
        logfile (str): The path to the logfile.
    Returns:
        dict: The contents of the logfile as a dictionary.
    """
    with open(logfile, 'r') as f:
        contents = f.read()
        data = json.loads(contents)
    return data

##############################################################################################################################
# Other Utility functions for data generation and preprocessing
##############################################################################################################################

def full_factorial(l_bounds, u_bounds, n):
        if isinstance(n, int):
            n = [int(n**0.5), int(n**0.5)]
        x = np.linspace(l_bounds[0], u_bounds[0]-1, n[0], dtype=int)
        y = np.linspace(l_bounds[1], u_bounds[1]-1, n[1], dtype=int)

        return np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

def reset_random_seeds(seed=42):
    """
        Function to reset all random seeds for reproducibility.
        Args:
            seed (int): The seed value. Default is 42.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def limit_memory_growth():
    """
        Function to limit the memory growth of the GPU.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

###############################################################################################################################
# Utility functions for loss calculations
###############################################################################################################################

def MSE_zero(y, axis = None):
    return ops.mean(ops.square(y), axis = axis)

def MSE(y_true, y_pred, axis = None):
    return ops.mean(ops.square(y_true - y_pred), axis = axis)

def MAE(y_true, y_pred, axis = None):
    return ops.mean(ops.abs(y_true - y_pred), axis = axis)

def Huber(y_true, y_pred, delta=1.0, axis = -1):
    error = ops.subtract(y_pred, y_true)
    abs_error = ops.abs(error)
    half = ops.convert_to_tensor(0.5, dtype=abs_error.dtype)
    return ops.mean(
        ops.where(
            abs_error <= delta,
            half * ops.square(error),
            delta * abs_error - half * ops.square(delta),
        ),
        axis=axis,
    )