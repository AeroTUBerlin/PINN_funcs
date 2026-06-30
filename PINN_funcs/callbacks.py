import logging
import os

import keras
import tensorflow as tf
from keras import ops

from .utils import print_metrics

logger = logging.getLogger(__name__)


class custom_callback(keras.callbacks.Callback):
    """
    Custom callback to log training metrics, evaluate the model, and save weights at specified frequencies.
    Args:
        log_dir (str): Directory to save logs.
        print_freq (int): Frequency (in epochs) to print training metrics.
        eval_freq (int): Frequency (in epochs) to evaluate the model.
        save_freq (int): Frequency (in epochs) to save model weights.
    """
    def __init__(self, log_dir = "logs",
                 print_freq = 10, eval_freq = 100, save_freq = 500):
        super().__init__()
        self.log_dir = str(log_dir)
        self.writer = tf.summary.create_file_writer(os.path.join(self.log_dir, 'train'))

        self.print_freq = print_freq
        self.eval_freq = eval_freq
        self.save_freq = save_freq

    @staticmethod
    def _print_metrics(epoch, logs):
        if not logs:
            return

        print('Epoch: {} Loss: {:1.3e}, Learning Rate: {:1.1e}'.format(epoch, logs['loss'], logs['learning_rate']))
        for key, value in logs.items():
            if key.startswith('loss_'):
                print('{}: {:1.2e}'.format(key, value), end=", ")
        print("\n")

    def _eval_model(self, epoch):
        metrics = self.model.val_step(self.model.validation_data)
        print_metrics(metrics, tb_writer=self.writer, epoch=epoch)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            return

        if self.print_freq and epoch % self.print_freq == 0:
            self._print_metrics(epoch, logs)

        if self.eval_freq and epoch % self.eval_freq == 0:
            self._eval_model(epoch)

        if self.save_freq and epoch % self.save_freq == 0:
            self.model.save_weights(self.log_dir + '/ep' + str(epoch) + '.weights.h5')


class tb_custom(keras.callbacks.TensorBoard):
    """
    TensorBoard callback with adjustable recording frequency. (Can increase performance, if epochs are very short)

    Args:
        record_freq (int): Frequency (in epochs) to record logs.
    """
    def __init__(self, record_freq = 1, **kwargs):
        super().__init__(**kwargs)
        self.record_freq = record_freq

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.record_freq == 0:
            super().on_epoch_end(epoch, logs)


class cb_calfac(custom_callback):
    """
    Callback to adjust calibration factor during training based on logging callback.
    Args:
        log_dir (str): Directory to save logs.
        print_freq (int): Frequency (in epochs) to print training metrics.
        eval_freq (int): Frequency (in epochs) to evaluate the model.
        save_freq (int): Frequency (in epochs) to save model weights.
        cal_freq (int): Frequency (in epochs) to update the calibration factor.
    """
    def __init__(self, log_dir = "logs",
                 print_freq = 10, eval_freq = 100, save_freq = 500, cal_freq=None):
        super().__init__(log_dir, print_freq, eval_freq, save_freq)

        self.cal_freq = cal_freq
        self.calexp_old = 0.0

    @staticmethod
    def _print_metrics(epoch, logs):
        if not logs:
            return

        print('Epoch: {} Loss: {:1.3e}, Learning Rate: {:1.1e}'.format(
            epoch, logs['loss'], logs['learning_rate']))
        for key, value in logs.items():
            if key.startswith('cal_fac'):
                print('{}: {:6.3f}'.format(key, value), end=", ")
            elif key.startswith('loss_'):
                print('{}: {:1.2e}'.format(key, value), end=", ")
        print("\n")

    def on_epoch_end(self, epoch, logs=None):

        super(cb_calfac, self).on_epoch_end(epoch, logs)
        if self.cal_freq and epoch % self.cal_freq == 0:
            new_fac = (self.model.initial_guess + logs['cal_fac']) / 2
            self.model.initial_guess.assign(new_fac)
            logger.info("Set initial guess to: %6.3f", new_fac)

class Autoweight_cb(keras.callbacks.Callback):
    """
    Callback to automatically adjust loss weights based on gradient norms.
    Args:
        update_freq (int): Frequency (in epochs) to update the loss weights.
        alpha (float): Smoothing factor for updating the weights.
    """

    def __init__(self, update_freq=100, alpha = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.update_freq = update_freq
        self.alpha = alpha

    def on_epoch_end(self, epoch, logs=None):
        if self.update_freq > 0 and epoch % self.update_freq == 0 and epoch > 0:
            self.model._build_idx()  # Ensure idx is built
            grad_BC = self.model._get_gradient_BC()
            grad_data = self.model._get_gradient_data()
            grad_PDE = self.model._get_gradient_PDE()

            norm_BC = ops.linalg.norm(grad_BC)
            norm_data = ops.linalg.norm(grad_data)
            norm_PDE = ops.linalg.norm(grad_PDE)

            mean_norm = ops.mean([norm_BC, norm_data, norm_PDE])

            lambda_BC = mean_norm / norm_BC
            lambda_PDE = mean_norm / norm_PDE
            lambda_data = mean_norm / norm_data

            lambda_BC_new = (1 - self.alpha) * self.model.loss_weights.lambda_BC + self.alpha * lambda_BC
            lambda_PDE_new = (1 - self.alpha) * self.model.loss_weights.lambda_PDE + self.alpha * lambda_PDE
            lambda_data_new = (1 - self.alpha) * self.model.loss_weights.lambda_data + self.alpha * lambda_data

            loss_weights_new = {
                'lambda_BC': lambda_BC_new,
                'lambda_PDE':  lambda_PDE_new,
                'lambda_data': lambda_data_new
            }

            self.model.loss_weights.update_state(loss_weights_new)

            logger.info(
                "Updated loss weights: BC = %.4e, PDE = %.4e, data = %.4e",
                lambda_BC_new, lambda_PDE_new, lambda_data_new,
            )
