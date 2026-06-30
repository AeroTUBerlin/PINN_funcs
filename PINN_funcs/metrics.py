import keras
from keras import ops


class single_Value(keras.metrics.Metric):
    '''
    A custom metric to store and update a single value, such as a loss component or learning rate, during training.
    This metric allows for dynamic tracking and updating of a specific value throughout the training process.
    Attributes
    ----------
    value : tf.Variable
        A variable to store the single value being tracked.
    name : str
        The name of the metric.
    dtype : str
        The data type of the metric.
    Methods
    ----------
    update_state(value): Updates the value of the metric.
    result(): Returns the current value of the metric.
    reset_state(): Resets the value of the metric to zero.
    '''
    def __init__(self, name='Value', dtype=None):
        super(single_Value, self).__init__(name=name, dtype=dtype)
        self.value = self.add_variable(shape = (),
                                       name='value',
                                       initializer=keras.initializers.Zeros(),
                                       dtype=self.dtype)

    def update_state(self, value):
        self.value.assign(value)

    def result(self):
        return ops.cast(self.value, dtype = self.dtype)

    def reset_state(self):
        pass

class loss_weight_dict(keras.metrics.Metric):
    '''
    A custom metric to store and update loss weights for different components of the loss function.
    This metric is useful for dynamically adjusting the importance of different loss components during training.
    Attributes
    ----------
    loss_weights : dict
        A dictionary to store the loss weights for different components of the loss function.
        name : str
            The name of the metric.
        dtype : str         
            The data type of the metric.
            Methods
            ----------
            update_state(value_dict): Updates the loss weights based on the provided dictionary.
            result(): Returns the current loss weights.
            reset_state(): Ensures that the loss weights are not reset after each epoch.
    '''
    def __init__(self, name='lambda', dtype=None):
        super(loss_weight_dict, self).__init__(name=name, dtype=dtype)

        self.lambda_data = self.add_variable(shape = (),
                                           name='lambda_data',
                                           initializer='ones',
                                           dtype=self.dtype)
        self.lambda_BC = self.add_variable(shape = (),
                                           name='lambda_BC',
                                           initializer='ones',
                                           dtype=self.dtype)
        self.lambda_PDE = self.add_variable(shape = (),
                                           name='lambda_PDE',
                                           initializer='ones',
                                           dtype=self.dtype)
        
        self.loss_weights = {'lambda_data':self.lambda_data.value,
                            'lambda_BC':self.lambda_BC.value,
                            'lambda_PDE':self.lambda_PDE.value
                            }

    def update_state(self, value_dict):
        """
        Updates the loss weights based on the provided dictionary.
        Parameters
        ----------
        value_dict : dict
            A dictionary containing the new values for the loss weights.
            The keys should match the names of the loss weights in the metric.
        """
        if not isinstance(value_dict, dict):
            raise ValueError("value_dict should be a dictionary.")
        
        for key, value in value_dict.items():
            if key in self.loss_weights:
                self.loss_weights[key].assign(value)
            else:
                raise ValueError(f"Invalid loss weight key: {key}. Expected one of {list(self.loss_weights.keys())}.")

    def result(self):
        return self.loss_weights

    def reset_state(self):
        return 
