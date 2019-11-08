import numpy as np
from core.activation import get_activation_function, get_derivative_activation_function


class AntLayer:
    def __init__(self, units=None, input_dim=None):
        self.units = units
        self.input_dim = input_dim
        self.output_dim = None
        self.name = "L"
        self.input = None
        self.output = None

    def feedforward(self, input):
        raise NotImplementedError

    def backpropagate(self, output_error, learning_rate):
        raise NotImplementedError

    def create(self, prev_layer_dim):
        self.input_dim = prev_layer_dim
        self.output_dim = prev_layer_dim


class AntDense(AntLayer):

    def __init__(self, units=None, input_dim=None, activation=None):
        super().__init__(units=units, input_dim=input_dim)
        self.weights = np.array([])
        self.bias = np.array([])
        self.activation = activation

    def create(self, prev_layer_dim):

        self.input_dim = prev_layer_dim
        self.output_dim = self.units

        self.weights = np.random.rand(self.input_dim, self.output_dim) - 0.5
        self.bias = np.random.rand(1, self.output_dim) - 0.5

    def feedforward(self, input_data):
        self.input = input_data

        self.output = np.dot(self.input, self.weights) + self.bias

        if self.activation is not None:
            return get_activation_function(self.activation)(self.output)

        return self.output

    def backpropagate(self, output_error, learning_rate):

        #print(output_error.shape, self.input.shape)

        if self.activation is not None:
            output_error *= get_derivative_activation_function(self.activation)(self.output)

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

