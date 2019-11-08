import numpy as np


def get_activation_function(name):
    return globals()[name]


def get_derivative_activation_function(name):
    return globals()[name + "_derivative"]


def get_loss_function(name):
    return globals()[name]


def get_loss_derivative_function(name):
    return globals()[name + "_derivative"]


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(x):
    return np.where(x <= 0, 0, 1)
    #x[x <= 0] = 0
    #x[x > 0] = 1
    #return x

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


def mse_derivative(y_true, y_pred):
    return - 2 * (y_true - y_pred)


