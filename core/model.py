import numpy as np
from core.activation import get_loss_function, get_loss_derivative_function
from core.layer import AntLayer, AntDense


class AntNetModel:

    def __init__(self):
        self.layers = []
        self.loss = None

    def create_layers(self):

        last_dim = self.layers[0].input_dim

        for i, layer in enumerate(self.layers):
            layer.create(last_dim)

            last_dim = layer.output_dim

            layer.name = "L-" + str(len(self.layers) - 1 - i)

    def add(self, layer):

        assert isinstance(layer, AntLayer), "Param 'layer' must be of instance AntLayer"

        self.layers.append(layer)

    def compile(self, loss="mse", metrics="mse", optimizer="sgd"):
        self.loss = loss
        self.create_layers()

    def summary(self):

        print("_" * 60)
        print("Layer (type)".ljust(30) + "Output Shape".ljust(20) + "Param #".ljust(10))
        print("=" * 60)

        for i, layer in enumerate(self.layers):
            n_params = len(np.array(layer.weights).flatten()) + len(np.array(layer.bias).flatten())

            print((layer.name + " (" + layer.__class__.__name__ + ")").ljust(30) + str(layer.output_dim).ljust(20) + str(n_params).ljust(10))

            if len(self.layers) - 1 > i:
                print("-" * 60)

        print("=" * 60)

    def feedforward(self, inputs):

        output = inputs

        for layer in self.layers:
            output = layer.feedforward(output)

        return output

    def backpropagate(self, y_true, y_pred):

        learning_rate = 0.1
        error = get_loss_derivative_function(self.loss)(y_true, y_pred)

        for layer in reversed(self.layers):
            error = layer.backpropagate(error, learning_rate)

    def get_weights(self):

        weights = []

        for layer in self.layers:
            weights.append([layer.weights, layer.bias])

        return weights

    def train(self, x_train, y_train, epochs=100):

        for epoch in range(epochs):

            total_loss = 0

            for x, y in zip(x_train, y_train):

                x = np.array(x)
                if len(x.shape) < 2:
                    x = x.reshape(1, x.shape[0])

                y_hat = self.feedforward(x)

                total_loss += get_loss_function(self.loss)(y, y_hat)

                self.backpropagate(y, y_hat)

            print("Epoch {0}/{1} - loss: {2:.4f}".format(epoch + 1, epochs, total_loss / len(y_train)))

    def predict(self, x):
        return self.feedforward(np.array(x))

"""

x = [1.]
y = 2.
w_1 = 1.
b_1 = 1.
w_2 = 2.
b_2 = 2.
w_3 = 3.
b_3 = 3.
w_4 = -1.
b_4 = -1.
w_5 = -1.

ant = AntNetModel()
ant.add(AntDense(units=1, input_dim=1, activation="relu"))
ant.add(AntDense(units=2, activation="relu"))
ant.add(AntDense(units=1))

ant.compile(loss="mse")
ant.summary()

ant.get_weights()

ant.layers[0].weights = np.array([[w_1]])
ant.layers[0].bias = np.array([[b_1]])
ant.layers[1].weights = np.array([[w_2, w_3]])
ant.layers[1].bias = np.array([[b_2, b_3]])
ant.layers[2].weights = np.array([[w_4], [w_5]])
ant.layers[2].bias = np.array([[b_4]])

ant.feedforward(x)

ant.train([x], [y], epochs=5)
"""




