"""
https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0
https://medium.com/datadriveninvestor/math-neural-network-from-scratch-in-python-d6da9f29ce65

TODO:
- zaokruživanje decimali na 6-7 mjesta
- neuroni moraju imat ime: n1, n2...
- biasi moraju imat ime: b1, b2,...
- weightovi moraju imat ime: w1, w2, w3
- mogućnost ispisa formule cijele mreže
- mogućnost ispisa derivacija za izračunavanje gradijenta
- vizualizacija pomoću netflow.js-a -> click na neuron baca wan njegovu formulu i formulu za derivaciju

"""

# from .layer import Layer
import numpy as np


def sigmoid(Z):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1/(1+np.exp(-Z))


def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)


def relu(Z):
    return np.maximum(0, Z)


def mse(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()


def deriv_mse(y_true, y_pred):
  return - 2 * (y_true - y_pred)



class Neuron:

    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation
        #self.name_index = "L-1_2" # Means neuron in layer L-1 in the index 2
        self.name = "n_i" # n_1, n_2, ....

    @property
    def bias_name(self):
        return self.name.replace("n", "b")

    def feedforward(self, inputs):

        # Weight inputs, add bias, then use the activation function
        total = np.dot(self.weights, inputs) + self.bias

        return self.activation(total) if self.activation is not None else total, total


class Layer:

    def __init__(self, units=None, input_dim=None, activation=None):

        assert units is not None, "Param 'units' must be defined"

        self.units = units
        self.activation = sigmoid if activation is not None else None# TOOD from name
        self.neurons = []
        self.name = "L-0" # Will be L, L-1...

        self.input_dim = input_dim
        self.output_dim = self.units

    def create(self, prev_layer_dim):

        for u in range(self.units):

            weights = np.array([np.random.normal() for i in range(prev_layer_dim)])
            bias = np.random.normal()
            #bias = 0

            n = Neuron(weights=weights, bias=bias, activation=self.activation)

            self.neurons.append(n)

    def feedforward(self, inputs):

        output = []
        computed = []

        for neuron in self.neurons:

            a, z = neuron.feedforward(inputs)

            output.append(a)
            computed.append(z)

        return output, computed

    def backward_propagation(self):
        pass

"""
class Dense(Layer):

    def __init__(self, units):

        super().__init__()

        self.units = units
    
    def create(self):
       
        self.neurons = []
"""

class AntNetModel:

    def __init__(self):
        self.layers = [Layer(units=10)] #TMP
        self.layers = []

        self.loss = mse

    def create_layers(self):

        last_dim = self.layers[0].input_dim

        for i, layer in enumerate(self.layers):

            layer.create(last_dim)

            last_dim = layer.output_dim

            j = len(self.layers) - 1 - i
            layer.name = "L-" + str(j)
            #layer.name = "L" + ("-" + str(j) if j > 0 else "")

    def find_layer(self, name):

        for l in self.layers:
            if l.name == name:
                return l

        return None

    def add(self, layer):

        """assert is instance Layer"""

        self.layers.append(layer)

    def compile(self):

        """TODO add params loss, optimizer..."""

        self.create_layers()

    def summary(self):

        print("_"*60)
        print("Layer (type)".ljust(30) + "Output Shape".ljust(20) + "Param #".ljust(10))
        print("="*60)

        for i, layer in enumerate(self.layers):
            #        = number of weights                            + number of biases
            n_params = sum([len(n.weights) for n in layer.neurons]) + len(layer.neurons)

            print((layer.name + " (" + layer.__class__.__name__+")").ljust(30) + str(layer.output_dim).ljust(20) + str(n_params).ljust(10))

            if len(self.layers) - 1 > i:
                print("-" * 60)

        print("="*60)


    def feedforward(self, inputs):

        memory = {}
        output = inputs

        #memory["X"] = {"A": inputs, "Z": inputs}
        memory["L-"+str(len(self.layers))] = {"A": inputs, "Z": inputs}

        for layer in self.layers:

            output, computed = layer.feedforward(output)

            memory[layer.name] = {"A": output, "Z": computed}

        return output, memory


    def backpropagate(self, memory, y_true, y_pred):

        """
            svaki layer moren gledat izolirano i množit sa nekon varijablon koja će za 0-i sloj bit 1

            kako se svi neuroni layera završe, spremi se cost za d_aL_d_zL

        """

        lr = 0.1
        update_list = {}

        # --- Naming: d_L_d_wL represents "partial L / partial wL" ---

        # Drivation of loss function
        d_c_d_yhat = deriv_mse(y_true, y_pred)

        gradient_of_L = d_c_d_yhat

        for i in range(len(memory.keys()) - 1):

            layer_in_focus = "L-" + str(i)
            layer_before_focused = "L-" + str(i + 1)
            layer_after_focused = "L-" + str(i - 1)

            d_zLnext_d_aL = 1 if layer_in_focus == "L-0" else self.find_layer(layer_after_focused).neurons[0].weights[0]

            d_aL_d_zL = deriv_sigmoid(memory[layer_in_focus]["Z"][0])

            d_zL_d_wL = memory[layer_before_focused]["A"][0]

            d_c_d_wL = d_zL_d_wL * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L

            d_c_d_bL = 1 * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L


            gradient_of_L *= d_aL_d_zL * d_zLnext_d_aL


            update_list[layer_in_focus] =  {"w": d_c_d_wL, "b": d_c_d_bL}


        #print(update_list)

        for key in update_list.keys():
            self.find_layer(key).neurons[0].weights[0] -= lr * update_list[key]["w"]
            self.find_layer(key).neurons[0].bias -= lr * update_list[key]["b"]


    def get_weights(self):

        weights = []

        for layer in self.layers:
            for neuron in layer.neurons:
                weights.append([neuron.weights, neuron.bias])

        return weights

    def train(self, x_train, y_train, epochs=100):

        for epoch in range(epochs):

            total_loss = 0

            for x, y in zip(x_train, y_train):

                output, memory = self.feedforward(x)

                y_hat = np.array(output).flatten()[0]

                total_loss += mse(y, y_hat)

                self.backpropagate(memory, y, y_hat)

            print("Epoch {0}/{1} - loss: {2:.4f}".format(epoch+1, epochs, total_loss/len(y_train)))



#i1 = [i for i in range(1, 3 )]
#i2 = [i for i in reversed(range(1, 3))]
#y_i1, y_i2 = 1, 2

x = [1]
y = 2
w_1 = 1
b_1 = 1
w_2 = 2
b_2 = 2
w_3 = 3
b_3 = 3
w_4 = -1
b_4 = -1
w_5 = -1

ant = AntNetModel()
ant.add(Layer(units=1, input_dim=1, activation="sigmoid"))
ant.add(Layer(units=2, activation="sigmoid"))
ant.add(Layer(units=1, activation="sigmoid"))

ant.compile()
ant.summary()

ant.layers[0].neurons[0].weights[0] = w_1
ant.layers[0].neurons[0].bias = b_1

ant.layers[1].neurons[0].weights[0] = w_2
ant.layers[1].neurons[0].bias = b_2
ant.layers[1].neurons[1].weights[0] = w_3
ant.layers[1].neurons[1].bias = b_3

ant.layers[2].neurons[0].weights[0] = w_4
ant.layers[2].neurons[0].bias = b_4
ant.layers[2].neurons[0].weights[1] = w_5

ant.train([x], [y], epochs=1)
ant.get_weights()

ant.feedforward(x)

model.get_weights()


"""
    w1      w2      w4
x ---- n1 ---- n2 ----- n4 ----- y_hat
        |_____ n3______|
           w3      w5
"""

output, memory = ant.feedforward(x)
output[0], y
memory

output = output[0]
# ant.backpropagate(memory, y, output)

print("MSE:", mse(y, output))

lr = 0.1

d_c = deriv_mse(y, output)

"""
    d_c_d_w5 = d_z5_d_w5 * d_a5_d_z5 * d_c_d_a5
"""
d_a5_d_z5 = deriv_sigmoid(memory["L-0"]["Z"][0])

d_z5_d_w5 = memory["L-1"]["A"][1]

d_c_d_w5 = d_z5_d_w5 * d_a5_d_z5 * d_c
w_5 - lr * d_c_d_w5


"""
    d_c_d_w4 = d_z4_d_w4 * d_a4_d_z4 * d_c_d_a4
"""
d_a4_d_z4 = deriv_sigmoid(memory["L-0"]["Z"][0])

d_z4_d_w4 = memory["L-1"]["A"][0]

d_c_d_w4 = d_z4_d_w4 * d_a4_d_z4 * d_c
w_4 - lr * d_c_d_w4


"""
    d_c_d_b4 = d_z4_d_b4 * d_a4_d_z4 * d_c_d_a4
"""
d_z4_d_b4 = 1
d_c_d_b4 = d_z4_d_b4 * d_a4_d_z4 * d_c
b_4 - lr * d_c_d_b4


"""
    d_c_d_w3 = d_z3_d_w3 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c_d_a4
"""

d_z4_d_a3 = ant.find_layer("L-0").neurons[0].weights[1]

d_a3_d_z3 = deriv_sigmoid(memory["L-1"]["Z"][1])

d_z3_d_w3 = memory["L-2"]["A"][0]

d_c_d_w3 = d_z3_d_w3 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c
w_3 - lr * d_c_d_w3

## b_3
d_z3_d_b3 = 1
d_c_d_b3 = d_z3_d_b3 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c
b_3 - lr * d_c_d_b3


"""
    d_c_d_w2 = d_z2_d_w2 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c_d_a4
"""

d_z4_d_a3 = ant.find_layer("L-0").neurons[0].weights[0]

d_a2_d_z2 = deriv_sigmoid(memory["L-1"]["Z"][0])

d_z2_d_w2 = memory["L-2"]["A"][0]

d_c_d_w2 = d_z2_d_w2 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c
w_2 - lr * d_c_d_w2

## b_2
d_z2_d_b2 = 1
d_c_d_b2 = d_z2_d_b2 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c
b_2 - lr * d_c_d_b2


"""
    d_c_d_w1 = d_z1_d_w1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c
    d_c_d_w1 = d_z1_d_w1 * d_a1_d_z1 * d_z3_d_a1 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c
"""
d_z2_d_a1 = ant.find_layer("L-1").neurons[0].weights[0]

d_a1_d_z1 = deriv_sigmoid(memory["L-2"]["Z"][0])

d_z1_d_w1 = memory["L-3"]["A"][0]

d_c_d_w1_2 = d_z1_d_w1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c

d_z3_d_a1 = ant.find_layer("L-1").neurons[1].weights[0]

d_c_d_w1_3 = d_z1_d_w1 * d_a1_d_z1 * d_z3_d_a1 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c

d_c_d_w1 = d_c_d_w1_2 + d_c_d_w1_3

w_1 - lr * d_c_d_w1

## b_1

d_z1_d_b1 = 1

d_c_d_b1 = d_z1_d_b1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z4_d_a3 * d_a4_d_z4 * d_c + d_z1_d_b1 * d_a1_d_z1 * d_z3_d_a1 * d_a3_d_z3 * d_z4_d_a3 * d_a4_d_z4 * d_c



w_1-d_c_d_w1*lr, b_1-d_c_d_b1*lr,  w_2-d_c_d_w2*lr, b_2-d_c_d_b2*lr, w_3-d_c_d_w3*lr, b_3-d_c_d_b3*lr, w_4-d_c_d_w4*lr, b_4-d_c_d_b4*lr, w_5-d_c_d_w5*lr

model.get_weights()


# Update
ant.layers[0].neurons[0].weights[0] -= lr * d_c_d_w1
ant.layers[0].neurons[0].bias -= lr * d_c_d_b1

ant.layers[1].neurons[0].weights[0] -= lr * d_c_d_w2
ant.layers[1].neurons[0].bias -= lr * d_c_d_b2
ant.layers[1].neurons[1].weights[0] -= lr * d_c_d_w3
ant.layers[1].neurons[1].bias -= lr * d_c_d_b3

ant.layers[2].neurons[0].weights[0] -= lr * d_c_d_w4
ant.layers[2].neurons[0].bias -= lr * d_c_d_b4
ant.layers[2].neurons[0].weights[1] -= lr * d_c_d_w5


ant.feedforward(x)
model.predict(np.array(x).reshape(1, 1))

ant.get_weights()
model.get_weights()




d_c_d_yhat = deriv_mse(y, output)

gradient_of_L = [d_c_d_yhat]
lr = 0.1

update_list = {}

for i in range(len(memory.keys()) - 1):

    layer_in_focus = "L-" + str(i)
    layer_before_focused = "L-" + str(i + 1)
    layer_after_focused = "L-" + str(i - 1)

    d_zLnext_d_aL = 1 if layer_in_focus == "L-0" else ant.find_layer(layer_after_focused).neurons[0].weights[0]

    d_aL_d_zL = deriv_sigmoid(memory[layer_in_focus]["Z"][0])

    d_zL_d_wL = memory[layer_before_focused]["A"][0]

    d_c_d_wL = d_zL_d_wL * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L[0]

    d_c_d_bL = 1 * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L[0]


    gradient_of_L[0] *= d_aL_d_zL * d_zLnext_d_aL


    update_list[layer_in_focus] =  {"w": d_c_d_wL, "b": d_c_d_bL}


print(update_list)


gradient_of_L = [d_c_d_yhat]

for i in range(len(memory.keys()) - 1):

    i = 0
    i = 1

    layer_in_focus = "L-" + str(i)
    layer_before_focused = "L-" + str(i + 1)
    layer_after_focused = "L-" + str(i - 1)

    new_gradient_of_L = []

    for j, neuron in enumerate(ant.find_layer(layer_in_focus).neurons):

        j = 0
        len(ant.find_layer(layer_in_focus).neurons[j].weights)
        len(ant.find_layer(layer_in_focus).neurons)

        # TODO j=1

        tmp_gradient = 1
        for k, _ in enumerate(ant.find_layer(layer_before_focused).neurons):

            len(ant.find_layer(layer_before_focused).neurons)
            k = 0
            k = 1

            d_zLnext_d_aL = 1 if layer_in_focus == "L-0" else ant.find_layer(layer_after_focused).neurons[k].weights[j]

            d_aL_d_zL = deriv_sigmoid(memory[layer_in_focus]["Z"][j])

            d_zL_d_wL = memory[layer_before_focused]["A"][k]

            d_c_d_wL = d_zL_d_wL * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L[j]

            d_c_d_bL = 1 * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L[j]

            #gradient_of_L[0] *= d_aL_d_zL * d_zLnext_d_aL

            #print(d_c_d_wL, d_c_d_bL)

            w_5 - lr*d_c_d_wL, b_4 - lr * d_c_d_bL # -0.98235549, -0.98194530911
            w_4 - lr*d_c_d_wL, b_4 - lr * d_c_d_bL # -0.98200908, -0.98194530911

            w_2 - lr*d_c_d_wL, b_2 - lr * d_c_d_bL # 1.99964693, 1.9995991426901973
            w_3 - lr*d_c_d_wL, b_3 - lr * d_c_d_bL # 2.99994403, 2.9999364584929626

            tmp_gradient *= gradient_of_L[j] * d_aL_d_zL * d_zLnext_d_aL

        new_gradient_of_L.append(tmp_gradient)

    gradient_of_L = new_gradient_of_L





### mix

x = [1]
y = 2
w_1 = 1
b_1 = 1
w_2 = 2
b_2 = 2
w_3 = 3
b_3 = 3
w_4 = -1
b_4 = -1
w_5 = -1

model = Sequential()
model.add(Dense(1, input_dim=1, use_bias=True, activation="sigmoid"))
model.add(Dense(2, use_bias=True, activation="sigmoid"))
model.add(Dense(1, use_bias=True, activation="sigmoid"))
model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['mse'])
model.summary()

model.set_weights([ [[w_1]], [b_1], [[w_2, w_3]], [b_2, b_3], [[w_4], [w_5]], [b_4] ])
model.get_weights()
model.predict(np.array(x).reshape(1, 1))

#model.fit(np.array([x]), [y], epochs=10000, verbose=1)

model.fit(np.array([x]), [y], epochs=1, verbose=0)
model.get_weights()
model.predict(np.array(x).reshape(1, 1))









from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import sigmoid
from tensorflow.python.keras.optimizers import Adam, SGD


i1 = [i for i in range(1, 3 )]
i2 = [i for i in reversed(range(1, 3))]

y_i1 = 1
y_i2 = 2


## Print values for layer

from keras import backend as K
def do_bkwd(model, input_val):
    inp = model.input  # input placeholder
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    # Testing
    test = np.array(input_val).reshape(1, 2)
    # layer_outs = [func([test, 1]) for func in functors]

    for func in functors:
        print("-"*20)
        print(func([test, 1]))



















model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

model.summary()

model.fit(np.array([i1, i2]), [y_i1, y_i2], epochs=1000)

model.predict(np.array(i1).reshape(1, 2))
model.predict(np.array(i2).reshape(1, 2))


ant = AntNetModel()
ant.add(Layer(units=1, input_dim=2))
ant.add(Layer(units=1))

ant.compile()
ant.summary()

ant.layers[0].neurons[0].weights = model.layers[0].weights[0].numpy().flatten()
ant.layers[0].neurons[0].bias = model.layers[0].weights[1].numpy()[0]

ant.layers[1].neurons[0].weights = model.layers[1].weights[0].numpy().flatten()
ant.layers[1].neurons[0].bias = model.layers[1].weights[1].numpy()[0]

ant.feedforward(i1)
ant.feedforward(i2)



#### 2 neurona


model = Sequential()
model.add(Dense(2, input_dim=2))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])

model.summary()

model.fit(np.array([i1, i2]), [y_i1, y_i2], epochs=1000)

model.predict(np.array(i1).reshape(1, 2))
model.predict(np.array(i2).reshape(1, 2))


ant = AntNetModel()
ant.add(Layer(units=2, input_dim=2))
ant.add(Layer(units=1))

ant.compile()
ant.summary()

## L-1
ant.layers[0].neurons[0].weights = model.layers[0].weights[0][:, 0].numpy()
ant.layers[0].neurons[1].weights = model.layers[0].weights[0][:, 1].numpy()

ant.layers[0].neurons[0].bias = model.layers[0].weights[1][0].numpy()
ant.layers[0].neurons[1].bias = model.layers[0].weights[1][1].numpy()


# L
ant.layers[1].neurons[0].weights = model.layers[1].weights[0].numpy().flatten()
ant.layers[1].neurons[0].bias = model.layers[1].weights[1].numpy()[0]

ant.feedforward(i1)
do_bkwd(model, i1)
ant.feedforward(i2)
do_bkwd(model, i2)





def apply_weights(tf_model, ant_model):

    for i in range(len(ant_model.layers)):

        for n in range(len(ant.layers[i].neurons)):
            ant_model.layers[i].neurons[n].weights = tf_model.layers[i].weights[0][:, n].numpy()
            ant_model.layers[i].neurons[n].bias = tf_model.layers[i].weights[1][n].numpy()






model = Sequential()
model.add(Dense(10, input_dim=2, activation="sigmoid"))
model.add(Dense(20, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(20, activation="sigmoid"))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam', metrics=['mae'])
model.summary()
model.fit(np.array([i1, i2]), [y_i1, y_i2], epochs=1000)

model.predict(np.array(i1).reshape(1, 2))
model.predict(np.array(i2).reshape(1, 2))


ant = AntNetModel()
ant.add(Layer(units=10, input_dim=2, activation="sigmoid"))
ant.add(Layer(units=20, activation="sigmoid"))
ant.add(Layer(units=10, activation="sigmoid"))
ant.add(Layer(units=20, activation="sigmoid"))
ant.add(Layer(units=1))

ant.compile()
ant.summary()

apply_weights(model, ant)

ant.feedforward(i1)
ant.feedforward(i2)











### Super-simple example

x = [1]
y = 1
w = 2

model = Sequential()
model.add(Dense(1, input_dim=1, use_bias=False))
model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['mse'])
model.summary()

model.set_weights([[[w]]])
model.fit(np.array([x]), [y], epochs=1)
model.predict(np.array(x).reshape(1, 1))
model.get_weights()


### Super-simple example 2
x = [1]
y = 1
w_1 = 2
w_2 = 3

model = Sequential()
model.add(Dense(1, input_dim=1, use_bias=False))
model.add(Dense(1, use_bias=False))
model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['mse'])
model.summary()

model.set_weights([[[w_1]], [[w_2]]])
model.fit(np.array([x]), [y], epochs=1)
model.predict(np.array(x).reshape(1, 1))
model.get_weights()

model.fit(np.array([x]), [y], epochs=1)
model.predict(np.array(x).reshape(1, 1))
model.get_weights()



### mix

x = [1]
y = 2
w_1 = 1
b_1 = 1
w_2 = 2
b_2 = 2
w_3 = -1
b_3 = -1

model = Sequential()
model.add(Dense(1, input_dim=1, use_bias=True, activation="sigmoid"))
model.add(Dense(1, use_bias=True, activation="sigmoid"))
model.add(Dense(1, use_bias=True, activation="sigmoid"))
model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['mse'])
model.summary()

model.set_weights([[[w_1]], [b_1], [[w_2]], [b_2], [[w_3]], [b_3]])
model.get_weights()
model.predict(np.array(x).reshape(1, 1))

# model.fit(np.array([x]), [y], epochs=1000, verbose=1)

model.fit(np.array([x]), [y], epochs=1, verbose=0)
model.get_weights()
model.predict(np.array(x).reshape(1, 1))






##### OLD MANUAL SGD
for klmn in range(100000):

    output, memory = ant.feedforward(x)
    output[0], y
    memory

    output = output[0]
    # ant.backpropagate(memory, y, output)

    print("MSE:", mse(y, output))

    lr = 0.1

    d_c = deriv_mse(y, output)

    """
        d_c_d_w3 = d_z3_d_w3 * d_a3_d_z3 * d_c_d_a3
    """
    d_a3_d_z3 = deriv_sigmoid(memory["L-0"]["Z"][0])

    d_z3_d_w3 = memory["L-1"]["A"][0]

    d_c_d_w3 = d_z3_d_w3 * d_a3_d_z3 * d_c

    """
        d_c_d_b3 = d_z3_d_b3 * d_a3_d_z3 * d_c_d_a3
    """
    d_z3_d_b3 = 1
    d_c_d_b3 = d_z3_d_b3 * d_a3_d_z3 * d_c


    """
        d_c_d_w2 = d_z2_d_w2 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c_d_a3
    """

    d_z3_d_a2 = ant.find_layer("L-0").neurons[0].weights[0]

    d_a2_d_z2 = deriv_sigmoid(memory["L-1"]["Z"][0])

    d_z2_d_w2 = memory["L-2"]["A"][0]

    d_c_d_w2 = d_z2_d_w2 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c


    ## b_2
    d_z2_d_b2 = 1

    d_c_d_b2 = d_z2_d_b2 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c


    """
        d_c_d_w1 = d_z1_d_w1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c_d_a3
    """
    d_z2_d_a1 = ant.find_layer("L-1").neurons[0].weights[0]

    d_a1_d_z1 = deriv_sigmoid(memory["L-2"]["Z"][0])

    d_z1_d_w1 = memory["L-3"]["A"][0]

    d_c_d_w1 = d_z1_d_w1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c


    ## b_1

    d_z1_d_b1 = 1

    d_c_d_b1 = d_z1_d_b1 * d_a1_d_z1 * d_z2_d_a1 * d_a2_d_z2 * d_z3_d_a2 * d_a3_d_z3 * d_c



    gradients = 1-d_c_d_w1*lr, 1-d_c_d_w2*lr, 2-d_c_d_w2*lr, 2-d_c_d_b2*lr, -1-d_c_d_w3*lr, -1-d_c_d_b3*lr



    # Update
    ant.layers[1].neurons[0].weights[0] -= lr * d_c_d_w2
    ant.layers[1].neurons[0].bias -= lr * d_c_d_b2

    ant.layers[0].neurons[0].weights[0] -= lr * d_c_d_w1
    ant.layers[0].neurons[0].bias -= lr * d_c_d_b1






d_c_d_yhat = deriv_mse(y, output)

gradient_of_L = d_c_d_yhat
lr = 0.1

for i in range(len(memory.keys())-1):

    layer_in_focus = "L-"+str(i)
    layer_before_focused = "L-"+str(i+1)
    layer_after_focused = "L-"+str(i-1)


    d_zLnext_d_aL = 1 if layer_in_focus == "L-0" else ant.find_layer(layer_after_focused).neurons[0].weights[0]

    d_aL_d_zL = deriv_sigmoid(memory[layer_in_focus]["Z"][0])

    d_zL_d_wL = memory[layer_before_focused]["A"][0]

    d_c_d_wL = d_zL_d_wL * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L

    d_c_d_bL = 1 * d_aL_d_zL * d_zLnext_d_aL * gradient_of_L


    gradient_of_L *= d_aL_d_zL * d_zLnext_d_aL

    #print(d_c_d_wL, d_c_d_bL)

    if i == 0:
        print(-1-lr*d_c_d_wL, -1-lr*d_c_d_bL)
    elif i==1:
        print(2-lr*d_c_d_wL, 2-lr*d_c_d_bL)
    else:
        print(1-lr*d_c_d_wL, 1-lr*d_c_d_bL)

    gradients


