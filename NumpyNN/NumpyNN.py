import _pickle
import copy
import random
import time
from abc import abstractmethod, ABC
from typing import Union
import cProfile

import numpy as np

from idoAi.AIMath import SigmoidDerivative, Sigmoid, Relu, ReluDerivative, Softmax, SoftmaxDerivative, Softplus, \
    SoftplusDerivative


# shape is by definition (y,x) or (outputs, inputs) or (rows, columns)


def NetTestConnect3():
    conv1 = Convolution2D(np.array([[[1, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 0, 0, 1]]]), activation_method="relu")
    conv2 = Convolution2D(np.array([[[1, 1, 1, 1, 1]]]), activation_method="relu")
    conv3 = Convolution2D(np.array([[[1],
                                     [1],
                                     [1],
                                     [1],
                                     [1]]]), activation_method="relu")
    conv4 = Convolution2D(np.array([[[0, 0, 0, 0, 1],
                                     [0, 0, 0, 1, 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 1, 0, 0, 0],
                                     [1, 0, 0, 0, 0]]]), activation_method="relu")
    net = NN([
        Dense2D((6, 7)),
        CombinationFlattenLayer([
            conv1,
            conv2,
            conv3,
            conv4
        ]),
        # Flatten(),
        # Dropout(0.1),
        # Shapen((4, 6, 7))
        Dense(168, activation_method="relu"),
        Dense(168, activation_method="relu"),
        Dense(168, activation_method="relu"),
        Dense(168, activation_method="relu"),
        Dropout(0.1),
        Dense(7, activation_method="softmax")
    ])
    prediction = net.predict(1 * np.array([[0, 0, 0, 0, 0, 0, 0],
                                           [1, 0, 0, 0, 0, 0, 0],
                                           [1, 1, 0, 0, 1, 0, 0],
                                           [1, 1, 0, 0, 1, 0, 0],
                                           [1, 0, 1, 1, 0, 0, 1],
                                           [1, 1, 0, 1, 1, 0, 1]]))
    print(prediction)


def NetTestOdd():
    pre_net = loadFromPickle("netodd.pickle")
    if pre_net is not None:
        net = pre_net
        print("Loaded previous net")
    else:
        net = NN([
            BasicLayer(16),
            Dense(4, activation_method="relu"),
            Dropout(0.1),
            Dense(1, activation_method="sigmoid")
        ])

    def convertTo16BitArray(num: int):
        result = [0 for _ in range(16)]
        binary_str = bin(num)
        # has two first chars for type and the rest are 1 or 0, e.g 5 = '0b101'
        binary_number_len = len(binary_str) - 2
        count = 1
        for binary_digit in binary_str[2:]:
            result[binary_number_len - count] = int(binary_digit)
            count += 1
        return result

    for _ in range(30_000):
        number = round(random.random() * 32768)
        numBin = convertTo16BitArray(number)
        net.train(numBin, [number % 2])
        net.mutate()

    for _ in range(10):
        number = round(random.random() * 32768)
        print(f"[{number}] = {net.predict(convertTo16BitArray(number))}")
    net.saveAsPickle("netodd.pickle")


def NetTestXor():
    print("Xor Net Test")
    pre_net = loadFromPickle("netxor.pickle")
    if pre_net is not None:
        net = pre_net
        print("Loaded previous netxor")
    else:
        net = NN([
            BasicLayer(2),
            Dense(16, activation_method="relu"),
            Dropout(0.1),
            Dense(1, activation_method="sigmoid")
        ])

    for _ in range(50_000):
        # put = np.random.randint(0, 2, (2,))  # slowest
        # put = [np.random.choice([0, 1]), np.random.choice([0, 1])]  # slower
        put = [round(random.random()), round(random.random())]  # good speed
        net.train(put, [put[0] ^ put[1]])

        if _ % 10:
            net.mutate()

    print("1,0 = " + str(net.predict([1, 0])))
    print("0,0 = " + str(net.predict([0, 0])))
    print("1,1 = " + str(net.predict([1, 1])))
    print("0,1 = " + str(net.predict([0, 1])))
    net.saveAsPickle("netxor.pickle")
    print("netxor Saved")


class BasicLayer:
    activation: np.ndarray
    bias: np.ndarray
    bias_change: np.ndarray

    input_weights: np.ndarray
    input_weights_change: np.ndarray

    shape: tuple

    @abstractmethod
    def __init__(self, node_shape: Union[int, tuple], input_shape: int = None, activation_method: str = "relu"):
        if activation_method == "sigmoid":
            self.activation_method = Sigmoid
            self.activation_derivative_method = SigmoidDerivative
        elif activation_method == "relu":
            self.activation_method = Relu
            self.activation_derivative_method = ReluDerivative
        elif activation_method == "softmax":
            self.activation_method = Softmax
            self.activation_derivative_method = SoftmaxDerivative
        elif activation_method == "softplus":
            self.activation_method = Softplus
            self.activation_derivative_method = SoftplusDerivative
        else:
            raise Exception("activation method not supported - supports [sigmoid, relu]")

        self.shape = (node_shape, input_shape)

        self.pre_activation = []

        # an array of floats each one represent the activation of a node
        self.activation = np.NAN
        self.activation_change = []

        # an array of floats each one represent the bias of a node
        self.bias = np.NAN
        self.bias = np.random.randint(-100, 100, node_shape) / 100
        self.bias_change = np.NAN

        # an array of arrays, each one represent the a node weights
        self.input_weights = np.NAN
        if input_shape:
            self.input_weights = np.random.randint(-100, 100, self.shape) / 100
        self.input_weights_change = np.NAN

        self.state = {}

    def predict(self, input_array):
        self.pre_activation = np.dot(self.input_weights, input_array) + self.bias
        # self.pre_activation = np.sum(self.input_weights * input_array, 1) + self.bias  # slower

        self.activation = self.activation_method(self.pre_activation)
        # self.activation = np.vectorize(self.activation_method)(self.pre_activation)  # much slower

        return self.activation

    def getInputShape(self):
        return self.shape[1]

    def getOutputShape(self):
        return self.shape[0]

    def setInputCount(self, input_count: int):
        self.shape = (self.shape[0], input_count)

    @abstractmethod
    def isTrainable(self):
        return True

    @abstractmethod
    def saveState(self, key):
        self.state[key] = copy.deepcopy(self)

    @abstractmethod
    def clearSavedStates(self):
        self.state.clear()

    def mutate(self, mutation_coefficient=0.1):
        if self.bias_change is not np.NAN:
            self.bias += self.bias_change * mutation_coefficient
        if self.input_weights_change is not np.NAN:
            self.input_weights += self.input_weights_change * mutation_coefficient

    def __str__(self):
        return "BasicLayer shape = " + str(self.shape) + ", weights = " + str(self.input_weights) + ", bias = " + str(
            self.bias)


class WeightedLayer(BasicLayer):

    @abstractmethod
    def isTrainable(self):
        pass

    def setInputCount(self, input_count: int):
        super(WeightedLayer, self).setInputCount(input_count)
        self.input_weights = np.random.randint(-100, 100, self.shape) / 100


class Dense(WeightedLayer):
    def __init__(self, node_shape: int, input_shape=0, **kwargs):
        super(Dense, self).__init__(node_shape, input_shape, **kwargs)
        pass

    def isTrainable(self):
        return True

    def __str__(self):
        return "Dense shape = " + str(self.shape) + ", weights = " + str(self.input_weights) + ", bias = " + str(
            self.bias)


class Dropout(BasicLayer):
    """ A layer that takes the input and drop some of them,
    configured by dropout_chance as probability """

    def __init__(self, dropout_chance: float = 0.2, **kwargs):
        """ A layer that takes the input and drop some of them,
        configured by dropout_chance as probability

        :param dropout_chance: the chance to drop a node, between 0 and 1
        :param kwargs: BasicLayer kwargs
        """
        assert 0 <= dropout_chance <= 1, "dropout_chance not between 0 and 1 , dropout_chance=" + str(dropout_chance)
        super(Dropout, self).__init__(0, **kwargs)
        self.dropout_chance = dropout_chance
        self.dropout_array = np.nan

    def predict(self, input_array: np.ndarray):
        self.shape = (input_array.shape[0], input_array.shape[0])
        self.dropout_array = []
        for _ in range(self.shape[0]):
            self.dropout_array.append(1 if random.random() / self.dropout_chance >= 1 else 0)
        self.dropout_array = np.array(self.dropout_array)
        # self.dropout_array = np.random.randint(0, 1 / self.dropout_chance, input_array.shape)
        # self.dropout_array = np.where(self.dropout_array >= 1, 1, 0)
        self.activation = input_array * self.dropout_array
        return self.activation

    def setInputCount(self, input_shape: int):
        self.shape = (input_shape, input_shape)

    def isTrainable(self):
        return False

    def __str__(self):
        return "Dropout shape = " + str(self.shape)


class CombinationFlattenLayer(BasicLayer):
    layers: Union[np.ndarray, list]

    def __init__(self, layers):
        # TODO how does this layer handle output shape to other layers? its a list of predictions not a regular prediction shape
        # TODO and moreover the layers can output different shaped outputs
        super(CombinationFlattenLayer, self).__init__(0)
        self.layers = layers
        # TODO

    def setInputCount(self, input_shape):
        output_count = 0
        for layer in self.layers:
            layer.setInputCount(input_shape)
            layer_output_shape = layer.shape[0]

            if isinstance(layer_output_shape, int):
                output_count *= layer_output_shape

            if isinstance(layer_output_shape, tuple):
                layer_flatten_shape = 1
                for layer_output_count in layer_output_shape:
                    layer_flatten_shape *= layer_output_count
                output_count += layer_flatten_shape
        self.shape = (output_count, input_shape)

    def predict(self, input_array: np.ndarray):
        result = []
        for layer in self.layers:
            layer: Union[BasicLayer, NN]
            result.append(layer.predict(input_array))
        self.activation = np.array(result).reshape(self.shape[0])
        return self.activation

    def __str__(self):
        return "Combination layers = " + str(self.layers)

    def isTrainable(self):
        return False  # need to check other layers for this


class Flatten(BasicLayer):
    def __init__(self):
        super(Flatten, self).__init__(0)

    def setInputCount(self, input_shape: set):
        output_count = 1
        for size in input_shape:
            output_count *= size
        self.shape = (output_count, input_shape)

    def predict(self, input_array: np.ndarray):
        self.activation = input_array.reshape((self.shape[0],))
        return self.activation

    def __str__(self):
        return "Flatten shape = " + str(self.shape)

    def isTrainable(self):
        return False


class Shapen(BasicLayer):
    def __init__(self, node_shape: tuple):
        super(Shapen, self).__init__(node_shape)

    def setInputCount(self, input_shape: set):
        self.shape = (self.shape[0], input_shape)

    def predict(self, input_array: np.ndarray):
        self.activation = input_array.reshape(self.shape[0])
        return self.activation

    def __str__(self):
        return "Shapen"

    def isTrainable(self):
        return False


class Basic2DLayer(BasicLayer, ABC):
    def __init__(self, *args, **kwargs):
        super(Basic2DLayer, self).__init__(*args, **kwargs)


class Dense2D(WeightedLayer):

    def __init__(self, node_shape: tuple, input_shape: int = None, activation_method: str = "relu"):
        super(Dense2D, self).__init__(node_shape, activation_method=activation_method)

    def setInputCount(self, input_shape: int):
        super(Dense2D, self).setInputCount(input_shape)

    def isTrainable(self):
        return True


class Convolution2D(Basic2DLayer):
    activation: np.ndarray
    position_velocity: tuple
    filters: np.ndarray

    def __init__(self, filters: np.ndarray, stride: tuple = (1, 1), **kwargs):
        """
        A filter is a matching map that is trying to fit on a particular place,
        like trying to match a triangle to an image like the filter below
        [[ 0 0 1 0 0 ],
         [ 0 1 0 1 0 ],
         [ 1 1 1 1 1 ]]
        This class takes all the filters, checks them on every possible position, and returns the sums
        :param node_shape:
        :param filters: a list of filters, need to be 3D , each filter is 2D so a list of 2D is 3D
                        The filters must be in the same size, if not make a CombinedLayer with two Convolution2D
        :param input_shape:
        :param kwargs:
        """
        super(Convolution2D, self).__init__(0, 0, **kwargs)
        if "activation_method" not in kwargs:
            self.activation_method = Relu
        if not isinstance(filters, np.ndarray):
            assert isinstance(filters, list), "filters must be a 3D list or 3D np.ndarray as a list of filters"
            filters = np.array(filters)
        filter1 = filters[0]
        for filter2 in filters[1:]:
            assert filter1.shape == filter2.shape, "all filters must be in the same size , " \
                                                   + str(filter1.shape) + " != " + str(filter2.shape)
        self.filters = filters
        self.shape = (filter1.shape, filter1.shape)
        self.stride = stride

    def predict(self, input_array: np.ndarray):
        self.activation = np.zeros(input_array.shape)

        filter_width = self.filters[0].shape[1]
        filter_height = self.filters[0].shape[0]
        padded_input_array = np.pad(input_array, ((int(filter_height / 2),), (int(filter_width / 2),)))

        for y_index in range(0, input_array.shape[0], self.stride[0]):
            for x_index in range(0, input_array.shape[1], self.stride[1]):
                input_slice = padded_input_array[y_index:y_index + filter_height,
                              x_index:x_index + filter_width]
                result = np.sum(input_slice * self.filters)
                self.activation[y_index][x_index] = result

        self.activation = self.activation_method(self.activation)
        return self.activation

    """
    Trains the filter_index filter
    Good use will be as an individual layer and train to desired outputs
    This is more complex but more accurate then hard-coded filters 
    """
    def trainFilter(self, input_array, filter_index, desired_activation):
        # TODO
        pass

    def setInputCount(self, input_shape):
        self.shape = (input_shape, input_shape)

    def __str__(self):
        return "Convolutional shape = " + str(self.shape) + "filters = " + str(self.filters)

    def isTrainable(self):
        # TODO
        return False


class Pool2D(Basic2DLayer):
    """
    also called inception

    """

    def __init__(self, node_shape, pool_type="max", **kwargs):
        super().__init__(0, **kwargs)
        self.shape = node_shape
        """
            Max Pooling
            Average Pooling
            Sum Pooling
        """
        # TODO pool_type
        self.pool_type = pool_type

    def setInputCount(self, input_count):
        raise Exception("setInputCount is deprecated for class Pool2D")

    def predict(self, input_array: np.ndarray):
        # TODO
        pass

    def __str__(self):
        return "Pool2D shape = " + str(self.shape)

    def isTrainable(self):
        return False


class NN:
    layers: list

    def __init__(self, layers: list = None):
        assert len(layers) > 1
        self.layers = []
        for layer in layers:
            self.addLayer(layer)

    def addLayer(self, layer: BasicLayer):
        if len(self.layers) > 0:
            previous_layer = self.layers[-1]
            layer.setInputCount(previous_layer.shape[0])
        self.layers.append(layer)

    def predict(self, input_array: Union[np.ndarray, list]):
        if isinstance(input_array, list):
            input_array = np.array(input_array)
        input_shape = input_array.shape if len(input_array.shape) > 1 else input_array.shape[0]
        assert input_shape == self.layers[0].shape[0], \
            "input shape " + str(input_array.shape) + " not compatible with input layer shape " + str(
                self.layers[0].shape[0])

        self.layers[0].activation = input_array
        for layer_index in range(1, len(self.layers), 1):
            self.layers[layer_index].predict(self.layers[layer_index - 1].activation)
        return self.layers[-1].activation

    def train(self, input_array, desired_output_array: Union[list, range, np.ndarray], learning_rate=0.8):
        layer = self.layers[-1]
        layer: BasicLayer

        if isinstance(desired_output_array, (range, list)):
            desired_output_array = np.array(desired_output_array)
        assert desired_output_array.shape[0] == layer.shape[0], "desired_output_array shape " + str(
            desired_output_array.shape) + " not compatible with last layer shape " + str(layer.shape)

        self.predict(input_array)

        cost_after_method_der = learning_rate * np.array([desired_output_array - layer.activation])
        # d(Cost)/d(AMethod) -- Cost = ((desired-predict)/2)^2
        # can add weight decay algorithm if needed over-fitting solution

        # momentum is like sliding of a hill, it has a velocity and it continues to go the same direction
        # but with a friction coefficient that slow it down ~[0.1:0.5] presumably
        # moreover, the new d(Cost)/d(AMethod) will be its accelerator
        # helps overcome lengthy squiggles
        # cost_after_method_der += momentum_coefficient * last_cost_after_method_der

        for layer_index in range(len(self.layers) - 1, 0, -1):
            layer: BasicLayer
            layer = self.layers[layer_index]
            if isinstance(layer, Dropout):
                cost_after_method_der = cost_after_method_der * layer.dropout_array

            if layer.isTrainable():
                cost_before_method_der = cost_after_method_der * [layer.activation_derivative_method(layer.activation)]
                # d(Cost)/d(AMethod) * d(AMethod)/d(BMethod) , d(AMethod)/d(BMethod) = activation derivative method

                layer.input_weights_change = cost_before_method_der.T * [self.layers[layer_index - 1].activation]
                # d(Cost)/d(BMethod) * d(BMethod)/d(Weights) , d(BMethod)/d(Weights) = previous activation

                layer.bias_change = cost_before_method_der[0]
                # d(Cost)/d(BMethod) * d(BMethod)/d(Bias) , d(BMethod)/d(Bias) = 1

                cost_after_method_1_der = np.sum(layer.input_weights.T * cost_before_method_der, 1)
                # d(Cost)/d(BMethod) * d(BMethod)/d(AMethod(-1))

                cost_after_method_der = np.array([cost_after_method_1_der])
                # continues the pass on the last d(Cost)/d(AMethod) to the next layer
        pass

    def trainBatch(self, input_batch, desired_output_batch, learning_rate=0.8):
        assert len(input_batch) == len(desired_output_batch), "input and output batches must be in the same length " \
                                                              + f"{len(input_batch)} != {len(desired_output_batch)}"

        for index in range(len(input_batch)):
            input_array = input_batch[index]
            desired_output_array = desired_output_batch[index]
            self.train(input_array, desired_output_array, learning_rate=learning_rate)

    def trainBellman(self, input_array):
        # TODO Study
        pass

    def trainByCost(self, cost: float):
        pass

    def mutate(self, mutation_coefficient=0.1):
        for layer in self.layers[1:]:
            layer: BasicLayer
            layer.mutate(mutation_coefficient=mutation_coefficient)

    def saveAsPickle(self, filename: str):
        with open(filename, mode='wb') as file:
            _pickle.dump(self, file, -1)
            file.close()

    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer) + "\n"
        return string


def loadFromPickle(filename: str):
    try:
        with open(filename, mode='rb') as file:
            obj = _pickle.load(file)
            file.close()
            if isinstance(obj, NN):
                return obj
    except IOError:
        return None


start_time = time.time()
NetTestConnect3()
# NetTestXor()
# NetTestOdd()
# cProfile.run('Test()', sort=1)
# cProfile.runctx("Test()", globals(), locals(), filename="NumpyNN.profile")
print(f"time elapsed = {time.time() - start_time} seconds")
