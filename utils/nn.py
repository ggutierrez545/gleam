import numpy as np


class NeuralNetwork(object):

    def __init__(self, l_rate=0.01, m_factor=0.9):
        self.l_rate = l_rate
        self.m_factor = m_factor
        self.input_layer = None
        self.layers = []
        self.segments = []

    def add_input_layer(self, size):
        # Before adding an InputLayer, verify one has not already been initialized
        if [type(i) for i in self.layers].__contains__(InputLayer):
            raise AssertionError("NeuralNetwork instance already contains InputLayer")
        else:
            # Accessibility of InputLayer makes feedforward method much easier. Same instance of InputLayer class is
            # referenced in both self.input_layer and self.layers
            self.input_layer = InputLayer(size)
            self.layers.append(self.input_layer)

    def add_connected_layer(self, size):
        # Before adding ConnectedLayer, verify an InputLayer has already been initialized
        if [type(i) for i in self.layers].__contains__(InputLayer):
            self.layers.append(ConnectedLayer(size))
            # After each ConnectedLayer is added, create a NetSegment from the last two elements in self.layers.
            # Using elements from self.layers to create the NetSegment instance allows the chain of InputLayer and
            # ConnectedLayer references to be maintained. This is crucial for this architecture
            self.segments.append(NetSegment(*self.layers[-2:]))
        else:
            raise AssertionError("Neural Network instance must contain an InputLayer before adding a ConnectedLayer")

    def nn_feedforward(self, x, a_function='relu'):
        # Update the InputLayer instance with a new set of values. This update will now be available in the first
        # NetSegment instance in the self.segments list.
        self.input_layer.a_vals = x

        # And just simply iterate through each NetSegment instance, calling the forward_pass method on each which will
        # update the relevant ConnectedLayer for use in the next NetSegment
        for segment in self.segments:
            segment.forward_pass(activation=a_function)

    def nn_backpropagate(self, truth, a_function='relu', updater='sgd', batch_size=50, momentum=True):

        cost = self.segments[-1].back.a_vals - truth

        for segment in reversed(self.segments):
            try:
                delta = (cost * segment.activation(segment.back.z_vals, func=a_function, derivative=True))
                del cost
            except NameError:
                pass
            segment.back_propagate(delta)
            delta = segment.setup_next_delta(delta, activation=a_function)
            segment.update_weights(self.l_rate, self.m_factor, updater=updater, batch_size=batch_size, momentum=momentum)


class InputLayer(object):

    def __init__(self, size):
        self.size = size
        self.shape = (size, 1)
        # a_vals are the "Activation Values" i.e. values that have been passed through an activation function
        self.a_vals = None

    @property
    def a_vals(self):
        return self.__a_vals

    @a_vals.setter
    def a_vals(self, a_vals):
        try:
            assert len(a_vals) == self.size, f"New layer size, {len(a_vals)}, != initial layer size {self.size}"
            if len(a_vals.shape) == 1:
                self.__a_vals = a_vals.reshape(-1, 1)
            elif a_vals.shape == self.shape:
                self.__a_vals = a_vals
            else:
                raise ValueError(f"New layer shape, {a_vals.shape}, != initial layer shape {self.shape}")
        except TypeError:
            self.__a_vals = None


class ConnectedLayer(InputLayer):

    def __init__(self, size):
        super().__init__(size)
        # z_vals are raw values that come from the previous layers Activation Values multipled by the weights b/w
        # the two layers.
        self.z_vals = None
        self.biases = np.zeros((size, 1)) + 0.01

    @property
    def z_vals(self):
        return self.__z_vals

    @z_vals.setter
    def z_vals(self, z_vals):
        try:
            assert z_vals.shape == self.shape, f"New z layer shape, {z_vals.shape}, != initial layer shape {self.shape}"
            self.__z_vals = z_vals
        except AttributeError:
            self.__z_vals = None

    @property
    def biases(self):
        return self.__biases

    @biases.setter
    def biases(self, biases):
        assert biases.shape == self.shape, f"New biases shape, {biases.shape}, != shape of layer {self.shape}"
        self.__biases = biases


class NetSegment(object):

    def __init__(self, input_layer, output_layer):
        self.front = input_layer
        self.back = output_layer
        self.weights = np.random.randn(output_layer.size, input_layer.size) * np.sqrt(1 / input_layer.size)
        self.shape = self.weights.shape
        self.w_updates = None
        self.prev_w_updates = 0
        self.w_batch = None
        self.b_updates = None
        self.prev_b_updates = 0
        self.b_batch = None
        self.forward_passes = 0

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        try:
            if weights.shape != self.shape:
                raise ValueError(f"Updated weights shape, {weights.shape}, != initial weights shape {self.shape}")
            else:
                self.__weights = weights
        except AttributeError:
            self.__weights = weights

    @property
    def w_updates(self):
        return self.__w_updates

    @w_updates.setter
    def w_updates(self, w_updates):
        try:
            assert w_updates.shape == self.shape, f"Weight updates shape, {w_updates.shape}, != initial weights shape {self.shape}"
            self.__w_updates = w_updates
        except AttributeError:
            self.__w_updates = None

    @property
    def b_updates(self):
        return self.__b_updates

    @b_updates.setter
    def b_updates(self, b_updates):
        try:
            assert b_updates.shape == self.back.shape, f"Bias updates shape, {b_updates.shape}, != initial bias shape {self.back.shape}"
            self.__b_updates = b_updates
        except AttributeError:
            self.__b_updates = None

    def forward_pass(self, activation=''):
        self.back.z_vals = self.weights @ self.front.a_vals + self.back.biases
        self.back.a_vals = self.activation(self.back.z_vals, func=activation)
        self.forward_passes += 1

    def back_propagate(self, delta):
        self.w_updates = delta @ self.front.a_vals.T
        self.b_updates = delta

    def setup_next_delta(self, delta, activation=''):
        try:
            return (self.weights.T @ delta) * self.activation(self.front.z_vals, func=activation, derivative=True)
        # If self.front is an InputLayer, i.e. we've backpropagated to the intial layer, there will be an AttributeError
        # becuase InputLayers do not have z_vals. Catch this error and pass because backproagation is complete
        except AttributeError:
            return None

    def update_weights(self, l_rate, m_factor, updater='', batch_size=50, momentum=True):

        if updater == 'sgd':

            w_update = (l_rate * self.w_updates) + (m_factor * self.prev_w_updates)
            b_update = (l_rate * self.b_updates) + (m_factor * self.prev_b_updates)

            self.weights -= w_update
            self.back.biases -= b_update

            if momentum:
                self.prev_w_updates = -w_update
                self.prev_b_updates = -b_update

        elif updater == 'mini_batch':

            if self.forward_passes % batch_size != 0:
                try:
                    self.w_batch += self.w_updates
                    self.b_batch += self.b_updates
                except TypeError:
                    self.w_batch = self.w_updates
                    self.b_batch = self.b_updates

            else:
                w_update = (l_rate * (self.w_batch/batch_size)) + (m_factor * self.prev_w_updates)
                b_update = (l_rate * (self.b_batch/batch_size)) + (m_factor * self.prev_b_updates)

                self.weights -= w_update
                self.back.biases -= b_update

                if momentum:
                    self.prev_w_updates -= w_update
                    self.prev_b_updates -= b_update

                self.w_batch = self.w_updates
                self.b_batch = self.b_updates

        else:

            raise KeyError(f"Unrecognized updater: {updater}")

    @staticmethod
    def activation(val, func='', derivative=False):

        if func == 'sigmoid':
            if derivative:
                return (np.exp(-val)) / ((1 + np.exp(-val)) ** 2)
            else:
                return 1 / (1 + np.exp(-val))

        elif func == 'relu':
            if derivative:
                return np.array([1.0 if i > 0.0 else 0.0 for i in val]).reshape(-1, 1)
            else:
                return np.array([max(0, i[0]) for i in val]).reshape(-1, 1)


