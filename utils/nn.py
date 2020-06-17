import numpy as np


class NeuralNetwork(object):
    """Base class representation of a neural network.

        Contains the logic and framework
        to build any fully connected, feed-forward neural network.

        Parameters
        ----------
        seed : int
            Seed for pseudo random number generation.
        l_rate : float
            Learning rate for gradient descent back propagation.
        m_factor : float
            Momentum factor for gradient descent back propagation.

        Attributes
        ----------
        seed
        l_rate
        m_factor
        input_layer : None or :obj:`InputLayer`
            None when `NeuralNetwork` class is initialized.
            `InputLayer` instance once user calls `add_input_layer` method.
        layers : :obj:`list` of :obj:`InputLayer` and :obj:`ConnectedLayer`
            List containing the `InputLayer` and `ConnectedLayer`
            instances that form the architecture of the `NeuralNetwork`.
            First element in list is an `InputLayer` instance and
            all subsequent elements are `ConnectedLayer` instances.
        segments : list
            List containing `NetSegment` instances which contain much of the
            primary feed forward / back propagation logic.

        Methods
        -------
        input_layer()
            Get or set the current `InputLayer` instance.
        add_input_layer(size)
            Add an `InputLayer` instance to the `NeuralNetwork` and append it to the `layers` attribute.
        add_connected_layer(size)
            Add an `InputLayer` instance to the `NeuralNetwork' and append it to the `layers` attribute.
        nn_feedforward(x, a_function='relu')
            Feed the `NeuralNetwork` an example to make a prediction on.
        nn_backpropagate(truth, a_function='relu', updater='sgd', batch_size=50, momentum=True)
            Back propagate the resulting error from a `nn_feedforward` pass.

    """
    def __init__(self, seed=10, l_rate=0.01, m_factor=0.9):
        self.seed = seed
        self.l_rate = l_rate
        self.m_factor = m_factor
        self.input_layer = None
        self.layers = []
        self.segments = []

        np.random.seed(seed)

    @property
    def input_layer(self):
        """:obj:`InputLayer` : `InputLayer` instance serving as the first layer in the `NeuralNetwork`.

        Setter method ensures input_layer must be `InputLayer` instance.

        Raises
        ------
        AssertionError
            If input_layer attempted is not `InputLayer` instance.

        """
        return self.__input_layer

    @input_layer.setter
    def input_layer(self, input_layer):
        if input_layer is None:
            self.__input_layer = None
        else:
            assert type(input_layer) is InputLayer, f"Cannot set input_layer with {type(input_layer)}; must be InputLayer instance"
            self.__input_layer = input_layer

    def add_input_layer(self, size):
        """Method to add an input layer of inputted size.

        Parameters
        ----------
        size : int
            Number of neurons in input layer.

        Raises
        ------
        AssertionError
            If `NeuralNetwork` instance already contains an `InputLayer` in the `layers` attribute.

        Notes
        -----
        Method does not return anything. Instead, it sets the `input_layer` attribute to an `InputLayer`
        instance and appends it to the beginning of the `layers` attribute.
        """
        # Before adding an InputLayer, verify one has not already been initialized
        if [type(i) for i in self.layers].__contains__(InputLayer):
            raise AssertionError("NeuralNetwork instance already contains InputLayer")
        else:
            # Accessibility of InputLayer makes feedforward method much easier. Same instance of InputLayer class is
            # referenced in both self.input_layer and self.layers
            self.input_layer = InputLayer(size)
            self.layers.append(self.input_layer)

    def add_connected_layer(self, size):
        """Method to add `ConnectedLayer` of inputted size.

        Parameters
        ----------
        size : int
            Number of neurons in connected layer.

        Raises
        ------
        AssertionError
            If `NeuralNetwork` does not already contain an `InputLayer` instance.

        """
        # Before adding ConnectedLayer, verify an InputLayer has already been initialized
        if [type(i) for i in self.layers].__contains__(InputLayer):
            self.layers.append(ConnectedLayer(size))
            # After each ConnectedLayer is added, create a NetSegment from the last two elements in self.layers.
            # Using elements from self.layers to create the NetSegment instance allows the chain of InputLayer and
            # ConnectedLayer references to be maintained. This is crucial for this architecture
            self.segments.append(NetSegment(*self.layers[-2:]))
        else:
            raise AssertionError("NeuralNetwork instance must contain an InputLayer before adding a ConnectedLayer")

    def nn_feedforward(self, x, a_function='relu'):
        """Method to feed forward an example through the `NeuralNetwork` and make a prediction.

        Parameters
        ----------
        x : `numpy.ndarray`
            Numpy array containing training example input data.
        a_function : str
            String keyword for activation function. Supported keywords are 'relu' and 'sigmoid'.

        Notes
        -----
        `x` overwrites the `a_vals` attribute in the `NeuralNetwork` instance's `InputLayer` allowing the information
        to be transfered to the `NetSegment` instance containing the `InputLayer` as well, thereby making the
        feed forward process very simple.

        """
        # Update the InputLayer instance with a new set of values. This update will now be available in the first
        # NetSegment instance in the self.segments list.
        self.input_layer.a_vals = x

        # And just simply iterate through each NetSegment instance, calling the forward_pass method on each which will
        # update the relevant ConnectedLayer for use in the next NetSegment
        for segment in self.segments:
            segment.forward_pass(activation=a_function)

    def nn_backpropagate(self, truth, a_function='relu', updater='sgd', batch_size=50, momentum=True):
        """Method to back propagate the error from a training example

        Parameters
        ----------
        truth : `np.ndarray`
            Array depicting the training example's actual value.
        a_function : :obj:str, default 'relu'
            String keyword for activation function. Supported keywords are 'relu' and 'sigmoid'.
        updater : :obj:str, default 'sgd'
            String keyword for weights and biases updater method. Support keywords are 'sgd' and 'mini-batch'.
        batch_size : :obj:int, default 50
            Size of the mini-batch to update.
        momentum : :obj:bool, default `True`
            Toggle to include momentum calculation in updater method.

        """
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
    """Simple class depicting the first layer in a neural network.

    Serves as base class for `ConnectedLayer`.

    Parameters
    ----------
    size : int
        Number of neurons in the layer.

    Attributes
    ----------
    shape : tuple
        Shape of the layer as an array,
    a_vals : :obj:`None` or :obj:`numpy.ndarray`
        Array of activation values.

    """

    def __init__(self, size):
        self.size = size
        self.shape = (size, 1)
        self.a_vals = None

    @property
    def a_vals(self):
        """Array container for activation values.

        Setter method has a number of checks to ensure the new `a_vals` is the same size and shape as the
        previous `a_vals`. This is to maintain dimensional continuity within the `NeuralNetwork` instance.

        Raises
        ------
        AssertionError
            If number of neurons in new `a_vals` does not match previous neuron count.
        ValueError
            If new `a_vals` array shape does not match previous `a_vals` array shape.

        """
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
    """Child class of `InputLayer` depicting connected layers in a neural network.

    Parameters
    ----------
    size : int
        Number of neurons in the connected layer.

    Attributes
    ----------
    z_vals : :obj:`None` or :obj:`numpy.array`
        Layer's raw values, i.e. before passing through activation function.
    biases : `numpy.array`
        Bias value associated with each neuron in the layer.

    See Also
    --------
    `InputLayer`

    """
    def __init__(self, size):
        super().__init__(size)
        self.z_vals = None
        self.biases = np.zeros((size, 1)) + 0.01

    @property
    def z_vals(self):
        """Array container for layer values pre-activation function.

        Setter method verifies the new `z_vals` array shape matches the old `z_vals` array shape. If the attempted
        `z_vals` is not an array and does not have a `shape` method, `z_vals` is set to `None`.

        Raises
        ------
        AssertionError
            If new `z_vals` array shape is not equal to previous shape.

        """
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
        """Array container for neurons' bias terms.

        Setter method verifies attempted `biases` array shape matches old shape.

        Raises
        ------
        AssertionError
            If attempted `biases` shape does not match original shape.

        """
        return self.__biases

    @biases.setter
    def biases(self, biases):
        assert biases.shape == self.shape, f"New biases shape, {biases.shape}, != shape of layer {self.shape}"
        self.__biases = biases


class NetSegment(object):
    """Container class for two layers in a `NeuralNetwork` instance.

    `NetSegment` instances contain the weights between two layers in the neural network, as well as much of the logic
    for the feed forward and back propagation methods of the `NeuralNetwork` class. Consecutive `NetSegment` instances
    have overlapping `front` and `back` layers (i.e they are the same `ConnectedLayer` instance). This architecture
    allows for easy access to either preceding or following layers when making calculations and allows the `NeSegment`
    class to contain simplified logic feed forward and back propagation applications.

    Parameters
    ----------
    input_layer : :obj:`InputLayer` or :obj:`ConnectedLayer`
        The first layer in the `NetSegment` instance.
    output_layer : `ConnectedLayer`
        The last layer in the `NetSegment` instance.

    Attributes
    ----------
    front : :obj:`InputLayer` or :obj:`ConnectedLayer`
        The first layer in the `NetSegment` instance.
    back : `ConnectedLayer`
        The last layer in the `NetSegment` instance.
    weights : `ndarray`
        Weights of connections between front and back layers.
    shape : tuple
        Shape of the weights array.
    w_updates : :obj:`None` or :obj:`ndarray`
        Array containing each weights' update calculated from back propagation.
    prev_w_updates : :obj:`int` or :obj:`ndarray`
        Array containing previous weights' update for use with momentum.
    w_batch : :obj:`None` or :obj:`ndarray`
        Array containing sum of weight updates for mini-batch sgd.
    b_updates : :obj:`None` or :obj:`ndarray`
        Array containing each biases' update calculated from back propagation.
    prev_b_updates : :obj:`int` or :obj:`ndarray`
        Array containing previous biases' update for use with momentum.
    b_batch : :obj:`None` or :obj:`ndarray`
        Array containing sum of bias updates for mini-batch sgd.
    forward_passes : int
        Number of times a training example has been fed forward.

    Methods
    -------
    forward_pass(activation='')
        Calculate a forward pass from the front layer to the back layer.
    back_propagate(delta)
        Calculate the weight updates via back propagation.
    setup_next_delta(delta, activation='')
        Setup delta value for use in next `NegSegment` instance.
    update_weights(l_rate, m_factor, updater='', batch_size=50, momentum=True)
        Update the weights based on back propagation pass.
    activation(val, func='', derivative=False)
        Static method to access various activation functions.

    """
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
        """Array container for the weights connecting the front layer to the back layer.

        Setter method contains logic to ensure consistent dimensions when updating weights.

        Raises
        ------
        ValueError
            If attempting to set weights array that does not match original shape of weights array.

        """
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
        """Array container for updates to each weight calculated via back propagation.

        Setter method contains logic to ensure consistent dimensions with original weight array.

        Raises
        ------
        AssertionError
            If attempting to set `w_updates` with dimensionally inconsistent array.


        """
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
        """Array container for `NetSegment` back layer's bias updates.

        Setter method contains logic to ensure dimensional consistency with back layer's bias array.

        Raises
        ------
        AssertionError
            If attempting to set `b_updates` with dimensionally inconsistent array.

        """
        return self.__b_updates

    @b_updates.setter
    def b_updates(self, b_updates):
        try:
            assert b_updates.shape == self.back.shape, f"Bias updates shape, {b_updates.shape}, != initial bias shape {self.back.shape}"
            self.__b_updates = b_updates
        except AttributeError:
            self.__b_updates = None

    def forward_pass(self, activation=''):
        """Fundamental logic to calculate a forward pass between two layers in a `NeuralNetwork` instance.

        Parameters
        ----------
        activation : str
            Keyword conveying type of activation function to use.

        """
        self.back.z_vals = self.weights @ self.front.a_vals + self.back.biases
        self.back.a_vals = self.activation(self.back.z_vals, func=activation)
        self.forward_passes += 1

    def back_propagate(self, delta):
        """Fundamental logic to calculate weight and bias updates from back propagation in a `NeuralNetwork` instance.

        Parameters
        ----------
        delta : :obj:`ndarray`
            Array containing necessary computations from earlier portions of back propagation.

        """
        self.w_updates = delta @ self.front.a_vals.T
        self.b_updates = delta

    def setup_next_delta(self, delta, activation=''):
        """Logic to calculate new deltas for each layer in back propagation calculation.

        Parameters
        ----------
        delta : :obj:`ndarray`
            Array containing necessary computations from earlier portions of back propagation.
        activation
            Keyword conveying type of activation function to use.

        Returns
        -------
        NoneType
            When back propagation has reached `InputLayer` instance.
        :obj:`ndarray`
            Delta array to use for next layer in back propagation computation.

        """
        try:
            return (self.weights.T @ delta) * self.activation(self.front.z_vals, func=activation, derivative=True)
        # If self.front is an InputLayer, i.e. we've backpropagated to the intial layer, there will be an AttributeError
        # becuase InputLayers do not have z_vals. Catch this error and pass because backproagation is complete
        except AttributeError:
            return None

    def update_weights(self, l_rate, m_factor, updater='', batch_size=50, momentum=True):
        """Function to update `NetSegment` instance's weights and biases based on user input.

        Parameters
        ----------
        l_rate : float
            The `NeuralNetwork`s learning rate.
        m_factor : float
            The `NeuralNetwork`s momentum factor.
        updater : str
            Keyword conveying type of activation function to use.
        batch_size : int
            Size of batch for mini-batch gradient descent.
        momentum : bool
            Whether or not to include momentum optimization.

        Raises
        ------
        KeyError
            If `updater` is unsupported.

        """

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
        """Primary function to handle neuron activations.

        Parameters
        ----------
        val : :obj:`float`, :obj:`int`, :obj:`ndarray`
            Inputted value.
        func : str
            Keyword conveying type of activation function to use.
        derivative : bool
            Whether or not to use derivative form of activation function.

        Returns
        -------
        float
            If `val` is `float` or `int`.
        :obj:`ndarray:
            If `val` is :obj:`ndarray`.

        Raises
        ------
        KeyError
            If `func` input is unsupported.

        """

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

        else:
            raise KeyError(f"Unrecognized activation function: {func}")


