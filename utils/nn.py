import numpy as np


class NeuralNetwork(object):

    def __init__(self, sizes, l_rate=0.01):
        self.sizes = sizes
        self.layers = len(sizes)
        self.l_rate = l_rate

        self.params = self._initialization()

    def _initialization(self):

        params = {}
        for layer, neurons in enumerate(self.sizes):
            if layer == 0:
                continue
            else:
                params[f'W{layer-1}'] = np.random.randn(neurons, self.sizes[layer-1]) * np.sqrt(1 / neurons)

        return params

    def feedforward(self, input_layer, a_func='sigmoid'):
        params = self.params

        params['A0'] = input_layer

        for layer in range(self.layers):
            if layer == 0:
                continue
            else:
                params[f'Z{layer-1}'] = params[f'W{layer-1}'] @ params[f'A{layer-1}']
                params[f'A{layer}'] = self.activation(params[f'Z{layer-1}'], func=a_func)

        return params[f'A{layer}']

    def backprop(self, prediction, actual, a_func='sigmoid'):

        params = self.params
        weight_changes = {}
        deltas = [0 for _ in range(self.layers-1)]
        cost = prediction - actual
        deltas[-1] = (cost * self.activation(params[f'Z{self.layers-2}'], func=a_func, derivative=True))

        l_layer = self.layers - 2
        for layer in range(self.layers - 1):
            if layer == 0:
                weight_changes[f'W{l_layer-layer}'] = deltas[l_layer-layer] @ params[f'A{l_layer-layer}'].T
            else:
                new_delt = (params[f'W{l_layer-layer+1}'].T @ deltas[l_layer-layer+1])
                deltas[l_layer-layer] = new_delt * self.activation(params[f'Z{l_layer-layer}'], func=a_func, derivative=True)
                weight_changes[f'W{l_layer-layer}'] = deltas[l_layer-layer] @ params[f'A{l_layer-layer}'].T

        return weight_changes

    @staticmethod
    def activation(val, func='sigmoid', derivative=False):

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

    def update_weights(self, weight_changes):

        for layer, values in weight_changes.items():
            self.params[layer] -= (self.l_rate * values)


