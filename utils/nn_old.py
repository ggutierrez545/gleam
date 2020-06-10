import numpy as np


class NeuralNetwork2(object):

    def __init__(self, sizes, l_rate=0.01, m_factor=0.9):
        self.sizes = sizes
        self.layers = len(sizes)
        self.l_rate = l_rate
        self.m_factor = m_factor
        self.forward_passes = 0
        self.prev_update = {}
        self.batch_retention = {}

        self.params = self._connected_layer_init()

    def _connected_layer_init(self):

        params = {}
        for layer, neurons in enumerate(self.sizes[1:]):
            params[f'W{layer}'] = np.random.randn(neurons, self.sizes[layer]) * np.sqrt(1 / neurons)
            params[f'B{layer}'] = np.zeros((neurons, 1)) + 0.01
            self.prev_update[f'W{layer}'] = 0
            self.prev_update[f'B{layer}'] = 0

        return params

    def nn_feedforward(self, input_layer, a_func='sigmoid'):

        self.forward_passes += 1
        params = self.params

        params['A0'] = input_layer

        for layer in range(self.layers):
            if layer == 0:
                continue
            else:
                params[f'Z{layer-1}'] = params[f'W{layer-1}'] @ params[f'A{layer-1}'] + params[f'B{layer-1}']
                params[f'A{layer}'] = self.activation(params[f'Z{layer-1}'], func=a_func)

        return params[f'A{layer}']

    def nn_backprop(self, prediction, actual, a_func='sigmoid'):

        params = self.params
        wb_changes = {}
        deltas = [0 for _ in range(self.layers-1)]
        cost = prediction - actual
        deltas[-1] = (cost * self.activation(params[f'Z{self.layers-2}'], func=a_func, derivative=True))

        l_layer = self.layers - 2
        for layer in range(self.layers - 1):
            if layer == 0:
                wb_changes[f'W{l_layer-layer}'] = deltas[l_layer-layer] @ params[f'A{l_layer-layer}'].T
                wb_changes[f'B{l_layer-layer}'] = deltas[l_layer-layer]
            else:
                new_delt = (params[f'W{l_layer-layer+1}'].T @ deltas[l_layer-layer+1])
                deltas[l_layer-layer] = new_delt * self.activation(params[f'Z{l_layer-layer}'], func=a_func, derivative=True)
                wb_changes[f'W{l_layer-layer}'] = deltas[l_layer-layer] @ params[f'A{l_layer-layer}'].T
                wb_changes[f'B{l_layer-layer}'] = deltas[l_layer-layer]

        return wb_changes

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

    def _update_weights(self, weight_bias_changes, updater='sgd', batch_size=50, momentum=True):

        if updater == 'sgd':

            for layer, values in weight_bias_changes.items():
                update = (self.l_rate * values) + (self.m_factor * self.prev_update[layer])
                self.params[layer] -= update
                if momentum:
                    self.prev_update[layer] = -update

        elif updater == 'mini_batch':

            if self.forward_passes % batch_size != 0:
                for layer, values in weight_bias_changes.items():
                    try:
                        self.batch_retention[layer] += values
                    except KeyError:
                        self.batch_retention[layer] = values

            else:
                for layer, values in self.batch_retention.items():
                    update = (self.l_rate * (values/batch_size)) + (self.m_factor * self.prev_update[layer])
                    self.params[layer] -= update
                    if momentum:
                        self.prev_update[layer] = -update
                self.batch_retention = weight_bias_changes

        else:

            raise KeyError(f"Unrecognized updater: {updater}")
