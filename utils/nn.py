import numpy as np


class NeuralNetwork(object):

    def __init__(self, sizes, l_rate=0.01, m_factor=0.9):
        self.sizes = sizes
        self.layers = len(sizes)
        self.l_rate = l_rate
        self.m_factor = m_factor
        self.forward_passes = 0
        self.prev_update = {}
        self.batch_retention = {}

        self.params = self._initialization()

    def _initialization(self):

        params = {}
        for layer, neurons in enumerate(self.sizes[1:]):
            params[f'W{layer}'] = np.random.randn(neurons, self.sizes[layer]) * np.sqrt(1 / neurons)
            params[f'B{layer}'] = np.zeros((neurons, 1)) + 0.01
            self.prev_update[f'W{layer}'] = 0
            self.prev_update[f'B{layer}'] = 0

        return params

    def feedforward(self, input_layer, a_func='sigmoid'):

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

    def backprop(self, prediction, actual, a_func='sigmoid'):

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


class ConvNeuralNetwork(NeuralNetwork):

    def __init__(self, sizes, l_rate=0.01, m_factor=0.9):

        super().__init__(sizes, l_rate, m_factor)
        self.kernals = {}

    def _add_convolution_layer(self, number_filters, filter_size):

        if type(number_filters) != list:
            self.kernals[f'K{len(self.kernals)}'] = np.random.randn(number_filters, filter_size, filter_size) * np.sqrt(1 / filter_size)

        else:
            for num_filters, size in zip(number_filters, filter_size):
                self.kernals[f'K{len(self.kernals)}'] = np.random.randn(num_filters, size, size) * np.sqrt(1 / size)

    def _convolve_image(self, image, padding='valid'):

        if padding == 'valid':

            # initialize img variable with original image
            img = image

            for layer, filters_array in self.kernals.items():
                filters, _, f_size = filters_array.shape
                h, w = img.shape
                horiz_segs = (w - f_size) + 1
                vert_segs = (h - f_size) + 1
                last_v = 0
                for v_layer, img_seg in self._segment_image(img, f_size, vert_segs, horiz_segs):
                    if v_layer != last_v:
                        try:
                            new_img = np.hstack((new_img, temp))
                            del temp
                        except NameError:
                            new_img = temp
                    try:
                        temp = np.dstack((temp, np.sum(np.sum(filters_array * img_seg, axis=1), axis=1).reshape(filters, 1, 1)))
                    except NameError:
                        temp = np.sum(np.sum(filters_array * img_seg, axis=1), axis=1).reshape(filters, 1, 1)
                    last_v = v_layer
                # update
                img = new_img
                # TODO: Figure out convolution layers that come after first one
    @staticmethod
    def _segment_image(image, segment_size, v_segs, h_segs):

        if len(image.shape) == 2:
            image = [image]

        for feature_map in image:
            for i in range(v_segs):
                for j in range(h_segs):
                    rows = i + segment_size
                    cols = j + segment_size
                    yield i, feature_map[i:rows, j:cols]




