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
        self.image = []

    def _add_convolution_layer(self, number_filters, filter_size):

        if type(number_filters) != list:
            self.kernals[f'K{len(self.kernals)}'] = np.random.randn(number_filters, filter_size, filter_size) * np.sqrt(1 / filter_size)

        else:
            for num_filters, size in zip(number_filters, filter_size):
                self.kernals[f'K{len(self.kernals)}'] = np.random.randn(num_filters, size, size) * np.sqrt(1 / size)

    def _convolve_image(self, image, padding='valid', pooling_size=2):

        if padding == 'valid':

            # initialize img variable with original image
            img = image

            for layer, filters_array in self.kernals.items():
                filters, _, f_size = filters_array.shape
                try:
                    h, w = img.shape
                except ValueError:
                    _, h, w = img.shape
                horiz_segs = (w - f_size) + 1
                vert_segs = (h - f_size) + 1
                last_v = 0
                depth_tracker = 0
                max_depth = horiz_segs * vert_segs
                for v_layer, img_seg in self.segment_images(img, f_size, v_segs=vert_segs, h_segs=horiz_segs):
                    depth_tracker += 1
                    if v_layer != last_v:
                        try:
                            self.image = [new_img.shape, temp.shape]
                            new_img = np.hstack((new_img, temp))
                            del temp
                        except NameError:
                            new_img = temp
                            del temp
                    try:
                        if len(img_seg.shape) == 3:
                            tmp_check = temp.copy()
                            del tmp_check
                            for seg in img_seg:
                                try:
                                    next_layer = np.vstack((next_layer, self.apply_filters(filters_array, seg)))
                                except NameError:
                                    next_layer = self.apply_filters(filters_array, seg)
                            self.image = [temp.shape, next_layer.shape]
                            temp = np.dstack((temp, next_layer))
                            del next_layer
                        else:
                            temp = np.dstack((temp, self.apply_filters(filters_array, img_seg)))
                    except NameError:
                        if len(img_seg.shape) == 3:
                            for seg in img_seg:
                                try:
                                    temp = np.vstack((temp, self.apply_filters(filters_array, seg)))
                                except NameError:
                                    temp = self.apply_filters(filters_array, seg)
                        else:
                            temp = self.apply_filters(filters_array, img_seg)
                    last_v = v_layer
                    if depth_tracker == max_depth:
                        new_img = np.hstack((new_img, temp))
                # update the image with the new set of images
                img = new_img
                del new_img
                del temp
                try:
                    del next_layer
                except UnboundLocalError:
                    pass
            return img

    @staticmethod
    def segment_images(image, segment_size, v_segs='', h_segs='', for_pooling=False):

        if for_pooling:
            _, h, w = image.shape
            assert (sum(np.array([h, w]) % segment_size) == 0), f"Pool size of {segment_size} does not fit image dims {(h, w)}"

            v_segs = [i*segment_size for i in range(h // segment_size)]
            h_segs = [i*segment_size for i in range(w // segment_size)]

        else:

            v_segs = range(v_segs)
            h_segs = range(h_segs)

        for i in v_segs:
            rows = i + segment_size
            for j in h_segs:
                cols = j + segment_size
                try:
                    yield i, image[:, i:rows, j:cols]
                except IndexError:
                    yield i, image[i:rows, j:cols]

    @staticmethod
    def pool_segment(img_seg, agg_type='max'):

        if agg_type == 'max':
            return np.amax(np.amax(img_seg, axis=1), axis=1).reshape((len(img_seg), 1, 1))

        elif agg_type == 'min':
            return np.amin(np.amin(img_seg, 1), 1).reshape((len(img_seg), 1, 1))

    @staticmethod
    def apply_filters(filters_array, segment):
        return np.sum(np.sum(filters_array * segment, axis=1), axis=1).reshape((len(filters_array), 1, 1))






