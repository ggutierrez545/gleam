import warnings
import numpy as np
from .nn import NeuralNetwork, InputLayer


class ConvNeuralNetwork(NeuralNetwork):

    def __init__(self, l_rate=0.01, m_factor=0.9):

        super().__init__(l_rate, m_factor)
        self.kernals = {}
        self.conv_params = {}
        self.conv_layers = 0

    def add_convolution_layer(self, number_filters, filter_size):

        if type(number_filters) != list:
            self.kernals[f'K{len(self.kernals)}'] = np.random.randn(number_filters, filter_size, filter_size) * np.sqrt(1 / filter_size)
            self.conv_layers += 1

        else:
            for num_filters, size in zip(number_filters, filter_size):
                self.kernals[f'K{len(self.kernals)}'] = np.random.randn(num_filters, size, size) * np.sqrt(1 / size)
                self.conv_layers += 1

    def cnn_feedforward(self, image, padding='', pool=True, pooling_size=2, pool_agg='max', a_func='sigmoid'):

        flat_imgs = self._convolve_image(image, padding=padding, pool=pool, pooling_size=pooling_size, pool_agg=pool_agg)

        if self.forward_passes == 0:
            self.sizes = [len(flat_imgs)] + self.sizes
            self.layers = len(self.sizes)
            self.params = self._connected_layer_init()

        input_layer = self.activation(flat_imgs, func=a_func)
        return self.nn_feedforward(input_layer, a_func=a_func)

    def cnn_backprop(self, prediction, actual, a_func='sigmoid'):

        wb_changes = {**self.nn_backprop(prediction, actual, a_func=a_func)}

        l_layer = len(self.conv_params)
        # for layer in range(self.conv_layers):

    def _convolve_image(self, image, padding='', pool=True, pooling_size=2, pool_agg='max'):

        # initialize img variable with original image
        img = image

        for layer, filters_array in self.kernals.items():

            if padding == 'same':
                self.conv_params[f'PP{layer[-1]}'] = img
                img = self.pad_image(img, filters_array.shape[-1])

            img = self._convolve_layer(img, f_array=filters_array)
            self.conv_params[f'C{layer[-1]}'] = img

            if pool:
                try:
                    img = self._convolve_layer(img, pooling_size=pooling_size, pool_agg=pool_agg)
                except AssertionError:
                    warnings.warn(f"Conv layer {layer} was unable to be pooled due to size mismatch.")


        return img.ravel().reshape(-1, 1)

    def _convolve_layer(self, image, f_array='', pooling_size='', pool_agg=''):

        depth_tracker = 0
        last_v = 0

        try:
            _, h, w = image.shape
        except ValueError:
            h, w = image.shape

        if type(f_array) == str:
            pooling = True
            vert_segs = ''
            horiz_segs = ''
            max_depth = (h // pooling_size) * (w // pooling_size)
            segment_size = pooling_size

        else:
            pooling = False
            filters, _, f_size = f_array.shape
            horiz_segs = (w - f_size) + 1
            vert_segs = (h - f_size) + 1
            max_depth = horiz_segs * vert_segs
            segment_size = f_size

        for v_layer, img_seg in self.segment_images(image, segment_size, v_segs=vert_segs, h_segs=horiz_segs, pool=pooling):
            depth_tracker += 1
            if v_layer != last_v:
                try:
                    new_img = np.hstack((new_img, temp))
                    del temp
                except NameError:
                    new_img = temp
                    del temp
            try:
                if len(img_seg.shape) == 3:
                    tmp_check = temp.copy()
                    del tmp_check
                    if pooling:
                        try:
                            next_layer = np.vstack((next_layer, self.pool_segment(img_seg, agg_type=pool_agg)))
                        except NameError:
                            next_layer = self.pool_segment(img_seg, agg_type=pool_agg)
                    else:
                        for seg in img_seg:
                            try:
                                next_layer = np.vstack((next_layer, self.apply_filters(f_array, seg)))
                            except NameError:
                                next_layer = self.apply_filters(f_array, seg)
                    temp = np.dstack((temp, next_layer))
                    del next_layer
                else:
                    temp = np.dstack((temp, self.apply_filters(f_array, img_seg)))
            except NameError:
                if len(img_seg.shape) == 3:
                    if pooling:
                        try:
                            temp = np.vstack((temp, self.pool_segment(img_seg, agg_type=pool_agg)))
                        except NameError:
                            temp = self.pool_segment(img_seg, agg_type=pool_agg)
                    else:
                        for seg in img_seg:
                            try:
                                temp = np.vstack((temp, self.apply_filters(f_array, seg)))
                            except NameError:
                                temp = self.apply_filters(f_array, seg)
                else:
                    temp = self.apply_filters(f_array, img_seg)
            last_v = v_layer
            if depth_tracker == max_depth:
                new_img = np.hstack((new_img, temp))

        return new_img

    @staticmethod
    def segment_images(image, segment_size, v_segs='', h_segs='', pool=False):

        if pool:
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

        elif agg_type == 'avg':
            return

    @staticmethod
    def apply_filters(filters_array, segment):
        return np.sum(np.sum(filters_array * segment, axis=1), axis=1).reshape((len(filters_array), 1, 1))

    @staticmethod
    def pad_image(image, filter_size):

        pad = (filter_size - 1) // 2

        try:
            return np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        except ValueError:
            return np.pad(image, (pad, pad), 'constant', constant_values=0)



