import numpy as np
from glearn.resources.activation import activation, loss
from glearn.utils.nn import NeuralNetwork, InputLayer, ConnectedLayer, ConnectedSegment


class ConvolutionalNeuralNet(NeuralNetwork):

    def __init__(self, seed=10, l_rate=0.01, m_factor=0.9, loss_func=''):
        super().__init__(seed=seed, l_rate=l_rate, m_factor=m_factor)
        self.loss_func = loss_func

    def add_connected_layer(self, size, activation_func='relu'):
        """Method to add `ConnectedLayer` of inputted size.

        Overwrites method of same name from `NeuralNetwork` class.

        Parameters
        ----------
        size : int
            Number of neurons in connected layer.
        activation_func : str
            Keyword indicating the activation function to use in the layer.

        Raises
        ------
        AssertionError
            If `NeuralNetwork` does not already contain an `InputLayer` instance.

        """
        # Before adding ConnectedLayer, verify an InputLayer has already been initialized
        if [type(i) for i in self.layers].__contains__(InputLayer):
            self.layers.append(ConnectedLayer(size, activation=activation_func, parent=self.__class__))
            # After each ConnectedLayer is added, create a ConnectedSegment from the last two elements in self.layers.
            # Using elements from self.layers to create the ConnectedSegment instance allows the chain of InputLayer and
            # ConnectedLayer references to be maintained. This is crucial for this architecture
            self.segments.append(ConnectedSegment(*self.layers[-2:]))
        else:
            raise AssertionError("NeuralNetwork instance must contain an InputLayer before adding a ConnectedLayer")

    def add_convolution_layer(self, num_kernals, kernal_size=3, padding='valid'):
        if [type(i) for i in self.layers].__contains__(InputLayer):
            self.layers.append(ConvolutionLayer(num_kernals, kernal_size=kernal_size, padding=padding))
            # Calculate output characteristics from new layer
            self.layers[-1].calc_output_chars(self.layers[-2])
            # After each ConvolutionLayer is added, create a ConvolutionSegment from the last two elements in
            # self.layers.
            self.segments.append(ConvolutionSegment(*self.layers[-2:]))
        else:
            raise AssertionError("ConvolutionlNeuralNet instance must contain"
                                 " an InputLayer before adding a ConvolutionLayer")

    def add_pooling_layer(self, agg_type, pool_size=2):
        if type(self.layers[-1]) is not ConvolutionLayer:
            raise AssertionError(f"PoolingLayer must come after ConvolutionLayer, not {type(self.layers[-1])}")
        else:
            self.layers.append(PoolingLayer(agg_type=agg_type, pool_size=pool_size))
            # Calculate output characteristics from new layer
            self.layers[-1].calc_output_chars(self.layers[-2])
            self.segments.append(PoolingSegment(*self.layers[-2:]))

    def feedforward(self, x):
        self.input_layer.act_vals = x
        for segment in self.segments:
            segment.forward_pass()

    def backpropagate(self, truth, updater='sgd', batch_size=50, momentum=True):

        cost = loss(self.layers[-1].act_vals, truth, loss_type=self.loss_func)

        delta = None
        for segment in reversed(self.segments):
            if delta is None:
                delta = (cost * activation(segment.back.raw_vals, func=segment.back.a_func, derivative=True))
            segment.back_propagate(delta)
            delta = segment.setup_next_delta(delta)
            if type(segment.back) is not PoolingLayer:
                segment.update_weights(self.l_rate, self.m_factor, updater=updater, batch_size=batch_size, momentum=momentum)


class SegmentationLayer(object):

    def __init__(self, segment_size):
        self.segment_size = segment_size

    def segment_image(self, image, new_image_shape):
        """Generator to segment `image` into kernal-sized chunks.

        Parameters
        ----------
        image : :obj:`ndarray`
            2d array (an image) or 3d array (array of images).
        new_image_shape : tuple
            Shape of the filtered image.

        Yields
        ------
        :obj:`ndarray`
            2d or 3d depending on `image` dimensions.

        """
        rows_post_filter, cols_post_filter = new_image_shape

        if self.__class__ is PoolingLayer:
            # Segmenting an image for pooling does not overlap pixels, so we
            # must make our iterables reflect that
            row_segs = [i*self.segment_size for i in range(rows_post_filter)]
            col_segs = [i*self.segment_size for i in range(cols_post_filter)]
        else:
            # Segmenting an image for convolution does overlap pixels, so
            # iterable is just the range of rows / columns
            row_segs = range(rows_post_filter)
            col_segs = range(cols_post_filter)

        for i, row in enumerate(row_segs):
            lst_row = row + self.segment_size
            for j, col in enumerate(col_segs):
                lst_col = col + self.segment_size
                try:
                    yield i, j, image[:, row:lst_row, col:lst_col]
                except IndexError:
                    yield i, j, image[row:lst_row, col:lst_col]

    @staticmethod
    def array_crawl(array):
        try:
            _, h, w = array.shape
        except ValueError:
            h, w = array.shape
        for i in range(h):
            for j in range(w):
                yield i, j


class PoolingLayer(SegmentationLayer):

    def __init__(self, agg_type='max', pool_size=2):
        super().__init__(pool_size)
        self.shape = None
        self.agg_type = agg_type
        self.pool_size = pool_size
        self.output_image = None
        self.output_size = None

    @property
    def agg_type(self):
        return self.__agg_type

    @agg_type.setter
    def agg_type(self, agg_type):
        assert agg_type in ['max', 'avg'], f"Unsupported agg_type, {agg_type}"
        self.__agg_type = agg_type

    @property
    def output_image(self):
        return self.__output_image

    @output_image.setter
    def output_image(self, output_image):
        self.__output_image = output_image

    @property
    def raveled_output(self):
        try:
            return self.output_image.ravel().reshape(-1, 1)
        except AttributeError:
            return None

    def process_image(self, image):

        try:
            d, h, w = image.shape
        except ValueError:
            d = 1
            h, w = image.shape

        if (h % 2 != 0) or (w % 2 != 0):
            image = self._pad_image(image)

        new_rows = h // self.pool_size
        new_cols = w // self.pool_size

        filtered_image = np.zeros((d, new_rows, new_cols))

        for row, col, img_seg in self.segment_image(image, (new_rows, new_cols)):
            try:
                filtered_image[:, row, col] = self._pool_segment(img_seg)
            except IndexError:
                filtered_image[row, col] = self._pool_segment(img_seg)

        self.output_image = filtered_image

    def _pool_segment(self, image_segment):

        if self.agg_type == 'max':
            return np.amax(np.amax(image_segment, axis=1), axis=1)

        else:
            return np.average(np.average(image_segment, axis=1), axis=1)

    def calc_output_chars(self, prev_layer):

        if len(prev_layer.shape) == 3:
            self.shape = [prev_layer.shape[0], *[x // self.pool_size for x in prev_layer.shape[-2:]]]
        else:
            self.shape = [prev_layer.num_kernals, *[x // self.pool_size for x in prev_layer.shape[-2:]]]
        self.output_size = np.prod(self.shape)

    @staticmethod
    def _pad_image(image):
        if image.shape[-2] % 2 == 0:
            try:
                image = np.pad(image, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)
            except ValueError:
                image = np.pad(image, ((0, 1), (0, 0)), 'constant', constant_values=0)
        if image.shape[-1] % 2 == 0:
            try:
                image = np.pad(image, ((0, 0), (0, 0), (0, 1)), 'constant', constant_values=0)
            except ValueError:
                image = np.pad(image, ((0, 0), (0, 1)), 'constant', constant_values=0)
        return image


class ConvolutionLayer(SegmentationLayer):
    """Convolution layer for use in a CNN.

    Child class of `SegmentationLayer`; contains kernal (filter) arrays along with
    logic to apply them to an image.

    Parameters
    ----------
    num_kernals : int
        Number of kernals in layer.
    kernal_size : int
        Size of each kernal, i.e. 3 -> 3x3, 5 -> 5x5, etc.
    padding : str
    activation_func : str

    Attributes
    ----------
    kernal_size
    kernals : :obj:`ndarray`
        Array of convolution kernals.

    """
    def __init__(self, num_kernals, kernal_size=3, padding='valid', activation_func='relu'):
        super().__init__(kernal_size)
        self.num_kernals = num_kernals
        self.kernals = None
        self.kernal_size = kernal_size
        # Initialize random kernals
        self._create_kernals()
        self.raw_output = None
        self.output_image = None
        self.shape = None
        self.output_size = None
        self.padding = padding
        self.a_func = activation_func

    @property
    def kernals(self):
        return self.__kernals

    @kernals.setter
    def kernals(self, kernals):
        self.__kernals = kernals

    @property
    def raw_output(self):
        return self.__raw_output

    @raw_output.setter
    def raw_output(self, raw_output):
        self.__raw_output = raw_output
        if raw_output is not None:
            self.output_image = activation(self.raw_output, func=self.a_func)

    @property
    def output_image(self):
        return self.__output_image

    @output_image.setter
    def output_image(self, output_image):
        self.__output_image = output_image

    @property
    def raveled_output(self):
        try:
            return self.output_image.ravel().reshape(-1, 1)
        except AttributeError:
            return None

    def _create_kernals(self):
        """Private method to initialize random kernals.

        """
        args = [self.num_kernals, self.kernal_size, self.kernal_size]
        self.kernals = np.random.randn(*args) * np.sqrt(1 / self.kernal_size)

    def process_image(self, image):
        """Method to apply kernals to an image or array of images.

        Parameters
        ----------
        image : :obj:`ndarray`
            2d array (an image) or 3d array (array of images)

        Returns
        -------
        :obj:`ndarray`

        """
        if self.padding == 'same':
            image = self._pad_image(image)

        try:
            d, h, w = image.shape
        except ValueError:
            d = 1
            h, w = image.shape

        new_rows = (h - self.kernal_size) + 1
        new_cols = (w - self.kernal_size) + 1

        filtered_image = np.zeros((d * self.num_kernals, new_rows, new_cols))

        for row, col, img_seg in self.segment_image(image, (new_rows, new_cols)):
            filtered_image[:, row, col] = self._apply_kernals(img_seg)

        self.raw_output = filtered_image

    def _apply_kernals(self, image_segment):
        """Apply kernals (filters) to an image segment.

        Parameters
        ----------
        image_segment : :obj:`ndarray`
            2d or 3d array representing segment of image/s.

        Returns
        -------
        :obj:`ndarray`
            Filtered image segment.

        """
        if len(image_segment.shape) == 2:
            image_segment = [image_segment]

        sums = np.zeros(self.num_kernals * len(image_segment))
        begin_idx = 0
        for seg in image_segment:
            end_idx = begin_idx + self.num_kernals
            first_row_sum = np.sum(self.kernals * seg, axis=1)
            second_row_sum = np.sum(first_row_sum, axis=1)
            sums[begin_idx:end_idx] = second_row_sum
            begin_idx = end_idx
        return sums

    def _pad_image(self, image):
        """Apply padding to image or array of images.

        Used when user selects to do same padding when filtering an image.

        Parameters
        ----------
        image : :obj:`ndarray`
            2d array (an image) or 3d array (array of images).

        Returns
        -------
        :obj:`ndarray`
            Image or array of images with paddding of 0s.
        """
        pad = (self.kernal_size - 1) // 2
        try:
            return np.pad(image, ((0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
        except ValueError:
            return np.pad(image, (pad, pad), 'constant', constant_values=0)

    def calc_output_chars(self, prev_layer):
        """Method to calculate length of flattened output.

        Parameters
        ----------
        prev_layer : :obj:`InputLayer`, :obj:`PoolingLayer`, or :obj:`ConvoluionLayer`
            The previous layer in the CNN.

        Returns
        -------

        """
        if self.padding == 'valid':
            new_shape = [self.num_kernals, *[x - self.kernal_size + 1 for x in prev_layer.shape[-2:]]]
        else:
            new_shape = [self.num_kernals, *prev_layer.shape[-2:]]

        flat_len = np.prod(new_shape)
        if len(prev_layer.shape) == 3:
            flat_len = flat_len * prev_layer.shape[0]
            new_shape[0] = new_shape[0] * prev_layer.shape[0]
        self.shape = new_shape
        self.output_size = flat_len


class ConvolutionSegment(object):

    def __init__(self, input_layer, output_layer):
        self.front = input_layer
        self.back = output_layer
        self.kernal_updates = np.zeros(self.back.kernals.shape)
        self.bias_updates = np.zeros(self.back.num_kernals).reshape(-1, 1, 1)

    @property
    def front(self):
        """Front layer of `ConvolutionSegment`.

        Supported classes for front layer are `InputLayer`, `ConvolutionLayer`, or `PoolingLayer`.
        Setter method verifies above class restrictions.

        """
        return self.__front

    @front.setter
    def front(self, front):
        assert type(front) in [InputLayer, PoolingLayer, ConvolutionLayer], f"" \
            f"ConvolutionSegment input_layer cannot be {type(front)}; must be " \
            "InputLayer, ConvolutionLayer, or PoolingLayer instance."
        self.__front = front

    @property
    def back(self):
        """Back layer of `ConvolutionSegment`

        Back layer must be `ConvolutionLayer`; setter method verifies this.

        """
        return self.__back

    @back.setter
    def back(self, back):
        assert type(back) is ConvolutionLayer, f"ConvolutionSegment output_layer" \
            f"cannot be {type(back)}; must be ConvolutionLayer instance."
        self.__back = back

    def forward_pass(self):
        """Pass an image through a `ConvolutionLayer` instance.

        """
        if type(self.front) is InputLayer:
            self.back.process_image(self.front.act_vals)
        else:
            self.back.process_image(self.front.output_image)

    def back_propagate(self, delta):

        if type(self.front) is InputLayer:
            image = self.front.act_vals
        else:
            image = self.front.output_image

        for row, col, img_seg in self.back.segment_image(image, self.back.shape[-2:]):
            if type(self.front) is not InputLayer:
                for k in np.arange(0, len(img_seg), self.back.num_kernals):
                    lst_k = k + self.back.num_kernals
                    self.kernal_updates += delta[k:lst_k, row, col].reshape(-1, 1, 1) * img_seg[k:lst_k]
            else:
                self.kernal_updates += delta[:, row, col].reshape(-1, 1, 1) * img_seg

    def setup_next_delta(self, delta):

        if type(self.front) is InputLayer:
            return None

        else:
            next_delta = np.zeros(self.front.shape)
            rot_kern = np.rot90(self.back.kernals, 2, (1, 2))
            rot_del = np.rot90(delta, 2, (1, 2))
            for row, col in self.back.array_crawl(next_delta):
                for fst in np.arange(0, len(delta), self.back.num_kernals):
                    lst = fst + self.back.num_kernals
                    if col < self.back.kernal_size:
                        if row < self.back.kernal_size:
                            hold = rot_del[fst:lst, :row + 1, :col + 1] * rot_kern[:, :row + 1, :col + 1]

                        elif row + self.back.kernal_size > rot_del.shape[1]:
                            # In this case, e_row will be negative so we can utilize the reverse indices in rot_kern
                            e_row = rot_del.shape[1] - (row + self.back.kernal_size)
                            hold = rot_del[fst:lst, row:, :col + 1] * rot_kern[:, :e_row, :col + 1]

                        else:
                            e_row = row + self.back.kernal_size
                            hold = rot_del[fst:lst, row:e_row, :col + 1] * rot_kern[:, :, :col + 1]

                    elif col + self.back.kernal_size > rot_del.shape[2]:
                        e_col = rot_del.shape[2] - (col + self.back.kernal_size)

                        if row < self.back.kernal_size:
                            hold = rot_del[fst:lst, :row + 1, col:] * rot_kern[:, :row + 1, :e_col]

                        elif row + self.back.kernal_size > rot_del.shape[1]:
                            e_row = rot_del.shape[1] - (row + self.back.kernal_size)
                            hold = rot_del[fst:lst, row:, col:] * rot_kern[:, :e_row, :e_col]

                        else:
                            e_row = row + self.back.kernal_size
                            hold = rot_del[fst:lst, row:e_row, col:] * rot_kern[:, :, :e_col]

                    else:
                        e_col = col + self.back.kernal_size

                        if row < self.back.kernal_size:
                            hold = rot_del[fst:lst, :row + 1, col:e_col] * rot_kern[:, :row + 1, :]

                        elif row + self.back.kernal_size > rot_del.shape[1]:
                            e_row = rot_del.shape[1] - (row + self.back.kernal_size)
                            hold = rot_del[fst:lst, row:, col:e_col] * rot_kern[:, :e_row, :]

                        else:
                            e_row = row + self.back.kernal_size
                            hold = rot_del[fst:lst, row:e_row, col:e_col] * rot_kern

                    next_delta[:, row, col] += np.sum(np.sum(hold, axis=1), axis=1)

            return np.rot90(next_delta, 2, (1, 2))

    def update_weights(self, l_rate, m_factor, updater='', batch_size='', momentum=True):

        if updater == 'sgd':
            self.back.kernals -= (l_rate * self.kernal_updates)


class PoolingSegment(object):

    def __init__(self, input_layer, output_layer):
        self.front = input_layer
        self.back = output_layer
        self.unpool = np.zeros(self.front.shape)

    @property
    def front(self):
        return self.__front

    @front.setter
    def front(self, front):
        assert type(front) is ConvolutionLayer, f"PoolingSegment front must be ConvolutionLayer, not {type(front)}"
        self.__front = front

    @property
    def back(self):
        return self.__back

    @back.setter
    def back(self, back):
        assert type(back) is PoolingLayer, f"PoolingSegment back must be PoolingLayer, not {type(back)}"
        self.__back = back

    def forward_pass(self):
        self.back.process_image(self.front.output_image)

    def back_propagate(self, delta):
        for row, col, img_seg in self.back.segment_image(self.front.output_image, self.back.shape[-2:]):
            lst_row = row + self.back.pool_size
            lst_col = col + self.back.pool_size
            self.unpool[:, row:lst_row, col:lst_col][img_seg == img_seg.max()] = img_seg.max()

    def setup_next_delta(self, delta):
        prev_num_kernals = delta.shape[0] // self.front.shape[0]
        summed_delta = np.zeros((self.front.shape[0], *delta.shape[-2:]))
        for fst in np.arange(0, delta.shape[0], prev_num_kernals):
            lst = fst + prev_num_kernals
            summed_delta += delta[fst:lst]

        new_delta = np.zeros(self.unpool.shape)
        for row, col, img_seg in self.back.segment_image(self.unpool, self.back.shape[-2:]):
            new_row = row * self.back.pool_size
            lst_row = new_row + self.back.pool_size
            new_col = col * self.back.pool_size
            lst_col = new_col + self.back.pool_size
            for i, seg in enumerate(img_seg):
                bool_map = seg > 0
                if bool_map.any():
                    new_delta[i, new_row:lst_row, new_col:lst_col][bool_map] = summed_delta[i, row, col]

        return new_delta

