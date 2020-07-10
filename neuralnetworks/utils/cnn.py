import numpy as np
from neuralnetworks.utils.nn import NeuralNetwork, InputLayer, ConnectedLayer, ConnectedSegment


class ConvolutionalNeuralNet(NeuralNetwork):

    def __init__(self, seed=10, l_rate=0.01, m_factor=0.9):
        super().__init__(seed=seed, l_rate=l_rate, m_factor=m_factor)

    def add_connected_layer(self, size):
        """Method to add `ConnectedLayer` of inputted size.

        Overwrites method of same name from `NeuralNetwork` class.

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
            # After each ConnectedLayer is added, create a ConnectedSegment from the last two elements in self.layers.
            # Using elements from self.layers to create the ConnectedSegment instance allows the chain of InputLayer and
            # ConnectedLayer references to be maintained. This is crucial for this architecture
            if type(self.layers[-2]) in [ConvolutionLayer, PoolingLayer]:
                self.layers[-2].raveled_output = self.layers[-2].output_image.ravel().reshape(-1, 1)
            self.segments.append(ConnectedSegment(*self.layers[-2:]))
        else:
            raise AssertionError("NeuralNetwork instance must contain an InputLayer before adding a ConnectedLayer")

    def add_convolution_layer(self, num_kernals, kernal_size=3, padding='valid'):
        if [type(i) for i in self.layers].__contains__(InputLayer):
            temp = ConvolutionLayer(num_kernals, kernal_size=kernal_size, padding=padding)
            #temp.flat_len = self.layers[-1].shape
            self.layers.append(temp)
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
            self.segments.append(PoolingSegment(*self.layers[-2:]))

    def feedforward(self, x, a_function='relu'):

        self.input_layer = x
        for segment in self.segments:
            segment.forward_pass()


class SegmentationLayer(object):

    def __init__(self, segment_size):
        self.segment_size = segment_size

    def _segment_image(self, image, new_image_shape):
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

        for row in row_segs:
            lst_row = row + self.segment_size
            for col in col_segs:
                lst_col = col + self.segment_size
                try:
                    yield row, col, image[:, row:lst_row, col:lst_col]
                except IndexError:
                    yield row, col, image[row:lst_row, col:lst_col]


class PoolingLayer(SegmentationLayer):

    def __init__(self, agg_type='max', pool_size=2):
        super().__init__(pool_size)
        self.agg_type = agg_type
        self.pool_size = pool_size
        self.output_image = None
        self.raveled_output = None
        self.output_size = None
        self.output_shape = None

    @property
    def agg_type(self):
        return self.__agg_type

    @agg_type.setter
    def agg_type(self, agg_type):
        assert agg_type in ['max', 'avg'], f"Unsupported agg_type, {agg_type}"
        self.__agg_type = agg_type

    def process_image(self, image):

        try:
            d, h, w = image.shape
        except ValueError:
            d = 1
            h, w = image.shape

        new_rows = h // self.pool_size
        new_cols = w // self.pool_size

        filtered_image = np.zeros((d, new_rows, new_cols))

        for row, col, img_seg in self._segment_image(image, (new_rows, new_cols)):
            lst_row = row + self.pool_size
            lst_col = col + self.pool_size
            try:
                filtered_image[:, row:lst_row, col:lst_col] = self._pool_segment(img_seg)
            except IndexError:
                filtered_image[row:lst_row, col:lst_col] = self._pool_segment(img_seg)

        self.output_image = filtered_image

    def _pool_segment(self, image_segment):

        if self.agg_type == 'max':
            pool = np.amax(np.amax(image_segment, axis=1), axis=1)

        else:
            pool = np.average(np.average(image_segment, axis=1), axis=1)

        return pool.reshape((len(image_segment), 1, 1))

    def calc_output_chars(self, prev_layer):
        # TODO: this ish
        return


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
        self.size = num_kernals
        self.padding = padding
        self.a_func = activation_func
        self.kernals = None
        self.kernal_size = kernal_size
        self.raw_output = None
        self.output_image = None
        self.raveled_output = None
        self.output_shape = None
        self.output_size = None
        # Initialize random kernals
        self._create_kernals()

    @property
    def kernals(self):
        return self.__kernals()

    @kernals.setter
    def kernals(self, kernals):
        self.__kernals = kernals

    def _create_kernals(self):
        """Private method to initialize random kernals.

        """
        args = [self.size, self.kernal_size, self.kernal_size]
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

        filtered_image = np.zeros((d*self.size, new_rows, new_cols))

        for row, col, img_seg in self._segment_image(image, (new_rows, new_cols)):
            lst_row = row + self.kernal_size
            lst_col = col + self.kernal_size
            try:
                filtered_image[:, row:lst_row, col:lst_col] = self._apply_kernals(img_seg)
            except IndexError:
                filtered_image[row:lst_row, col:lst_col] = self._apply_kernals(img_seg)

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

        sums = np.zeros((self.size * len(image_segment), 1, 1))
        begin_idx = 0
        for seg in image_segment:
            end_idx = begin_idx + self.size
            first_row_sum = np.sum(self.kernals * seg, axis=1)
            second_row_sum = np.sum(first_row_sum, axis=1).reshape((self.size, 1, 1))
            sums[begin_idx:end_idx, 1, 1] = second_row_sum
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
            if type(prev_layer) is InputLayer:
                new_shape = [self.size, *[x - self.kernal_size + 1 for x in prev_layer.shape[-2:]]]
                flat_len = np.prod(new_shape)
                if len(prev_layer.shape) == 3:
                    flat_len = flat_len * prev_layer.shape[0]
                    new_shape[0] = new_shape[0] * prev_layer.shape[0]
                self.output_shape = new_shape
                self.output_size = flat_len
            else:
                return


        else:
            return


class ConvolutionSegment(object):

    def __init__(self, input_layer, output_layer):
        self.front = input_layer
        self.back = output_layer

    @property
    def front(self):
        """Front layer of `ConvolutionSegment`.

        Supported classes for front layer are `InputLayer`, `ConvolutionLayer`, or `PoolingLayer`.
        Setter method verifies above class restrictions.

        """
        return self.__front

    @front.setter
    def front(self, front):
        assert type(front) in [InputLayer, PoolingLayer], f"" \
            f"ConvolutionSegment input_layer cannot be {type(front)}; must be " \
            "InputLayer or PoolingLayer instance."
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
            self.back.process_image(self.front.a_vals)
        else:
            self.front.output_image = self.activation(self.front.raw_output, self.front.a_func)
            self.back.process_image(self.front.output_image)

    @staticmethod
    def activation(val, func='', derivative=False):
        """
        See Also
        --------
        ConnectedSegment

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


class PoolingSegment(object):

    def __init__(self, input_layer, output_layer):
        self.front = input_layer
        self.back = output_layer

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

