import warnings
import numpy as np
from .nn import NeuralNetwork, InputLayer


class ConvolutionLayer(InputLayer):
    """Convolution layer for use in a CNN.

    Child class of `InputLayer`; contains kernal (filter) arrays along with
    logic to apply them to an image.

    Parameters
    ----------
    num_kernals : int
        Number of kernals in layer.
    kernal_size : int
        Size of each kernal, i.e. 3 -> 3x3, 5 -> 5x5, etc.

    Attributes
    ----------
    kernal_size
    kernals : :obj:`ndarray`
        Array of convolution kernals.

    See Also
    --------
    InputLayer

    """

    def __init__(self, num_kernals, kernal_size=3):
        super().__init__(num_kernals)
        self.kernals = None
        self.kernal_size = kernal_size
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

        """
        last_vseg = 0
        for vert_seg, img_seg, last_seg in self._segment_image(image):

            if vert_seg != last_vseg:
                try:
                    new_img = np.hstack((new_img, temp))
                    del temp
                except NameError:
                    new_img = temp
                    del temp





        return

    def _segment_image(self, image):
        """Generator to segment `image` into kernal-sized chunks.

        Parameters
        ----------
        image : :obj:`ndarray`
            2d array (an image) or 3d array (array of images).

        Yields
        ------
        :obj:`ndarray`
            2d or 3d depending on `image` dimensions.

        """
        # Allowing image to be an 2d array (an image) or 3d array (an array of images)
        try:
            _, h, w = image.shape
        except ValueError:
            h, w = image.shape

        # Calculate how many times this layer's kernal can fit across the image
        row_segments = (h - self.kernal_size) + 1
        col_segments = (w - self.kernal_size) + 1

        # Track last row segment, column segment, and overall segment.
        # Helpful when building convolved images.
        lst_rseg, lst_cseg, lst_seg = False, False, False

        for row in range(row_segments):
            lst_row = row + self.kernal_size
            # If we've reached the last row segment
            if row == (row_segments - 1): lst_rseg = True
            for col in range(col_segments):
                lst_col = col + self.kernal_size
                # If we've reached last column segment
                if col == (col_segments - 1): lst_cseg = True
                # If we've reached last overall segment
                if lst_rseg and lst_cseg: lst_seg = True
                try:
                    yield row, image[:, row:lst_row, col:lst_col], lst_seg
                except IndexError:
                    yield row, image[row:lst_row, col:lst_col], lst_seg








