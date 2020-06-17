import warnings
import numpy as np
from .nn import NeuralNetwork, InputLayer


class ConvolutionLayer(InputLayer):

    def __init__(self, kernals, kernal_size=3):
        super().__init__(kernals)
        self.kernal_size = kernal_size

    def _create_filters(self):


