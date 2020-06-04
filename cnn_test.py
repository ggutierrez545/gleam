import csv
import numpy as np
from utils.cnn import ConvNeuralNetwork

test = ConvNeuralNetwork([100, 100, 10])
test._add_convolution_layer([8, 8], [3, 5])

with open("../../Kaggle/digit_recognizer/train.csv") as train:
    reader = csv.reader(train)
    heading = next(reader)

    image = np.array(next(reader)[1:]).reshape(28, 28).astype(int)

img = test._convolve_image(image, padding='same')

img.shape
