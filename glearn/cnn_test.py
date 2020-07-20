import csv
import numpy as np
from utils.cnn import ConvNeuralNetwork

test = ConvNeuralNetwork([100, 100, 10])
test._add_convolution_layer([8, 8, 8], [3, 3, 3])

with open("../../Kaggle/digit_recognizer/train.csv") as train:
    reader = csv.reader(train)
    heading = next(reader)

    image = np.array(next(reader)[1:]).reshape(28, 28).astype(int)

pred = test.cnn_feedforward(image)


