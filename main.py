import numpy as np
import pandas as pd
import csv
import time
from utils.nn import NeuralNetwork, NeuralNetwork2

np.random.seed(seed=10)


# nn = NeuralNetwork2([784, 128, 64, 10])

epochs = 1
nn = NeuralNetwork()

nn.add_input_layer(784)
nn.add_connected_layer(128)
nn.add_connected_layer(64)
nn.add_connected_layer(10)

for i in range(epochs):
    start = time.time()
    preds = []
    trues = []
    accuracy = []
    with open("../../Kaggle/digit_recognizer/train.csv") as train:
        reader = csv.reader(train)
        heading = next(reader)

        for row in reader:

            actual_idx = int(row[0])
            y = np.zeros(10).reshape(-1, 1)
            y[actual_idx][0] = 1

            x = np.array(row[1:]).reshape(-1, 1).astype(int)
            x = (x / 255).astype('float32')

            nn.nn_feedforward(x, a_function='relu')
            preds.append(np.argmax(nn.segments[-1].back.a_vals))
            trues.append(actual_idx)
            nn.nn_backpropagate(y, a_function='relu')

    for pred, true in zip(preds, trues):
        accuracy.append(pred == true)

    print(f'epoch {1+i} completed in {time.time() - start}')
    print(f'correct predictions: {sum(accuracy)}')

ids = []
guess = []
with open('../../Kaggle/digit_recognizer/test.csv') as test:
    reader = csv.reader(test)
    header = next(reader)
    for i, row in enumerate(reader):
        ids.append(i+1)

        x = np.array(row).reshape(-1, 1).astype(int)
        x = (x / 255).astype('float32')

        pred = nn.nn_feedforward(x, a_func='relu')
        guess.append(np.argmax(pred))

moo = pd.DataFrame()
moo['ImageId'] = ids
moo['Label'] = guess

moo.to_csv("../../Kaggle/digit_recognizer/submit.csv", index=False)

