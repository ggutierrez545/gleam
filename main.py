import numpy as np
import pandas as pd
import csv
import time
from utils.nn import NeuralNetwork


epochs = 5

nn = NeuralNetwork([784, 128, 64, 10])

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

            prediction = nn.feedforward(x, a_func='relu')
            preds.append(np.argmax(prediction))
            trues.append(actual_idx)
            changes = nn.backprop(prediction, y, a_func='relu')
            nn._update_weights(changes, updater='mini_batch')

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

        pred = nn.feedforward(x, a_func='relu')
        guess.append(np.argmax(pred))

moo = pd.DataFrame()
moo['ImageId'] = ids
moo['Label'] = guess

moo.to_csv("../../Kaggle/digit_recognizer/submit.csv", index=False)

