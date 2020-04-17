import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot():
    plt.title('Scatter plot')
    plt.xlabel('x1')
    plt.ylabel('x2')

    for i, label in enumerate(data["Label"]):
        if label == 1:
            plt.scatter(x1[i], x2[i], c="red")
        else:
            plt.scatter(x1[i], x2[i], c="blue")

    plt.show()


def train(n_epoch, lr):
    n = len(data)
    w1 = 1
    w2 = 1
    b = 1

    for i in range(n_epoch):
        grad_w1 = 0
        grad_w2 = 0
        grad_b = 0
        for j in range(n):
            y = 1 / (1 + np.exp(-(w1 * x1[j] + w2 * x2[j] + b)))
            # cost = -label[j] * np.log(y) Why?!!
            c = -labels[j] * (1 - y)
            grad_b += c
            grad_w1 += c * x1[j]
            grad_w2 += c * x2[j]
        w1 = w1 - (lr * grad_w1) / n
        w2 = w2 - (lr * grad_w2) / n
        b = b - (lr * grad_b) / n

    return w1, w2, b


def predict(w1, w2, b):
    n = len(data)

    for j in range(n):
        y = 1 / (1 + np.exp(-(w1 * x1[j] + w2 * x2[j] + b)))
        if y > 0:
            plt.scatter(x1[j], x2[j], c="red")
        else:
            plt.scatter(x1[j], x2[j], c="blue")
    plt.show()


csv_file = 'dataset.csv'
data = pd.read_csv(csv_file)

x1 = data["X1"]
x2 = data["X2"]
labels = data["Label"]

plot()

w_1, w_2, bb = train(1000, 0.05)

predict(w_1, w_2, bb)
