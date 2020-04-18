import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from neural_network import NeuralNetwork


def read_data(file_name):
    X = []

    data = pd.read_csv(file_name)

    X.append(data["X1"])
    X.append(data["X2"])

    return np.array(np.transpose(X)), data["Label"]


def plot(X, y):
    plt.title('Scatter plot')
    plt.xlabel('x1')
    plt.ylabel('x2')

    for xi, label in zip(X, y):
        if label > 0.5:
            plt.scatter(xi[0], xi[1], c="red")
        else:
            plt.scatter(xi[0], xi[1], c="blue")

    plt.show()


X, y = read_data('dataset.csv')

plot(X, y)

n = NeuralNetwork(X, y)
n1 = n.fit()
predicted_y = []

for x in X:
    predicted_y.append(n1.predict(x))

plot(X, predicted_y)
