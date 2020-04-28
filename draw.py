import matplotlib.pyplot as plt
import numpy as np


def plot(X, y):
    plt.xlabel('x1')
    plt.ylabel('x2')

    # If the label is predicted as 1 it will be shown by red color and if this is
    # predicted as 0 it's color will be blue
    for xi, label in zip(X, y):
        if label > 0.5:
            plt.scatter(xi[0], xi[1], c="red")
        else:
            plt.scatter(xi[0], xi[1], c="blue")

    plt.show()


def border(X, neuron):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1, x2 = np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02)

    grid = []

    for xx1 in x1:
        for xx2 in x2:
            grid.append([xx1, xx2])

    y = []
    for row in grid:
        y.append(neuron.predict(row))

    plot(grid, y)
