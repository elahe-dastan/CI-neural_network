import numpy as np
import random


class NeuralNetwork:
    def __init__(self, n_inputs, n_epoch=10000, lr=0.7):
        self.n_epoch = n_epoch
        self.lr = lr
        self.p1 = Node(n_inputs)
        self.p2 = Node(n_inputs)
        self.p3 = Node(n_inputs)

    # The neural network will be trained using this function in which the coefficients of each perceptron get<br/>
    # closer and closer to what they should be in the number of epochs
    def fit(self, X, y):
        np.random.seed(1)
        for _ in range(self.n_epoch):
            r = np.random.randint(low=0, high=len(y))
            xi = X[r]
            label = y[r]

            predicted_z0 = self.p1.predict(xi)
            predicted_z1 = self.p2.predict(xi)
            predicted_y = self.p3.predict([predicted_z0, predicted_z1])

            c = 2 * (predicted_y - label) * predicted_y * (1 - predicted_y)

            self.p1.grad_b = c * self.p3.w[0] * predicted_z0 * (1 - predicted_z0)
            self.p1.grad_w = c * self.p3.w[0] * predicted_z0 * (1 - predicted_z0) * xi

            self.p2.grad_b = c * self.p3.w[1] * predicted_z1 * (1 - predicted_z1)
            self.p2.grad_w = c * self.p3.w[1] * predicted_z1 * (1 - predicted_z1) * xi

            self.p3.grad_b = c
            self.p3.grad_w = c * np.array([predicted_z0, predicted_z1])

            self.p1.improve(self.lr, y.size)
            self.p2.improve(self.lr, y.size)
            self.p3.improve(self.lr, y.size)

        return self

    def predict(self, X):
        z0 = self.p1.predict(X)
        z1 = self.p2.predict(X)
        return self.p3.predict([z0, z1])


class Node:
    def __init__(self, n_columns):
        self.w = np.random.normal(loc=0.0, scale=0.5, size=n_columns)
        self.b = np.random.normal(loc=0.0, scale=0.5)
        self.n_columns = n_columns
        self.grad_w = np.zeros(n_columns)
        self.grad_b = 0

    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))

    # Amount if coefficients will be subtracted by derivation of cost
    def improve(self, lr, n):
        self.w -= (lr * self.grad_w) / n
        self.b -= (lr * self.grad_b) / n
