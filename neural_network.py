import numpy as np


class NeuralNetwork:
    def __init__(self, X, y, n_epoch=5000, lr=0.5):
        self.n_epoch = n_epoch
        self.lr = lr
        self.p1 = Node(X.shape[1])
        self.p2 = Node(X.shape[1])
        self.p3 = Node(X.shape[1])
        self.X = X
        self.y = y

    def fit(self):
        for _ in range(self.n_epoch):
            for xi, label in zip(self.X, self.y):
                predicted_z0 = self.p1.predict(xi)
                predicted_z1 = self.p2.predict(xi)
                predicted_y = self.p3.predict([predicted_z0, predicted_z1])

                c = 2 * (predicted_y - label) * predicted_y * (1 - predicted_y)

                self.p1.grad_b += c * self.p3.w[0] * predicted_z0 * (1 - predicted_z0)
                self.p1.grad_w += c * self.p3.w[0] * predicted_z0 * (1 - predicted_z0) * xi

                self.p2.grad_b += c * self.p3.w[1] * predicted_z1 * (1 - predicted_z1)
                self.p2.grad_w += c * self.p3.w[1] * predicted_z1 * (1 - predicted_z1) * xi

                self.p3.grad_b += c
                self.p3.grad_w += c * np.array([predicted_z0, predicted_z1])

            self.p1.improve(self.lr, self.y.size)
            self.p2.improve(self.lr, self.y.size)
            self.p3.improve(self.lr, self.y.size)

        return self

    def predict(self, X):
        z0 = self.p1.predict(X)
        z1 = self.p2.predict(X)
        return self.p3.predict([z0, z1])


class Node:
    def __init__(self, n_columns):
        self.w = np.random.normal(loc=0.0, scale=0.01, size=n_columns)
        self.b = np.random.normal(loc=0.0, scale=0.01)
        self.n_columns = n_columns
        self.grad_w = np.zeros(n_columns)
        self.grad_b = 0

    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))

    def improve(self, lr, n):
        self.w -= (lr * self.grad_w) / n
        self.grad_w = np.zeros(self.n_columns)
        self.b -= (lr * self.grad_b) / n
        self.grad_b = 0