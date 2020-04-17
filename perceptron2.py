import numpy as np


class Perceptron2:
    def __init__(self, n_epoch=100, lr=0.05):
        self.n_epoch = n_epoch
        self.lr = lr
        self.w = []
        self.b0 = 0.0
        self.v = []
        self.b1 = 0.0
        self.u = []
        self.b2 = 0.0

    def fit(self, X, y):
        self.w = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b0 = np.random.normal(loc=0.0, scale=0.01)

        self.v = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b1 = np.random.normal(loc=0.0, scale=0.01)

        self.u = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b2 = np.random.normal(loc=0.0, scale=0.01)

        for _ in range(self.n_epoch):
            grad_b0 = 0
            grad_w = np.zeros(X.shape[1])

            grad_b1 = 0
            grad_v = np.zeros(X.shape[1])

            grad_b2 = 0
            grad_u = np.zeros(X.shape[1])

            for xi, label in zip(X, y):
                predicted_y = self.predict(xi)
                c = 2 * (predicted_y - label) * predicted_y * (1 - predicted_y)

                z0 = 1 / (1 + np.exp(-(np.dot(xi, self.w) + self.b0)))
                grad_b0 += c * self.u[0] * z0 * (1 - z0)
                grad_w += c * self.u[0] * z0 * (1 - z0) * xi

                z1 = 1 / (1 + np.exp(-(np.dot(xi, self.v) + self.b1)))
                grad_b1 += c * self.u[1] * z1 * (1 - z1)
                grad_v += c * self.u[1] * z1 * (1 - z1) * xi

                grad_b2 += c
                grad_u += c * np.array([z0, z1])

            self.w -= (self.lr * grad_w) / y.size
            self.b0 -= (self.lr * grad_b0) / y.size

            self.v -= (self.lr * grad_v) / y.size
            self.b1 -= (self.lr * grad_b1) / y.size

            self.u -= (self.lr * grad_u) / y.size
            self.b2 -= (self.lr * grad_b2) / y.size

        return self

    def predict(self, X):
        z0 = 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b0)))
        z1 = 1 / (1 + np.exp(-(np.dot(X, self.v) + self.b1)))
        return 1 / (1 + np.exp(-(np.dot([z0, z1], self.u) + self.b2)))
