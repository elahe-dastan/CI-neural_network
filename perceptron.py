import numpy as np


class Perceptron:
    def __init__(self, n_epoch=1000, lr=0.5):
        self.n_epoch = n_epoch
        self.lr = lr
        self.w = []
        self.b = 0.0

    # This functions tries to train the neural network
    def fit(self, X, y):
        self.w = np.random.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b = np.random.normal(loc=0.0, scale=0.01)

        for _ in range(self.n_epoch):
            grad_b = 0
            grad_w = np.zeros(X.shape[1])
            for xi, label in zip(X, y):
                predicted_y = self.predict(xi)
                c = -(label * (1 - predicted_y) - (1 - label) * predicted_y)
                grad_b += c
                grad_w += xi * c
            self.w -= (self.lr * grad_w) / y.size
            self.b -= (self.lr * grad_b) / y.size

        return self

    # This method predicts a label for a given data using the neural network
    def predict(self, X):
        return 1 / (1 + np.exp(-(np.dot(X, self.w) + self.b)))
