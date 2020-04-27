import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import perceptron
from neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split

# This function gets the name of a CSV file and returns the coordinates of the points
# which are the inputs of the neural network and the labels which are the outputs
def read_data(file_name):
    X = []

    data = pd.read_csv(file_name)

    X.append(data["X1"])
    X.append(data["X2"])

    # Now array X has two rows, first row contains all the numbers for X1 and the
    # second row contains all the numbers for X2 but I want an output in which
    # the number of rows is equal to the number of data and each row contains
    # one X1 and X2 so I got the transpose of array X
    Xt = np.transpose(X)

    return np.array(Xt), data["Label"]


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


X, y = read_data('dataset.csv')

plot(X, y)

# I want to split the data to training and test data but before doing it I shuffle the data to get a different
# training and test data every time I run the code so the result I get may differ each time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Make a perceptron and train it
p = perceptron.Perceptron()
p.fit(X_train, y_train)

# Now I want to test the trained perceptron
predicted_y = []

for x in X_test:
    predicted_y.append(p.predict(x))

plot(X_test, predicted_y)


n = NeuralNetwork(X, y)
n1 = n.fit()
# predicted_y = []
#
# for x in X:
#     predicted_y.append(n1.predict(x))
#
# plot(X, predicted_y)
