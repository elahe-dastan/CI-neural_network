import pandas as pd
import numpy as np

import perceptron
from draw import plot, border
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


X, y = read_data('dataset.csv')

plot(X, y)

# I want to split the data to training and test data but before doing it I shuffle the data to get a different
# training and test data every time I run the code so the result I get may differ each time.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Make a perceptron and train it
p = perceptron.Perceptron()
p.fit(X_train, y_train)

border(X, p)

# Now I want to test the trained perceptron
predicted_y = []

for x in X_test:
    predicted_y.append(p.predict(x))

plot(X_test, predicted_y)

# Make a neural network and train it
n = NeuralNetwork(X.shape[1])
n.fit(X_train, y_train)

# Test the trained neural network
predicted_y = []

for x in X_test:
    predicted_y.append(n.predict(x))

plot(X_test, predicted_y)
