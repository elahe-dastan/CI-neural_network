import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

csv_file = 'dataset.csv'

data = pd.read_csv(csv_file)

x1 = data["X1"]
x2 = data["X2"]
label = data["Label"]

# plt.title('Scatter plot')
# plt.xlabel('x')
# plt.ylabel('y')
#
# for i, l in enumerate(data["Label"]):
#     if l == 1:
#         plt.scatter(x[i], y[i], c="red")
#     else:
#         plt.scatter(x[i], y[i], c="blue")
#
# plt.show()



# initialize W and b
#
n_epoch = 100
# lr = ?
n = 180
w1 = 1
w2 = 1
b = 1

for i in range(n_epoch):
    gradw1 = 0
    gradw2 = 0
    gradb = 0
    for j in range(n):
        y = 1 / (1 + np.exp(-(w1 * x1[j] + w2 * x2[j] + b)))
        cost = -label[j] * np.log(y)
