import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

digits = datasets.load_digits()

# We instantiate the model specifying which solver we want to use to find the parameters
clf_lr = LogisticRegression(solver='lbfgs', max_iter=9999)  # clf = classifier lr = logistic regression
# print(type(digits.images[98]))
array = np.array([])
for digit in digits.images:
    vector = np.reshape(digit, -1)
    array = np.concatenate(array, vector)

print(array)
# print(digits.images[98])
plt.imshow(digits.images[98], cmap='binary')
plt.show()
