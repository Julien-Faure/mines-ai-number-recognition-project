import sys

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ne plus être limité lors des prints
# np.set_printoptions(threshold=sys.maxsize)

digits = datasets.load_digits()
x = digits.images.reshape((len(digits.images), -1))
y = digits.target

# séparation du dataset en données "d'apprentissage" et de test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state=0)

model = LogisticRegression(verbose=True, max_iter=4000, solver='lbfgs')
model.fit(x_train, y_train)

predictions = model.predict(x_test)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

