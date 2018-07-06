import numpy as np
from general import *
import random
from sklearn import svm 
from sklearn import preprocessing

def testLinearSize():
    data = getData('clean_train.txt')
    data.sample(frac=1)
    data = data.reset_index(drop=True)
    validation = data[:int(data.shape[0] * 0.3)]
    validation = data.reset_index(drop=True) 
    X_val, y_val = createMatrices(validation)
    predictions = []
    errors = []
    sizes = [0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    sizes = [int(size * data.shape[0]) for size in sizes]
    for size in sizes:
        train = data[size:]
        train = data.reset_index(drop=True)
        X, y = createMatrices(train)
        y = [int(y.loc[i]) for i in range(y.shape[0])]
        clf = svm.LinearSVC()
        clf.fit(X, y)
        predictions.append(clf.predict(X_val))
    for prediction in predictions:
        errors.append(error(prediction, y_val)) 
    return errors, sizes

def testRBFsize():
    pass

errors, sizes = testLinearSize()
print(errors)
print(sizes)
