import numpy as np
from general import *
import random
from sklearn import svm 
from sklearn import preprocessing

def testLinearSize():
    val, data = validation('clean_train.txt')
    X_val, y_val = createMatrices(val)
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
    val, data = validation('clean_train.txt')
    X_val, y_val = createMatrices(val)
    predictions = []
    errors = []
    sizes = [0.001, 0.005, 0.01, 0.03, 0.06, 0.09, 0.12]
    sizes = [int(size * data.shape[0]) for size in sizes]
    for size in sizes:
        train = data[size:]
        train = data.reset_index(drop=True)
        X, y = createMatrices(train)
        y = [int(y.loc[i]) for i in range(y.shape[0])]
        clf = svm.SVC()
        clf.fit(X, y)
        predictions.append(clf.predict(X_val))
    for prediction in predictions:
        errors.append(error(prediction, y_val)) 
    return errors, sizes

#errors, sizes = testLinearSize()
errors, sizes = testRBFsize()
print(errors)
print(sizes)
