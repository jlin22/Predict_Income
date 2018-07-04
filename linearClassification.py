import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math
from general import *

def normalEquation(X, y):
    return np.dot(np.linalg.pinv(X), y) 

def predict(theta, x):
    return int(np.sign(np.dot(theta.T, x))) 

'''returns the weights obtained from normal equation'''
def normalTrain():
    train = getData('clean_train.txt')
    X, y = createMatrices(train)
    theta = normalEquation(X, y)
    return theta

def testError(X, y, theta):
    error = 0
    predictions = np.dot(X, theta)
    for i in range(len(y)):
        if int(np.sign(predictions[i])) != y.loc[i]['label']:
            error += 1
    return error / len(y)

'''returns generalization error obtained from test set'''
def test(theta, error):
    test = getData('clean_train.txt')
    X, y = createMatrices(test)
    error = testError(X, y, theta)
    return error

theta = normalTrain()
print(test(theta))
