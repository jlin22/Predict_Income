import numpy as np 
from linearClassification import *


def sigmoid(z):
    return 1 / (1 + np.exp(z))

def logisticCost(X, y, theta):
    h = np.dot(X, theta)
    return np.sum(y * h + (1 - y) * (1 - h))

test = getData('clean_test.txt')
X, y = createMatrices(test)
