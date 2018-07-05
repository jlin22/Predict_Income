import numpy as np 
from general import *
import random


def sigmoid(z):
    return 1 / (1 + np.exp(z))

def logisticCost(X, y, theta):
    h = np.dot(X, theta)
    return np.sum(y * h + (1 - y) * (1 - h))

def sGD(X, y, theta, alpha):
    m, c = X.shape
    theta = np.zeros((c, 1))
    epsilon = 0.001
    for i in range(len(theta)):
        theta[i] = 2 * epsilon * random.uniform(0, 1) - epsilon
    for i in range(10):
        cost_prev = logisticCost(X, y, theta)
        print(X.loc[i].shape)
        print(y.shape)
        gradient = 1/m * np.dot(X.T, y - sigmoid(np.dot(X, theta)))
        theta += alpha * gradient
        cost_cur = logisticCost(X, y, theta)
    return theta

test = getData('clean_test.txt')
X, y = createMatrices(test)
theta = np.zeros(0)
theta = sGD(X, y, theta, 0.1)
