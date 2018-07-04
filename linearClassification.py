import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import math

'''Plan for Linear Regression:
    stochasticGD
    regularized LR
    Validation and Cross Validation
    print weights function to see which weights had most impact
    '''
def getData(filename):
    return pd.read_csv(filename, index_col = 0)

def getChunk(filename, size):
    return pd.read_csv(filename, index_col = 0, nrows = size)

def insertBias(df):
    r, c = df.shape
    df = df.insert(loc=0, column='bias', value = np.ones(r)) 

def createMatrices(df):
    r, c = df.shape
    X = df.iloc[:, 0:c-1]
    y = df.iloc[:, c-1:c]
    insertBias(X)
    return [X, y]
    
def testError(X, y, theta):
    error = 0
    predictions = np.dot(X, theta)
    for i in range(len(y)):
        if int(np.sign(predictions[i])) != y.loc[i]['label']:
            error += 1
    return error / len(y)

def normalEquation(X, y):
    return np.dot(np.linalg.pinv(X), y) 

def predict(theta, x):
    return int(np.sign(np.dot(theta.T, x))) 

def sGDTrain():
    train = getData('clean_train.txt')
    X, y = createMatrices(train)
    theta = stochasticGD(X, y, 10 ** -14)
    return theta

'''returns the weights obtained from normal equation'''
def normalTrain():
    train = getData('clean_train.txt')
    X, y = createMatrices(train)
    theta = normalEquation(X, y)
    return theta

'''returns generalization error obtained from test set'''
def test(theta):
    test = getData('clean_train.txt')
    X, y = createMatrices(test)
    error = testError(X, y, theta)
    return error

