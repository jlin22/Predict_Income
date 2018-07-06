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
    
def error(pred, test):
    error = 0
    for i in range(pred.shape[0]):
        if pred[i] != int(test.loc[i]):
            error += 1
    return error / pred.shape[0]

def validation(fn):
    data = getData(fn)
    data.sample(frac=1)
    data = data.reset_index(drop=True)
    validation = data[:int(data.shape[0] * 0.3)]
    return validation, data
