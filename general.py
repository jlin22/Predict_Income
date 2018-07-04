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
    

