import pandas as pd
import numpy as np

'''Plan for Linear Regression:
    convertOutputs
    stochasticGD
    regularized LR
    Validation and Cross Validation
    '''
def getData(filename):
    return pd.read_csv(filename, index_col = 0)

def createMatrices(df):
    r, c = df.shape
    X = df.iloc[:, 0:c-1]
    y = df.iloc[:, c-1:c]
    return [X, y]
    
def stochasticGD(X, y):
    pass

def costFunction(X, y, theta):
    pass
    
def normalEquation(X, y):
    return np.dot(np.linalg.pinv(X), y) 
    
train = getData('clean_train.txt')
X, y = createMatrices(train)
print(X[1:2])
#deal with string data
#theta = normalEquation(X, y)
#print(theta)
