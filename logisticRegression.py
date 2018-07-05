import numpy as np 
from general import *
import random
from sklearn.linear_model import LogisticRegression

train = getData('clean_train.txt')
X, y = createMatrices(train)
y = [int(y.loc[i]) for i in range(y.shape[0])]
lr = LogisticRegression(penalty = 'l2', C = 10 ** 14) 
lr.fit(X, y)
test = getData('clean_test.txt')
pred_x, corr_y = createMatrices(test)
pred_y = lr.predict(pred_x)
error = 0
for i in range(pred_y.shape[0]):
    if pred_y[i] != int(corr_y.loc[i]):
        error += 1
print(error/pred_y.shape[0])




