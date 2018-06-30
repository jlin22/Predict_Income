import pandas as pd

def clean(file_name):
    test = pd.read_csv(file_name)
    r, c = test.shape
    for i in test.index:
        if '?' in list(test[i:i+1]):
            test.drop(i)
    return test
    
test = clean("adult.data.txt")
print('?' in test)

