import pandas as pd

def clean(file_name, char):
    if file_name == 'adult.test.txt':
        df = pd.read_csv(file_name, skiprows = 1)
    else:
        df = pd.read_csv(file_name)
    r, c = df.shape
    for i in df.index:
        if char in list(df[i:i+1]):
            df.drop(i)
    return df 
    

train = clean("adult.data.txt", '?')
test = clean("adult.test.txt", '?')
print('?' in train)
print('?' in test)



