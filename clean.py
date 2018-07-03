import pandas as pd

def clean(filename, char):
    if filename == 'adult.test.txt':
        df = pd.read_csv(filename, skiprows = 1)
    else:
        df = pd.read_csv(filename)
    df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    r, c = df.shape
    for i in df.index:
        if char in list(df[i:i+1]):
            df.drop(i)
    return df 
    
def oneHot(df):
    y = df['label']
    #education-num uses convert to number, while the other categorical fields use one-hot encoding
    df = df.drop(['education', 'label'], axis=1)
    categorical = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    for category in categorical:
        dummy = pd.get_dummies(df[category])
        df = pd.concat([df, dummy], axis=1)
        df = df.drop(category, axis=1)
    df = pd.concat([df, y], axis=1)
    return df

def convertOutput(df):
    def convert(x):
        return 1 if '>' in x else -1
    df['label'] = df['label'].apply(convert)
    return df

#lets try merging the data frames and then separating
train = clean('adult.data.txt', '?')
r, c = train.shape
test = clean('adult.test.txt', '?')
merged = pd.concat([train, test])
merged = oneHot(merged)
merged = convertOutput(merged)
test = merged[0:r]
train = merged[r:]
train.to_csv('clean_train.txt')
test.to_csv('clean_test.txt')

