import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_(df):
    df = df.iloc[0:,1:]
    X = df.drop(['length','label'], axis = 1)
    y = df['label']

    df.label = pd.Categorical(df.label)
    y = np.array(df.label.cat.codes)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y