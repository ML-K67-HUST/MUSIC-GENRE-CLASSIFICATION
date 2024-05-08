
import pandas as pd
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
df = df.iloc[0:,1:]
X = df.drop(['length','label'], axis = 1)
y = df['label']

df.label = pd.Categorical(df.label)
y = np.array(df.label.cat.codes)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X, y)
if os.path.exists('saved_model/knn_model.pkl'):
    os.remove('saved_model/knn_model.pkl')
with open('saved_model/knn_model.pkl','wb') as file:
    pickle.dump(KNN, file)

