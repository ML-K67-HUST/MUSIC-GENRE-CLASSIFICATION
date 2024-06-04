
import pandas as pd
import pickle
import os
from sklearn.neighbors import KNeighborsClassifier
from preprocess import preprocess_
import numpy as np

df = pd.read_csv('../dataset/Dataset.csv')

X, y = preprocess_(df)

KNN = KNeighborsClassifier(n_neighbors=2, weights='distance', p=1)
KNN.fit(X, y)
if os.path.exists('saved_model/knn_model.pkl'):
    os.remove('saved_model/knn_model.pkl')
with open('saved_model/knn_model.pkl','wb') as file:
    pickle.dump(KNN, file)

