import pandas as pd
import numpy as np
from preprocess import preprocess_
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import pickle
import os

df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
X, y = preprocess_(df)


clf = svm.SVC(kernel='rbf', gamma='scale', C=50,probability=True)

clf.fit(X,y)

if os.path.exists('saved_model/svm_model.pkl'):
    os.remove('saved_model/svm_model.pkl')
with open('saved_model/svm_model.pkl','wb') as file:
    pickle.dump(clf, file)

