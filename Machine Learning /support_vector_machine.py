import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn import svm
import pickle
import os

df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
df = df.iloc[0:,1:]
X = df.drop(['length','label'], axis = 1)
y = df['label']

df.label = pd.Categorical(df.label)
y = np.array(df.label.cat.codes)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


clf = svm.SVC(kernel='rbf', gamma='scale', C=50,probability=True)

clf.fit(X,y)

if os.path.exists('saved_model/svm_model.pkl'):
    os.remove('saved_model/svm_model.pkl')
with open('saved_model/svm_model.pkl','wb') as file:
    pickle.dump(clf, file)

