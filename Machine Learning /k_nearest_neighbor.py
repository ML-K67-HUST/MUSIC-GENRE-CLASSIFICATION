
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
df = df.iloc[0:,1:]
x = df.drop(['length','label'], axis = 1)
y = df['label']
df.label = pd.Categorical(df.label)
y = np.array(df.label.cat.codes)
KNN = make_pipeline(StandardScaler(), 
                    KNeighborsClassifier(n_neighbors=1))
KNN.fit(x, y)
with open('knn_model.pkl','wb') as file:
  pickle.dump(KNN, file)
