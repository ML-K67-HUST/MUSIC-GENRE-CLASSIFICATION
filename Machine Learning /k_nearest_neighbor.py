
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
df = df.iloc[0:,1:]
x = df.drop(['length','label'], axis = 1)
y = df['label']

KNN = make_pipeline(MinMaxScaler(), 
                    KNeighborsClassifier(n_neighbors=5))
KNN.fit(x, y)
with open('knn_model.pkl','wb') as file:
  pickle.dump(KNN, file)
