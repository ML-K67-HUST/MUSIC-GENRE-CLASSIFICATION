
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

def KNN_predict(input):
  df = pd.read_csv('GTZAN/Data/features_30_sec.csv')
  df = df.iloc[0:,1:]
  x = df.drop(['length','label'], axis = 1)
  y = df['label']

  KNN = make_pipeline(MinMaxScaler(), 
                      KNeighborsClassifier(n_neighbors=5))
  KNN.fit(x, y)
  pred = KNN.predict(input)
  return pred[0]
