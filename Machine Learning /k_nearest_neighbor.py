
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

def KNN_predict(input):
  df = pd.read_csv('GTZAN/Data/features_3_sec.csv')
  df = df.iloc[0:,1:]
  x = df.drop(['length','label'], axis = 1)
  y = df['label']

  KNN = make_pipeline(MinMaxScaler(), 
                      KNeighborsClassifier(n_neighbors=1))
  genre_dict = {"blues":0,"classical":1,"country":2,"disco":3,"hiphop":4,"jazz":5,"metal":6,"pop":7,"reggae":8,"rock":9}

  KNN.fit(x, y)
  pred = KNN.predict(input)
  return pred[0]
