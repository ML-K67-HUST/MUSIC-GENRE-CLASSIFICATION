import pandas as pd
import numpy as np
import os 
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
import pickle 


df = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_30_sec.csv')
df = df.iloc[0:,1:]
X = df.drop(['length','label'], axis = 1)
y = df['label']

df.label = pd.Categorical(df.label)
y = np.array(df.label.cat.codes)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model = LGBMClassifier(boosting_type= 'gbdt', learning_rate=0.2, max_depth= 7, num_leaves=150)
model.fit(X, y)

if os.path.exists('saved_model/ens_model.pkl'):
    os.remove('saved_model/ens_model.pkl')
with open('saved_model/ens_model.pkl','wb') as file:
    pickle.dump(model, file)