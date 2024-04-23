import pandas as pd
import os 
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
import pickle 


data = pd.read_csv("GTZAN/Data/features_3_sec.csv")
df = data.sample(frac=1).reset_index(drop=True)

x = df.drop(['filename', 'length',"label"], axis=1)
y = df["label"]

model = make_pipeline(StandardScaler(),
                        LGBMClassifier(boosting_type= 'gbdt', learning_rate=0.2, max_depth= 7, num_leaves=150))
model.fit(x, y)

with open('ens_model.pkl','wb') as file:
    pickle.dump(model,file)