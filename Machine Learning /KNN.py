from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 


def Knn_predict(upload):
    model = KNeighborsClassifier()

    data = pd.read_csv('/home/khangpt/MUSIC-GEN-PROJ/GTZAN/Data/features_3_sec.csv')
    data = data.iloc[0:,1:]
    y = data['label'] # genre variable.
    X = data.drop(columns=['length','label'])
    model.fit(X,y)

    pred = model.predict(upload)

    return pred
    pass