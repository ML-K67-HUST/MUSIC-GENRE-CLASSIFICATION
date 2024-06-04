import pandas as pd
import numpy as np
from preprocess import preprocess_
from sklearn import svm
import pickle
import os

df = pd.read_csv('../dataset/Dataset.csv')
X, y = preprocess_(df)


clf = svm.SVC(kernel='rbf', gamma='scale', C=200,probability=True)

clf.fit(X,y)

if os.path.exists('saved_model/svm_model.pkl'):
    os.remove('saved_model/svm_model.pkl')
with open('saved_model/svm_model.pkl','wb') as file:
    pickle.dump(clf, file)

