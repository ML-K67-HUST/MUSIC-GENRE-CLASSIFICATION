import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import pickle
dataset = pd.read_csv("GTZAN/Data/features_30_sec.csv")

df = dataset.copy()

non_floats = []
for col in df.iloc[:,:-1]:
    if df[col].dtypes != "float64":
        non_floats.append(col)
df = df.drop(columns=non_floats)

L = len(df.columns)
# All the features are put into a matrix
X = df.iloc[:,:L-1].values
# Genre tags are converted to categorical data
df.label = pd.Categorical(df.label)
# Each genre is encoded as a numrical code
y = np.array(df.label.cat.codes)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


# # Create an SVM object with a rbf kernel
clf = svm.SVC(kernel='rbf', gamma='scale', C=50)

# # Fit the model to the training data (X_train contains the features and y_train contains the genre labels)
# clf.fit(X_train, y_train)

# # Use the trained model to predict the genre labels of the test data
# y_pred = clf.predict(X_test)

clf.fit(X,y)

with open('svm_model.pkl','wb') as file:
  pickle.dump(clf, file)


