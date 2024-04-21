import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm
from sklearn.metrics import classification_report

dataset = pd.read_csv("features_3_sec.csv")
df = dataset.sample(frac=1).reset_index(drop=True) # Shuffle the dataset

label_encoder = LabelEncoder()
target = "label"

X = df.drop(['filename','length',target],axis=1) 
y = df[target]
# Encode labels to integers
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM object with a rbf kernel
clf = svm.SVC(kernel='rbf', gamma='scale', C=50)

# Fit the model to the training data (X_train contains the features and y_train contains the genre labels)
clf.fit(X_train, y_train)

# Use the trained model to predict the genre labels of the test data
y_pred = clf.predict(X_test)

print(classification_report(y_test,y_pred))