import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

data = pd.read_csv("features_3_sec.csv")
df = data.sample(frac=1).reset_index(drop=True)

label_encoder = LabelEncoder()
target = "label"
x = df.drop(['filename', 'length',target], axis=1)
y = df[target]
y= label_encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=100)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LGBMClassifier(boosting_type= 'gbdt', learning_rate=0.2, max_depth= 7, num_leaves=31)
model.fit(x_train,y_train)

predictions = model.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy*100))
print(classification_report(y_test, predictions))