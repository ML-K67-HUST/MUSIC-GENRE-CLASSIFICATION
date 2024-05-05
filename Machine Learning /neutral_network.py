import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

dataset = pd.read_csv("GTZAN/Data/features_30_sec.csv")

df = dataset.copy()

non_floats = []
for col in df.iloc[:,:-1]:
    if df[col].dtypes != "float64":
        non_floats.append(col)
df = df.drop(columns=non_floats)

L = len(df.columns)
X = df.iloc[:,:L-1].values
df.label = pd.Categorical(df.label)
y = np.array(df.label.cat.codes)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_scaled, y, epochs=10, batch_size=32, verbose=2)

model.save("nn_model.h5")