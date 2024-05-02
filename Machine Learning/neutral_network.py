import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pickle

df = pd.read_csv("GTZAN/Data/features_30_sec.csv")

x = df.drop(['label'], axis=1)
y = df['label']

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(x_scaled.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_scaled, y, epochs=10, batch_size=32, verbose=2)

with open('nn_model.pkl', 'wb') as file:
    pickle.dump(model, file)