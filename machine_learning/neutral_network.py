import pandas as pd
import os
from preprocess import preprocess_ 
import tensorflow as tf

df = pd.read_csv("GTZAN/Data/features_30_sec.csv")

X, y = preprocess_(df)


model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)))
# model.add(tf.keras.layers.Dense(32, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.fit(X, y, epochs=10, batch_size=32, verbose=2)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(300, activation='relu'),                                                                      
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=245, batch_size=32, verbose=1)

if os.path.exists('saved_model/nn_model.keras'):
    os.remove('saved_model/nn_model.keras')

model.save('saved_model/nn_model.keras')