import pandas as pd
import os
from preprocess import preprocess_ 
import tensorflow as tf

df = pd.read_csv("../dataset/Dataset.csv")

X, y = preprocess_(df)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=[57,]),
    tf.keras.layers.Dense(300, activation='relu'),                                                                      
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=adam,
             loss="sparse_categorical_crossentropy",
             metrics=["accuracy"])

history = model.fit(X, y, 
                    epochs=100, 
                    batch_size=64, 
                    verbose=1)

if os.path.exists('saved_model/nn_model.keras'):
    os.remove('saved_model/nn_model.keras')

model.save('saved_model/nn_model.keras')