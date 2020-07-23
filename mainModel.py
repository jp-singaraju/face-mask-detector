import os
import numpy as np
from sklearn.model_selection import train_test_split
import main

# don't show any warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.callbacks import ModelCheckpoint

data = np.array(main.data)
target = np.array(main.labels)

print('retrieved lists...')

model = keras.Sequential()

print('building model...')

model.add(keras.layers.Flatten(input_shape=data.shape[1:]))
model.add(keras.layers.Dense(1000, activation='tanh'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(500, activation='tanh'))
model.add(keras.layers.Dense(20, activation='tanh'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

print('compiling model...')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('splitting data...')

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.2)

print('saving model...')

checkpoint = ModelCheckpoint(
    'model-{epoch:03d}.model',
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='auto')

print('training model...')

history = model.fit(
    train_data,
    train_target,
    batch_size=3200,
    epochs=20,
    callbacks=[checkpoint],
    validation_split=0.3)

print('update model...')

print(model.evaluate(test_data, test_target))
