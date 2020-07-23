import os
import numpy as np
import main

# don't show any warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from keras.callbacks import ModelCheckpoint

data = np.asarray(main.data)
target = np.asarray(main.labels)

print('retrieved lists...')

model = keras.Sequential()

print('building model...')

model.add(keras.layers.Flatten(input_shape=data.shape[1:]))
model.add(keras.layers.Dense(1000, activation='tanh'))
model.add(keras.layers.Dense(500, activation='tanh'))
model.add(keras.layers.Dense(20, activation='tanh'))
model.add(keras.layers.Dense(2, activation='softmax'))

print('compiling model...')

model.compile(optimizer=keras.optimizers.Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

print('splitting data...')

train_data = data[16000:]
train_target = target[16000:]
test_data = data[:16000]
test_target = target[:16000]

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
