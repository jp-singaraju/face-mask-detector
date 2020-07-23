import os
import numpy as np

# don't show any warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from main import data, labels

print('extracted data...')

trainX = np.array(data[:16000])
trainY = np.array(labels[:16000])
testX = np.array(data[16000:])
testY = np.array(labels[16000:])

print('split data...')

baseModel = MobileNetV2(input_shape=(trainX.shape[1:]), weights='imagenet', include_top=False)

print('built base model...')

headModel = baseModel.output
headModel = tensorflow.keras.layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = tensorflow.keras.layers.Flatten(name="flatten")(headModel)
headModel = tensorflow.keras.layers.Dense(128, activation="relu")(headModel)
headModel = tensorflow.keras.layers.Dropout(0.5)(headModel)
headModel = tensorflow.keras.layers.Dense(2, activation="softmax")(headModel)

print('built head model...')

model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False

print('freeze base model...')

INIT_LR = 1e-4
EPOCHS = 30
BS = 20

print('training model...')

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), metrics=["accuracy"])

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")

checkpoint = ModelCheckpoint(
    'model-{epoch:03d}.model',
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='auto')

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
    callbacks=[checkpoint])

print('testing model...')

print(model.evaluate(trainX, trainY))
