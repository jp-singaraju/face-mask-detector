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
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from process_data import dataX, dataY
import matplotlib.pyplot as plt

print('extracted data...')

dataX = np.array(dataX, dtype="float32")
dataY = np.array(dataY)

lb = LabelBinarizer()
dataY = lb.fit_transform(dataY)
dataY = to_categorical(dataY)

trainX = dataX[:8000]
trainY = dataY[:8000]
testX = dataX[8000:]
testY = dataY[8000:]

INIT_LR = 1e-5
EPOCHS = 3
BS = 5

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

print('converted data...')

baseModel = MobileNetV2(input_shape=(trainX.shape[1:]), weights='imagenet', include_top=False)

print('built base model...')

headModel = baseModel.output
headModel = tensorflow.keras.layers.AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = tensorflow.keras.layers.Flatten(name="flatten")(headModel)
headModel = tensorflow.keras.layers.Dense(512, activation="relu")(headModel)
headModel = tensorflow.keras.layers.Dropout(0.5)(headModel)
headModel = tensorflow.keras.layers.Dense(2, activation="softmax")(headModel)

print('built head model...')

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print('training model...')

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS), metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    'face-mask-detection-{epoch:03d}.model',
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='auto'
)

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    epochs=EPOCHS,
    callbacks=[checkpoint]
)

print('made model')

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("model-plot")
