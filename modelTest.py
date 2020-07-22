from tensorflow.keras.layers import Dense, AveragePooling2D, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from main import finalSet

trainSet = finalSet[:16000]
testSet = finalSet[16000:]

trainX = []
trainY = []
testX = []
testY = []

for tup in trainSet:
    trainX.append(tup[0])
    trainY.append(tup[1])
for tup2 in testSet:
    testX.append(tup2[0])
    testY.append(tup2[1])

INIT_LR = 1e-4
EPOCHS = 30
BS = 32

baseModel = MobileNetV2(input_shape=(256, 256, 3), weights='imagenet', include_top=False)

# headModel = baseModel.output
# headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
# headModel = Flatten(name="flatten")(headModel)
# headModel = Dense(128, activation="relu")(headModel)
# headModel = Dropout(0.5)(headModel)
# headModel = Dense(2, activation="softmax")(headModel)
#
# model = Model(inputs=baseModel.input, outputs=headModel)
# for layer in baseModel.layers:
#     layer.trainable = False
#
# print("[INFO] compiling model...")
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#
#
# print("[INFO] training head...")
# H = model.fit(
#     aug.flow(trainX, trainY, batch_size=BS),
#     steps_per_epoch=len(trainX) // BS,
#     validation_data=(testX, testY),
#     validation_steps=len(testX) // BS,
#     epochs=EPOCHS)
