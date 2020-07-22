import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from main import finalSet

data = np.array([])
target = np.array([])

for i in finalSet:
    np.append(data, i[0])

for i in finalSet:
    np.append(target, i[1])

model = Sequential()

print('step -1')

model.add(Conv2D(100, kernel_size=(3, 3), input_shape=data.shape[1:]))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(100, kernel_size=(3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

print('step 0')

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('step 1')

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.1)

print('step 2')

checkpoint = ModelCheckpoint(
    'model-{epoch:03d}.model',
    monitor='val_loss',
    verbose=0,
    save_best_only=True,
    mode='auto')

print('step 4')

history = model.fit(
    train_data,
    train_target,
    epochs=20,
    callbacks=[checkpoint],
    validation_split=0.2)

print('step 5')

print(model.evaluate(test_data, test_target))
