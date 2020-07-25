import os

# don't show any warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
import numpy as np
from keras_preprocessing.image import load_img
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array
from main import testX, testY

model = load_model('updated-model001.model')

# img = load_img('C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/with-mask-default-mask-seed0000.png')
# img = img_to_array(img)
# img = preprocess_input(img)
# img = np.array(img, dtype=np.float32)
# print(img.shape)
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# (mask, noMask) = model.predict(img)[0]
# print('mask') if mask > noMask else print('no mask')

testX = np.array(testX[:2000])
testY = np.array(testY[:2000])
pred = model.predict(testX)
pred = np.argmax(pred, axis=1)
print(pred)

counter = 0
for i in range(len(testY)):
    if testY[i] != pred[i]:
        counter += 1

print(counter)
