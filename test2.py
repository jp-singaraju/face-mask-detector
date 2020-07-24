import os

# don't show any warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import data, labels

model = load_model('face_detection_model')

# img = cv2.imread('C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/with-mask-default-mask-seed0000.png')
# img = np.array(img, dtype=np.float32)
# print(img.shape)
# img = tf.expand_dims(img, 0)
# print(img.shape)
# result = model.predict(img)
# predIdxs = np.argmax(result, axis=1)
# print(predIdxs)
#
testX = np.array(data[16000:], dtype=np.float32)
testY = np.array(labels[16000:])

result = model.predict(testX)
predIdxs = np.argmax(result, axis=1)
print(predIdxs[:500])
print('\n')
print(testY[:500])
diff = []

for i in range(len(testY[:500])):
    if testY[i] != predIdxs[i]:
        diff.append(testY[i])

print(len(diff))
