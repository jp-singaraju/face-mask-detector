import os

# don't show any warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
from main import data, labels

model = load_model('model-002.model')

testX = np.array(data[16000:])
testY = np.array(labels[16000:])

result = model.predict(testX)
predIdxs = np.argmax(result, axis=1)
print(predIdxs[:100])
print('\n')
print(testY[:100])
diff = []

for i in range(len(testY[:100])):
    if testY[i] != predIdxs[i]:
        diff.append(testY[i])

print(len(diff))
