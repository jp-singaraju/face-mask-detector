import os

# don't show any warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from keras_preprocessing.image import img_to_array
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model('working-first-model.model')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = {1: 'with_mask', 0: 'without_mask'}
color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}

while True:
    img = source.read()
    img = np.array(img[1])
    faces = face_clsfr.detectMultiScale(img, 1.3, 5)

    for x, y, w, h in faces:
        face_img = img[y:y + w, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = img_to_array(face_img)
        face_img = preprocess_input(face_img)
        face_img = np.array(face_img, dtype=np.float32)
        face_img = np.expand_dims(face_img, axis=0)
        (mask, noMask) = model.predict(face_img)[0]
        if mask > noMask:
            label = 1
        else:
            label = 0
        cv2.rectangle(img, (x, y), (x + w, y + h), color_dict[label], 2)
        cv2.rectangle(img, (x, y - 40), (x + w, y), color_dict[label], -1)
        cv2.putText(
            img, labels_dict[label],
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()
source.release()
