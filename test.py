import os

# don't show any warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = load_model('updated-model001.model')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = {1: 'with_mask', 0: 'without_mask'}
color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}

while True:
    img = source.read()
    img = np.array(img[1])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_clsfr.detectMultiScale(img, 1.3, 5)

    for x, y, w, h in faces:
        face_img = img[y:y + w, x:x + w]
        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.array(face_img, dtype=np.float32)
        face_img = tf.expand_dims(face_img, 0)
        result = model.predict(face_img)
        label = np.argmax(result, axis=1)[0]
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
