import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras_preprocessing.image import img_to_array
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

maskModel = load_model('mask-detection.model')

source = cv2.VideoCapture(0)


def predict_mask(image, model):
    net = cv2.dnn.readNet('face_detection_models/deploy.prototxt.txt', 'face_detection_models/res10_300x300_ssd_iter_140000.caffemodel')

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, size=(224, 224))

    net.setInput(blob)
    preds = net.forward()

    for i in range(len(preds)):
        if preds[0, 0, i, 2] > .6:
            box = preds[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            maskProb, noMaskProb = model.predict(face)[0]
            if maskProb > noMaskProb:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
            result = f'{label}: {round((max(maskProb, noMaskProb) * 100), 2)}%'
            cv2.putText(image, result, (startX, startY - 10), cv2.FONT_HERSHEY_DUPLEX, .7, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 6)


print('Press "q" to quit.')

while True:
    frame = source.read()
    frame = np.array(frame[1])
    predict_mask(frame, maskModel)
    cv2.imshow('Face Mask Detection', frame)
    key = cv2.waitKey(1)
    if key == 'q':  # fix the q command
        break

cv2.destroyAllWindows()
source.release()
