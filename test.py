from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('model-020.model')
face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

labels_dict = {1: 'with_mask', 0: 'without_mask'}
color_dict = {1: (0, 255, 0), 0: (0, 0, 255)}

while True:
    ret, img = source.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)
    faces = face_clsfr.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        face_img = gray[y:y + w, x:x + w]
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
