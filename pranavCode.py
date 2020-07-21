from imutils.video import VideoStream
import time
import argparse
import imutils
import time
import cv2
import os
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
# ap.add_argument("-m", "--model", type=str,
#                 default="mask_detector.model",
#                 help="path to trained face mask detector model")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], 'deploy.prototxt.txt'])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")


# model = load_model(args["model"])


def myImage():
    img = cv2.imread(r'C:\Users\Singaraju\Desktop\Face Mask Detection Data\20k_faces\without_mask\seed0000.png')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imshow(img)
    plt.show()
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img)
    # pass the blob through the network and obtain the face detections
    print("Computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        print(confidence)
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = img[startY:endY, startX:endX]
            # face = cv2.resize(face, (256, 256))
            plt.imshow(face)
            plt.show()


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

key = 'q'

# while key == 'q':
#     frame = vs.read()
#     # show the output frame
#     frame = imutils.resize(frame, width=400)
#
#     key = cv2.waitKey(1) & 0xFF
#     key = 'a'

myImage()

cv2.destroyAllWindows()
vs.stop()
