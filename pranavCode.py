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

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
args = vars(ap.parse_args())

prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt.txt"])
weightsPath = os.path.sep.join([args["face"],
                                "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)


def myImage(img):
    blob = cv2.dnn.blobFromImage(img, 1.0, (256, 256), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    print("Computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    print(detections.shape)
    detections = detections[0]
    plt.imshow(detections)
    plt.show()
    # cv2.imwrite('C:/Users/Singaraju/Documents/GitHub/Fask-Mask-Detection/images/', detections)


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

frame = vs.read()

key = 'a'

while key != 'q':
    frame = vs.read()
    # show the output frame
    myImage(frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    key = 'q'

cv2.destroyAllWindows()
vs.stop()
