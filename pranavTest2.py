import cv2
from imutils.video import VideoStream
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

filename = r'C:\Users\Singaraju\Desktop\Face Mask Detection Data\larxel\images\maksssksksss0.png'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')
upper_body = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# User message
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30, 30)
weared_mask_font_color = (255, 255, 255)
not_weared_mask_font_color = (0, 0, 255)
thickness = 2
font_scale = 1
weared_mask = "Thank You for wearing MASK"
not_weared_mask = "Please wear MASK to defeat Corona"

# Read video
vs = VideoStream(src=0).start()
time.sleep(2.0)

while 1:
    # Get individual frame
    img = vs.read()
    img = cv2.flip(img, 1)

    # Convert Image into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert image in black and white
    (thresh, black_and_white) = cv2.threshold(gray, bw_threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('black_and_white', black_and_white)

    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Face prediction for black and white
    faces_bw = face_cascade.detectMultiScale(black_and_white, 1.1, 4)

    if len(faces) == 0 and len(faces_bw) == 0:
        cv2.putText(img, "No face found...", org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    elif len(faces) == 0 and len(faces_bw) == 1:
        # It has been observed that for white mask covering mouth, with gray image face prediction is not happening
        cv2.putText(img, weared_mask, org, font, font_scale, weared_mask_font_color, thickness, cv2.LINE_AA)
    else:
        # Draw rectangle on gace
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        # Detect lips counters
        mouth_rects = mouth_cascade.detectMultiScale(img, 1.5, 5)

    # Show frame with results
    cv2.imshow('Mask Detection', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

vs.stop()
cv2.destroyAllWindows()

# # draw an image with detected objects
# def draw_image_with_boxes(filename, result_list):
#     # load the image
#     data = cv2.imread(filename)
#     # plot the image
#     plt.imshow(data)
#     # get the context for drawing boxes
#     ax = plt.gca()
#     # plot each box
#     for result in result_list:
#         # get coordinates
#         x, y, width, height = result['box']
#         # create the shape
#         rect = Rectangle((x, y), width, height, fill=False, color='red')
#         # draw the box
#         ax.add_patch(rect)
#     # show the plot
#     plt.show()
#
#
# # load image from file
# pixels = cv2.imread(filename)
# # create the detector, using default weights
# detector = MTCNN()
# # detect faces in the image
# faces = detector.detect_faces(pixels)
# # display faces on the original image
# draw_image_with_boxes(filename, faces)

classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# pixels = cv2.imread(r'C:\Users\Singaraju\Desktop\Face Mask Detection Data\larxel\images\maksssksksss0.png')
# imgBox = classifier.detectMultiScale(pixels, 1.06, 6)
# for box in imgBox:
#     x, y, width, height = box
#     x2, y2 = x + width, y + height
#     cv2.rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)
# # show the image
# cv2.imshow('face detection', pixels)
# # keep the window open until we press a key
# cv2.waitKey(0)
# # close the window
# cv2.destroyAllWindows()
