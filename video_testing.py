# import the specified packages
import numpy as np
import os
import cv2
import progress_bar
import time

# take off and don't show all the warnings for running tf in terminal
# have to import before getting tf/keras modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the following tf/keras modules
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from keras_preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# initialize total_time, i, and start the time
total_time = 0.0
i = 0
start = time.time()

# implement the bar method so to see if the model has loaded
# this really serves no use, except to show the progress bar in use
progress_bar.bar_method(0, 50000, prefix='loading model... ', suffix='Complete', length=50, time=0)

# run for the length of bar, which is 50000, so it is a static number for implementation purposes
for i in range(50000):
    # update the progress bar
    progress_bar.bar_method(i + 1, 50000, prefix='loading model... ', suffix='Complete', length=50,
                            time=float(total_time))
    i += 1  # increment i to move to the next bar iteration
    end = time.time()  # end time
    total_time = float(end - start)  # get total time for the progress bar

# print that the model is being optimized
print('optimizing model...')

# load the model from the file mentioned
mask_model = load_model('mask-detector.model')

# print that the video stream is starting
print('starting video stream...')

# start the video stream
# cv2.CAP_DSHOW is the argument that doesn't print an warnings when 'q' pressed, if any
source = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# declare a detection_net using the prototxt model architecture and the weights from the caffe model
detection_net = cv2.dnn.readNet('face-detection-models/deploy.prototxt.txt',
                                'face-detection-models/res10_300x300_ssd_iter_140000.caffemodel')


# declare the function of predict_mask with image (frame from video stream) and model
def predict_mask(image, model):
    # set the h and w variables to the images height and width -> i.e. (224, 224)
    (h, w) = image.shape[:2]

    # construct a blob from the image using the cv2 pre-trained deep neural network
    # blob is used to find the ROIs in the image
    blob = cv2.dnn.blobFromImage(image, size=(224, 224))

    # pass the constructed blob into the face_net pre-trained model
    # this model is used to recognize the faces within an image
    detection_net.setInput(blob)

    # get the predicted faces as an output from the model
    preds = detection_net.forward()

    # for all the faces that it sees in the image
    for i in range(preds.shape[2]):
        # if the confidence of the face it sees is above .6, continue
        # in other words, if it thinks that the chance that it is a face is .6 or more, then proceed

        if preds[0, 0, i, 2] > .6:
            # get the specified pixel locations for each face and multiply it by h and w from before
            # this will output the starting locations of the box
            # this is the ROI (region of interest) of the image, or the face we want to look for
            box = preds[0, 0, i, 3:7] * np.array([w, h, w, h])

            # convert the box into int, currently its in float
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # set the starting x and y, and ending x and y coordinates for the box
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(w - 1, end_x), min(h - 1, end_y))

            # get those specified pixels from the first numpy array of the image and set it equal to face
            face = image[start_y:end_y, start_x:end_x]

            # change color of the image, resize it, and pre-process it in order to pass it into the model
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # expand the dimensions of the image -> i.e. (1, 224, 224, 3)
            face = np.expand_dims(face, axis=0)

            # predict the mask_prob and no_mask_prob of the image by passing it into the model
            mask_prob, no_mask_prob = model.predict(face)[0]

            # if the mask_prob is greater than the no_mask_prob then say that there is a mask
            # else vice versa
            # set color to green (good)
            # color is in BGR and not RGB, so (B, G, R)
            if mask_prob > no_mask_prob:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            # the resulting label to be displayed on top of the bounding box
            result = f'{label}: {round((max(mask_prob, no_mask_prob) * 100), 2)}%'

            # place the text on top of the box at specified coordinates and construct a rectangle
            # this rectangle shows the ROI
            cv2.putText(image, result, (start_x, start_y - 10), cv2.FONT_HERSHEY_DUPLEX, .7, color, 2)
            cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color, 6)


# prints out the following statement
print('\nPress "q" to quit.')

# run forever until break
while True:
    # read the frame from the stream and get the array values of the image
    frame = source.read()
    frame = np.array(frame[1])

    # pass the array values and model into the predict_mask function to output the resulting label and box
    predict_mask(frame, mask_model)
    cv2.imshow('Face Mask Detection', frame)

    # if the user outputs a letter from the keyboard, set it equal to user
    user = cv2.waitKey(1)

    # if user == 'q', break out of the loop and stop
    if user == ord('q'):
        break

# destroy all the windows of the stream and stop the stream
cv2.destroyAllWindows()
source.release()
