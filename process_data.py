# import the packages listed below
from os import system
import os
import progress_bar
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

# this code is supposed to append all the array values of the 20k images
# it has added progress bars to show the progress so far
# this file is also for the model to be added below the pre-processing
# pre-processing couldn't be in another file because creating binary .npy files made it to large to push to github

# directories with 256 x 256 grayscale images
# Pranav Directories
newWithoutDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_without_mask/'
newWithDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/'

# Lavik Directories
# newWithoutDir = 'D:/Face Mask Detection Dataset/new_without_mask/'
# newWithDir = 'D:/Face Mask Detection Dataset/new_with_mask/'

# declare an empty list for no-mask and mask
maskSet = []
noMaskSet = []

# time, counter, i = 0
totalTime = 0.0
counter = 0
i = 0

print('Program Started... ')  # print that the program started
time.sleep(1)  # wait for 1 second
system('cls')  # clear the screen/console on call
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progress_bar.barMethod1(0, 1000, prefix='Loading Faces... ', suffix='Complete', length=50, time=0)

# loop in order to append all new mask image values
for image in os.listdir(newWithDir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progress_bar.barMethod1(i + 1, 1000, prefix='Loading Faces... ', suffix='Complete', length=50,
                                time=float(totalTime))
        i += 1
    imageMain = load_img(newWithDir + image)
    imageMain = img_to_array(imageMain)
    imageMain = preprocess_input(imageMain)
    maskSet.append((imageMain, 'mask'))  # append the list of image value and label to the data list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    totalTime = float(end - start)  # totalTime now equals the end value minus beginning value

# time, counter, i = 0
totalTime = 0.0
counter = 0
i = 0
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progress_bar.barMethod1(0, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50, time=0)

# loop in order to append all new without mask image values
for image in os.listdir(newWithoutDir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progress_bar.barMethod1(i + 1, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50,
                                time=float(totalTime))
        i += 1
    imageMain = load_img(newWithoutDir + image)
    imageMain = img_to_array(imageMain)
    imageMain = preprocess_input(imageMain)
    noMaskSet.append((imageMain, 'no mask'))  # append the list of image value and label to the data list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    totalTime = float(end - start)  # totalTime now equals the end value minus beginning value

data = noMaskSet[:5000] + maskSet[5000:]
test = noMaskSet[5000:] + maskSet[:5000]

# shuffle the list in place
random.shuffle(data)
random.shuffle(test)

# declare the data and target lists
dataX = []
dataY = []
testX = []
testY = []

# split the images list into data and labels
for i in data:
    dataX.append(i[0])
for i in data:
    dataY.append(i[1])
for i in test:
    testX.append(i[0])
for i in test:
    testY.append(i[1])
