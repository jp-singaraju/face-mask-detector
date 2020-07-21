# import the packages listed below
from os import system
import os
import cv2
import progressBar
import time
import random

# this code is supposed to append all the array values of the 20k images
# it has added progress bars to show the progress so far

# directories with 256 x 256 grayscale images
# Pranav Directories
newWithoutDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_without_mask/'
newWithDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/'

# Lavik Directories
# newWithoutDir = 'D:/Face Mask Detection Dataset/new_without_mask/'
# newWithDir = 'D:/Face Mask Detection Dataset/new_with_mask/'

# declare a mask set and no mask set array to append image values to
noMaskSet = []
maskSet = []

# time, counter, i = 0
totalTime = 0.0
counter = 0
i = 0

print('Program Started... ')  # print that the program started
time.sleep(1)  # wait for 1 second
system('cls')  # clear the screen/console on call
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progressBar.barMethod1(0, 1000, prefix='Loading Faces... ', suffix='Complete', length=50, time=0)

# loop in order to append all new mask image values
for image in os.listdir(newWithDir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progressBar.barMethod1(i + 1, 1000, prefix='Loading Faces... ', suffix='Complete', length=50,
                               time=float(totalTime))
        i += 1
    imageMain = cv2.imread(newWithDir + image)  # read the image from the directory
    maskSet.append((imageMain, 'mask'))  # append the array image value and label to the maskSet list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    totalTime = float(end - start)  # totalTime now equals the end value minus beginning value

# time, counter, i = 0
totalTime = 0.0
counter = 0
i = 0
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progressBar.barMethod1(0, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50, time=0)

# loop in order to append all new without mask image values
for image in os.listdir(newWithoutDir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progressBar.barMethod1(i + 1, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50,
                               time=float(totalTime))
        i += 1
    imageMain = cv2.imread(newWithoutDir + image)  # read the image from the directory
    noMaskSet.append((imageMain, 'no mask'))  # append the array image value and label to the maskSet list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    totalTime = float(end - start)  # totalTime now equals the end value minus beginning value

# say that the images got imported
print('Images Imported')

# creates a final dataset with all shuffled labels ('mask' & 'no mask') and 20k image arrays
# finalSet = ([image array values], label) x 20000
finalSet = random.sample((noMaskSet + maskSet), 20000)
