# this file only needs to be run once
import os
import cv2
import time

# this file converts images from 1024 x 1024, rgb, to 256 x 256, grayscale

# Dataset Folder Structure
# Face Mask Detection Data
#   |_ 20k_faces
#           |_ without_mask
#           |_ with_mask
#           |_ new_without_mask
#           |_ new_with_mask

# get the directory for the files
withoutDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/without_mask/'
withDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/'
newWithoutDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_without_mask/'
newWithDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/'

# initialize the time variable
totalTime = 0

# set the folder you want to iterate through, in this case withDir (rgb images with masks)
for image in os.listdir(withDir):
    startTime = time.time()  # start the time
    print(image)  # print the image name
    imageMain = cv2.imread(withDir + image)  # read the image from the directory specified
    scaleFactor = 25  # set the scale factor to 25% of the original image
    # find the new dimensions using the scaleFactor
    dim = int(imageMain.shape[1] * scaleFactor / 100), int(imageMain.shape[0] * scaleFactor / 100)
    imageMain = cv2.resize(imageMain, dim)  # resize the images with specs
    imageMain = cv2.cvtColor(imageMain, cv2.COLOR_BGR2GRAY)  # make the image grayscale
    cv2.imwrite(newWithDir + image, imageMain)  # write the image, or save it into the new folder (newWithDir)
    endTime = time.time()  # end the time
    totalTime += endTime - startTime  # add total time
    print(totalTime)  # print the time after each iteration

# the other folder to iterate through, withoutDir (rgb without masks), same code, different directory
for image in os.listdir(withoutDir):
    startTime = time.time()  # start the time
    print(image)  # print the image name
    imageMain = cv2.imread(withoutDir + image)  # read the image from the directory specified
    scaleFactor = 25  # set the scale factor to 25% of the original image
    # find the new dimensions using the scaleFactor
    dim = int(imageMain.shape[1] * scaleFactor / 100), int(imageMain.shape[0] * scaleFactor / 100)
    imageMain = cv2.resize(imageMain, dim)  # resize the images with specs
    imageMain = cv2.cvtColor(imageMain, cv2.COLOR_BGR2GRAY)  # make the image grayscale
    cv2.imwrite(newWithoutDir + image, imageMain)  # write the image, or save it into the new folder (newWithDir)
    endTime = time.time()  # end the time
    totalTime += endTime - startTime  # add total time
    print(totalTime)  # print the time after each iteration

# code to show image with array = imageMain
# plt.imshow(imageMain)
# plt.show()
