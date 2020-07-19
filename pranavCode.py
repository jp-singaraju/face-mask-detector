import os
import cv2
import progressBar

newWithoutDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_without_mask/'
newWithDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/new_with_mask/'

noMaskSet = []
maskSet = []
counter = 0
i = 0

# bar method with reading the image for the 10k images with a mask
progressBar.barMethod1(0, 1000, prefix='Loading Faces... ', suffix='Complete', length=50)
for image in os.listdir(newWithDir):
    if counter % 10 == 0:
        progressBar.barMethod1(i + 1, 1000, prefix='Loading Faces... ', suffix='Complete', length=50)
        i += 1
    imageMain = cv2.imread(newWithDir + image)
    maskSet.append(imageMain)
    counter += 1

# second bar code didn't do yet, still need to finish that
for image in os.listdir(newWithoutDir):
    imageMain = cv2.imread(newWithoutDir + image)
    noMaskSet.append(imageMain)
