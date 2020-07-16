import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Pranav directory -
noMaskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/without_mask/'
maskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/with_mask/'

# Lavik directory -
# nothing now
# nothing now

maskSet = noMaskSet = np.array([])

for image in os.listdir(noMaskDir):
    noMaskSet = np.append(noMaskSet, noMaskDir + image)
for image in os.listdir(maskDir):
    maskSet = np.append(maskSet, maskDir + image)

# show the image if needed
for image in maskSet:
    imageMain = mpimg.imread(image)
    plt.imshow(imageMain)
    plt.show()
