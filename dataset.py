import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

noMaskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/without_mask/'
maskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/with_mask/'

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

# Steps:
# 1. encoding = binary classifier (0 or 1) (DONE)
# 2. splitting the data (IN PROGRESS)
# 3. construct the image generator/data augmentation = training the model + improving the data
    # training the face mask model
# 4. import a base model (MobilNetV2) = fine-tuning

