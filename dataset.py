import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
import numpy as np
import os

# Pranav directory -
noMaskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/without_mask/'
maskDir = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/20k_faces/with_mask/'

# Lavik directory -
# nothing now

maskSet = noMaskSet = data = np.array([])

counter = 0
for image in os.listdir(noMaskDir):
    counter += 1
    print(counter)
    imageMain = mpimg.imread(noMaskDir + image)
    noMaskSet = np.append(noMaskSet, imageMain)
    # image = load_img(imgPath, color_mode="grayscale", target_size=(256, 256, 1))
    # imageMain = img_to_array(image)

# maskSet = np.array(maskSet, dtype='float32')
# noMaskSet = np.array(noMaskSet, dtype='float32')

# show the image if needed
# for image in maskSet:
#     imageMain = mpimg.imread(image)
#     plt.imshow(imageMain)
#     plt.show()
