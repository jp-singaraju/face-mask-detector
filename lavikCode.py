import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
import os
import cv2
import time

# Lavik directory -
no_mask_dir = r'D:\Face Mask Detection Dataset\without_mask'
mask_dir = r'D:\Face Mask Detection Dataset\with_mask'

train = np.empty(shape=(16000, 1), dtype=str)
test = np.empty(shape=(4000, 1), dtype=str)

for index in range(8000):
    # adds training data from both labels
    train[2 * index], train[2 * index + 1] = os.path.join(no_mask_dir, os.listdir(no_mask_dir)[index]), os.path.join(mask_dir, os.listdir(mask_dir)[index])

for index in range(8000, 10000):
    # adds testing data from both labels
    test[2 * index], test[2 * index + 1] = os.path.join(no_mask_dir, os.listdir(no_mask_dir)[index]), os.path.join(mask_dir, os.listdir(mask_dir)[index])

print(test[0])
print(train[0])
