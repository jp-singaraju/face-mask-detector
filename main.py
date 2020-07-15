import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

directory = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/larxel/images/'

imageMain = np.array([])
for image in os.listdir(directory):
    img = mpimg.imread(directory + image)
    imageMain = np.append(imageMain, img)

# code overloads here after a while due to no space left in numpy array
# MemoryError

print(len(imageMain))
for imgNum in range(len(imageMain)):
    plt.imshow(imageMain[imgNum])
    plt.show()
