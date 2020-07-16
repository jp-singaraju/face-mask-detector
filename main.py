import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from bs4 import BeautifulSoup
import lxml

imgDirectory = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/larxel/images/'
labelDirectory = 'C:/Users/Singaraju/Desktop/Face Mask Detection Data/larxel/annotations/'

imageNames = np.array([])
labels = np.array([])

for label in os.listdir(labelDirectory):
    with open(labelDirectory + label, 'r') as f:
        lbl = f.read()
    data = BeautifulSoup(lbl, 'xml')
    lbl = data.find_all('name')
    print(lbl)

for image in os.listdir(imgDirectory):
    imageNames = np.append(imageNames, (imgDirectory + image))

for image in imageNames:
    imageMain = mpimg.imread(image)
    # shows the image if needed
    plt.imshow(imageMain)
    plt.show()


# code overloads here after a while due to no space left in numpy array
# MemoryError

# print(len(imageMain))
# for imgNum in range(len(imageMain)):
#     plt.imshow(imageMain[imgNum])
#     plt.show()
