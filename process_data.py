# import the packages listed below
from os import system
import os
import progress_bar
import time
import random

# take off and don't show all the warnings for running tf in terminal
# have to import before getting tf/keras modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the following tf/keras packages
from tensorflow.python.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

# this code is supposed to append all the array values of the 20k images into the specified data lists
# it has added progress bars to show the progress so far when receiving the values
# this file is mainly for pre-processing all the data to be used in the main_model.py file

# use directories with grayscale images after unzipping and converting, or downloading
# after doing what is said above, uncomment the bottom two lines
# new_without_dir = your new directory where your updated no mask images are (224 x 224 grayscale)
# new_with_dir = your new directory where your mask images are (224 x 224 grayscale)

# declare an empty list for no-mask and mask
mask_set = []
no_mask_set = []

# time, counter, i = 0
total_time = 0.0
counter = 0
i = 0

print('Program Started... ')  # print that the program started
time.sleep(1)  # wait for 1 second
system('cls')  # clear the screen/console on call
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progress_bar.bar_method(0, 1000, prefix='Loading Faces... ', suffix='Complete', length=50, time=0)

# loop in order to append all new mask image values
for image in os.listdir(new_with_dir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progress_bar.bar_method(i + 1, 1000, prefix='Loading Faces... ', suffix='Complete', length=50,
                                time=float(total_time))
        i += 1
    image_main = load_img(new_with_dir + image)  # load the specified image from the directory
    image_main = img_to_array(image_main)  # convert the image to a numpy array
    image_main = preprocess_input(image_main)  # pre-process the input based on the MobileNetV2 model
    mask_set.append((image_main, 'mask'))  # append the list of image value and label to the data list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    total_time = float(end - start)  # total_time now equals the end value minus beginning value

# time, counter, i = 0
total_time = 0.0
counter = 0
i = 0
start = time.time()  # start the timer

# bar method with reading the image for the 10k images with a mask
progress_bar.bar_method(0, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50, time=0)

# loop in order to append all new without mask image values
for image in os.listdir(new_without_dir):
    # every 10 increments, update the bar
    if counter % 10 == 0:
        progress_bar.bar_method(i + 1, 1000, prefix='Optimizing Images... ', suffix='Complete', length=50,
                                time=float(total_time))
        i += 1
    image_main = load_img(new_without_dir + image)  # load the specified image from the directory
    image_main = img_to_array(image_main)  # convert the image to a numpy array
    image_main = preprocess_input(image_main)  # pre-process the input based on the MobileNetV2 model
    no_mask_set.append((image_main, 'no mask'))  # append the list of image value and label to the data list
    counter += 1  # increment counter by 1
    end = time.time()  # end time
    total_time = float(end - start)  # total_time now equals the end value minus beginning value

# no_mask_set = 10k images & mask_set = 10k images
# both sets have same people, and we don't want the people to overlap as it would mess the model up
# get the first 5000 from no_mask_set and last 5000 from mask_set and set it equal to train
# do the vice versa for test
train = no_mask_set[:5000] + mask_set[5000:]
test = no_mask_set[5000:] + mask_set[:5000]

# shuffle the lists in place, so that you don't have the same labels of images in a row
random.shuffle(train)
random.shuffle(test)

# declare the train and test lists
# train lists will be used to train the model and test lists could be used to test the model later
# we really don't use the test lists
train_x = []
train_y = []
test_x = []
test_y = []

# split the images list into data and labels
# the train and test lists are organized like this -> ([image array values], label) * 10000
for i in train:
    train_x.append(i[0])
for i in train:
    train_y.append(i[1])
for i in test:
    test_x.append(i[0])
for i in test:
    test_y.append(i[1])
