# this file only needs to be run once
import os
import cv2
import time

# this file converts images from 1024 x 1024, rgb, to 224 x 224, grayscale and saves them into files
# in this case, this is already done, so no need to be executed because the images are already resized and filtered
# just unzip then from the zipped file and place them into two different files
# however, if you do want to run this yourself download the link for the original dataset mentioned in GitHub

# Dataset Folder Structure
# Face Mask Detection Data
#     |_ without_mask
#     |_ with_mask
#     |_ new_without_mask
#     |_ new_with_mask

# get the directory for the files, uncomment these lines below

# without_dir = your directory of original no mask dataset (1024 x 1024 rgb)
# with_dir = your directory of original mask dataset (1024 x 1024 rgb)
# new_without_dir = your new directory where updated no mask images will be saved (224 x 224 grayscale)
# new_with_dir = your new directory where updated mask images will be saved (224 x 224 grayscale)

# initialize the time variable
total_time = 0

# set the folder you want to iterate through, in this case with_dir (rgb images with masks)
for image in os.listdir(with_dir):
    start_time = time.time()  # start the time
    print(image)  # print the image name
    image_main = cv2.imread(with_dir + image)  # read the image from the directory specified
    dim = (224, 224)  # set dimensions
    image_main = cv2.resize(image_main, dim)  # resize the images with dimensions
    image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2GRAY)  # make the image grayscale
    cv2.imwrite(new_with_dir + image, image_main)  # save the image into the new folder (new_with_dir)
    end_time = time.time()  # end the time
    total_time += end_time - start_time  # add total time
    print(total_time)  # print the time after each iteration

# the other folder to iterate through, without_dir (rgb without masks), same process, different directory
for image in os.listdir(without_dir):
    start_time = time.time()  # start the time
    print(image)  # print the image name
    image_main = cv2.imread(without_dir + image)  # read the image from the directory specified
    dim = (224, 224)  # set dimensions
    image_main = cv2.resize(image_main, dim)  # resize the images with dimensions
    image_main = cv2.cvtColor(image_main, cv2.COLOR_BGR2GRAY)  # make the image grayscale
    cv2.imwrite(new_without_dir + image, image_main)  # save the image into the new folder (new_without_dir)
    end_time = time.time()  # end the time
    total_time += end_time - start_time  # add total time
    print(total_time)  # print the time after each iteration

# code to show image with pixel values = image_main
# plt.imshow(image_main)
# plt.show()
