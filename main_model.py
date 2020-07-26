# import the specified packages
import os
import numpy as np

# take off and don't show all the warnings for running tf in terminal
# have to import before getting tf/keras modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# print that the model is extracting the data from process_data
print('extracting data...')

# import the following tf/keras modules
from tensorflow.python.keras.callbacks import ModelCheckpoint
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from tensorflow.python.keras.utils.np_utils import to_categorical
from process_data import train_x, train_y
import matplotlib.pyplot as plt

# print that it is converting data
print('converting data...')

# convert the data_x and data_y to numpy arrays so it can be easier for later transformations
data_x = np.array(train_x, dtype="float32")
data_y = np.array(train_y)

# label binarize data_y into 0s and 1s, from 'no mask' and 'mask' labels
lb = LabelBinarizer()
data_y = lb.fit_transform(data_y)
data_y = to_categorical(data_y)

# perform an 80/20 (train/test) split on data_x and data_y to form the training and testing np arrays
train_x = data_x[:8000]
train_y = data_y[:8000]
test_x = data_x[8000:]
test_y = data_y[8000:]

# initialize the learning rate, epochs, and batch size
learning_rate = 1e-5
epochs = 3
batch_size = 5

# create a data augmentation from ImageDataGenerator to improve accuracy by testing with specified params below
# it basically creates modified images to test with in all cases
data_aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
)

# print that the base model is being built
print('building base model...')

# set the base model to the MobileNetV2 model architecture with weights trained by imagenet dataset
# also, cut off the top layer
base_model = MobileNetV2(input_shape=(train_x.shape[1:]), weights='imagenet', include_top=False)

# print that the main model is being built
print('building main model...')

# set main_model equal to base_model.output, such that it gets the output of the base_model
main_model = base_model.output

# add the first layer, average pooling layer, to the main_model with a pool_size of (2, 2)
# this allows the image to be pooled by a 2 by 2 filter so it can extract the main details
main_model = tensorflow.keras.layers.AveragePooling2D(pool_size=(2, 2))(main_model)

# flatten out the layer so it only in 1 dimension and add it to main_model
# this makes it easier for the model to process the inputs and outputs from neurons
main_model = tensorflow.keras.layers.Flatten(name="flatten")(main_model)

# dense layers are fully connected layers with (n * n) params
# create a dense layer with 512 neurons and an activation function of relu and add it to main_model
# this reduces the number of output params, allowing the model to reduce the number of inputs to next layer
main_model = tensorflow.keras.layers.Dense(512, activation="relu")(main_model)

# randomly dropout 50% of the neurons remaining, such that only 256 input neurons remain for next layer
main_model = tensorflow.keras.layers.Dropout(0.5)(main_model)

# process those 256 inputs into 2 final neurons (two classes = no mask and mask)
# a softmax activation is best because it outputs the probability of each class, which is invaluable
main_model = tensorflow.keras.layers.Dense(2, activation="softmax")(main_model)

# create a model based on base_model.input and outputs as main_model
model = Model(inputs=base_model.input, outputs=main_model)

# freeze the base_model.layers in the original MobilNetV2 model
# this makes the params and layers in the model untrainable
for layer in base_model.layers:
    layer.trainable = False

# print that the model is training
print('training model...')

# set loss to binary cross entropy since there are only 2 output classes
# optimizer is Adam and decay is calculated based on the learning rate and epochs so that the model doesn't overshoot
# testing on the metric of accuracy
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=learning_rate, decay=learning_rate / epochs), metrics=["accuracy"])

# this saves the best model with val_loss under the name specified below
# it replaces the model with a model of better accuracy, if it exists
# the directory of face-mask-detection.model will contain the files to the model itself in the best epoch
checkpoint = ModelCheckpoint(
    'face-mask-detection.model',
    monitor='val_loss',
    verbose=2,
    save_best_only=True,
    mode='auto'
)

# fit the model and train it with the following params outlined below
# use the validation_data as test_x and test_y (2000 images)
M = model.fit(
    data_aug.flow(train_x, train_y, batch_size=batch_size),
    validation_data=(test_x, test_y),
    epochs=epochs,
    callbacks=[checkpoint]
)

# print that the training has finished
print('training finished...')

# print that the model is plotting and saving the graph
print('plotting and saving graph...')

# set the number of epochs to num_epochs in order to make a plot
num_epochs = epochs

# create a plot using the 'ggplot' style
plt.style.use("ggplot")
plt.figure()

# using matplotlib make a plot with the following params
# print the lines of loss, val-loss, accuracy, and val-accuracy over the time of num_epochs
# i.e. print the 4 lines described below, with specified labels
plt.plot(np.arange(0, num_epochs), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, num_epochs), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, num_epochs), M.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, num_epochs), M.history["val_accuracy"], label="val_acc")

# plot title is below along with x-axis and y-axis labels
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")

# place the legend for line colors/labels in the lower left corner of the plot
plt.legend(loc="lower left")

# finally save the plot under the name 'model-plot.png' into the current working directory
plt.savefig("model-plot")
