# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.applications import MobileNetV2

from main import finalSet

base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

INIT_LR = 1e-6
EPOCHS = 30
BATCH_SIZE = 32

