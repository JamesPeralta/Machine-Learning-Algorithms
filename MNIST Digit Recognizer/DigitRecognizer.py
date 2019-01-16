# Imports for array-handling and plotting
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Keras imports
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Retrieve the mnist data set
# Images = Pictures of handwritten digits
# Labels = Label of Pictures
(Images_train, Labels_train), (Images_test, Labels_test) = mnist.load_data()

# Convert 28x28 into a float32 input vector
Images_train = Images_train.reshape(60000, 784)
Images_test = Images_test.reshape(10000, 784)
Images_train = Images_train.astype('float32')
Images_test = Images_test.astype('float32')

# Normalize the data to help with the training
Images_train = Images_train / 255
Images_test = Images_test / 255

# Encode the training and test labels into one-hot encoding
n_classes = 10
Labels_train = np_utils.to_categorical(Labels_train, n_classes)
Labels_test = np_utils.to_categorical(Labels_test, n_classes)

# Building the network
# First Layer
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Second Layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# Compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train and save the model

