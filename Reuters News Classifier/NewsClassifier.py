# This model is a single label, multi-class classifier
# that separates news wires by their topics
# By: James Peralta

from keras.datasets import reuters
import numpy as np
from keras import models, layers

# Retrieve all data and their labels using Keras
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# Vectorize your labels
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros(len(sequences), dimensions)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Vectorize your sequences into a one hot encoding
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


def one_hot_encode(labels, dimensions=46):
    results = np.zeros(len(labels), dimensions)
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


# One hot encode all of your labels
y_train = one_hot_encode(train_labels)
y_test = one_hot_encode(test_labels)

# Define the network
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

# Compile the network by choosing the loss, optimizer and metrics
model.compile()

# Fit the network

# Infer using the network



""" Decode the paragraphs back into a String
word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) decoded_newswire = 
    ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
"""
