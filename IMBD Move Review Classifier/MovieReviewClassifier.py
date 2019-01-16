# This machine learning model is a binary classifier
# By: James Peralta

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
import matplotlib.pyplot as plt

# Upload data
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# This method vectorizes your sequences
def vectorize_sequences(sequences, dimension=10000): # Rows = number of reviews
    results = np.zeros((len(sequences), dimension))  # Columns = One-hot encoded classes
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Vectorize my data
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Vectorize my labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

# Define the models architecture
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Set aside a validation set
x_val = x_train[:10000] # First 10,000 of the training set
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

# Train my model
history = model.fit(x_train, y_train, epochs=4, batch_size=512) # After fitting it will store results in a History object

# Evaluate the model
results = model.evaluate(x_test, y_test)

# Use model to predict
print(model.predict(x_test))

"""  
Description:
How to plot the results when training your model
-------------------------------
# View the training History
history_dict = history.history # Retrieve the dictionary of metrics

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')

# Configure the graphs labels
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show() # Show the graph
"""

""" 
Description: 
The paragraphs are stored in an array of word indices where each indice represents a word from a dictionary of 20,000 words
but we specifically set the max amount to 10,000 to keep the vector data size small
To transform vectorized paragraph to plain text see below
------------------------------------------------
print(train_data[0])
print(train_labels[0])
word_index = imdb.get_word_index()
reverse_word_index = dict(
     [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_review)
"""
