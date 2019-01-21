# Solves the problem of classifying handwritten digits between 0-9
# Using a Convolutional Neural Network
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical


# Define the models architecture
def create_model():
    # Build the architecture
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the network
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Add a third dimension to the Mnist data set and normalize the vectors to floats ranging 0 - 1
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One hot encode your labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Instantiate model
my_model = create_model()

# Fit the model on batch sizes of 64 in 5 epochs
my_model.fit(train_images, train_labels, batch_size=64, epochs=5)

print(my_model.evaluate(test_images, test_labels))
