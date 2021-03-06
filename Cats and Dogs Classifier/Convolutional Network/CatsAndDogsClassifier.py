# This model will classify whether a picture contains a cat or a dog
import os, shutil
from keras import models, layers, optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image
from keras.applications import VGG16

# Directories on disk where we will store data
original_dataset_dir = '/Users/jamesperalta/Datasets/Unzipped/train'
base_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs'

# Cat Directories
train_cats_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/train/cats'
validation_cats_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/validation/cats'
test_cats_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/test/cats'

# Dog Directories
train_dogs_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/train/dogs'
validation_dogs_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/validation/dogs'
test_dogs_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/test/dogs'


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model


# Pre-process the images
train_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/train'
validation_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/validation'
test_dir = '/Users/jamesperalta/Datasets/Partitioned/CatsAndDogs/test'

# Add Data Augmentation to your training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create Generators from the directory's
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')


con_base = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(150, 150, 3))

print(con_base.summary())


# After pre-processing fit your model
# model = create_model()
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=100,
#       epochs=100,
#       validation_data=validation_generator,
#       validation_steps=50)
#
# model.save('cats_and_dogs_small_1.h5')
#
# # Plot your data here
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation Accuracy')
# plt.legend()
#
# plt.figure()
#
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()



# Create directories for training, validation, and testing
"""
os.mkdir(base_dir)  # Base Directory

# Training Directory
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# Validation Directory
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)

# Test Directory
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)
"""

# Transfer over Data from unzipped into the Cat and Dog folders
"""
cat_names = ['cat.{}.jpg'.format(i) for i in range(2000)]
# Training
for cat in cat_names[0:1000]:
    src = os.path.join(original_dataset_dir, cat)
    dst = os.path.join(train_cats_dir, cat)
    shutil.copyfile(src, dst)
# Validation
for cat in cat_names[1000:1500]:
    src = os.path.join(original_dataset_dir, cat)
    dst = os.path.join(validation_cats_dir, cat)
    shutil.copyfile(src, dst)
# Testing
for cat in cat_names[1500:2000]:
    # Training
    src = os.path.join(original_dataset_dir, cat)
    dst = os.path.join(test_cats_dir, cat)
    shutil.copyfile(src, dst)

# Transfer over Data from unzipped into the Dog folders
dog_names = ['dog.{}.jpg'.format(i) for i in range(2000)]
# Training
for dog in dog_names[0:1000]:
    src = os.path.join(original_dataset_dir, dog)
    dst = os.path.join(train_dogs_dir, dog)
    shutil.copyfile(src, dst)

# Validation
for dog in dog_names[1000:1500]:
    src = os.path.join(original_dataset_dir, dog)
    dst = os.path.join(validation_dogs_dir, dog)
    shutil.copyfile(src, dst)
# Testing
for dog in dog_names[1500:2000]:
    src = os.path.join(original_dataset_dir, dog)
    dst = os.path.join(test_dogs_dir, dog)
    shutil.copyfile(src, dst)
"""
