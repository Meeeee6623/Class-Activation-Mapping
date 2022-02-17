# imports
import os
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import scipy

# check for cuda
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# dataset name
dataset = 'mnist'

# model path
path = os.getcwd()
model_path = os.path.join(path, f'{dataset} Model')

if not os.path.exists(model_path):
    os.makedirs(model_path)

# image info
img_width, img_height = 224, 224

train_data_dir = os.path.join(path, rf'{dataset}\train')
test_data_dir = os.path.join(path, rf'{dataset}\test')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

batch_size = 16

# image generators for train and test data
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


def extract_labels(data_path):
    """ this function returns a list of labels corresponding to the path provided """
    return os.listdir(data_path)


def get_label(label_categorical, class_list):
    """
    this function returns the name of the label corresponding to the provided category

    Inputs:
        1) : label_categorical (numpy array) : categorical array containing only a 1 for the selected label
        2) : class_list (list of strings) : list of strings containing the class names (extracted form the directory)

    Outputs:
        returns the string containing the correct class name
    """
    # get index where label_categorical contains the 1
    index = np.where(label_categorical == 1)[0][0]
    # get label from class list using the index and return it
    return class_list[index]


# print one example image and it's corresponding label from training set
batch_num = 0
img_nr = np.random.choice(np.arange(0, batch_size))

# get list of class names
class_list = extract_labels(train_data_dir)

# train generator returns tuple (x, y) where x contains all the images of shape (batch_size, 224, 224, 3) and y contains the
# corresponding labels of shape (batch_size, 131)
train_batch_tuple = train_generator[batch_num]
train_batch_data = train_batch_tuple[0]
train_batch_label = train_batch_tuple[1]

# get label name from class num
class_name = get_label(train_batch_label[img_nr], class_list)

# plot image with labeled title
img = train_batch_data[img_nr]
plt.imshow(img)
plt.title(class_name)
plt.show()

# include_top false -> without last FC layer
model_res50 = applications.ResNet50(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = model_res50.output
x = GlobalAveragePooling2D()(x)

# add a flatten layer
x = Flatten()(x)

# and a fully connected output/classification layer
predictions = Dense(131, activation='softmax')(x)

# create the full network so we can train on it
transfer_model = Model(inputs=model_res50.input, outputs=predictions)

# compile model
transfer_model.compile(loss='categorical_crossentropy',
                       optimizer=optimizers.SGD(learning_rate=1e-4, momentum=0.9),
                       metrics=['accuracy'])


# train
with tf.device("/device:GPU:0"):
    history_pretrained = transfer_model.fit(
        train_generator,
        epochs=5, shuffle=True, verbose=1, validation_data=test_generator)

model_name = 'trained_fruits_classifier.h5'
model_path_new = os.path.join(model_path, model_name)
# save full model for CAM
transfer_model.save(model_path_new)
# load it later using "new_model = tf.keras.models.load_model(model_path_new)""
# summarize history for accuracy
plt.plot(history_pretrained.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# summarize history for loss
plt.plot(history_pretrained.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
