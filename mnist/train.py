# imports
import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import applications
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

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

data_dir = os.path.join(path, rf'{dataset}')
test_data_dir = os.path.join(path, rf'{dataset}\test')

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    data_dir=data_dir,
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
fig = tfds.show_examples(ds_train, ds_info)
plt.show()

BATCH_SIZE = 16


def normalize_img(image, label):
    """Re-sizes and normalizes images to fit ResNet model: `uint8` -> `float32`."""
    image = tf.image.resize(image, [244, 244])
    image = tf.image.grayscale_to_rgb(image)
    return tf.cast(image, tf.float32) / 255., label


# image generators for train and test data
ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(16)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


# include_top false -> without last FC layer
model_res50 = applications.ResNet50(weights='imagenet', include_top=False, input_shape = (ds_info))

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
        ds_train,
        epochs=5,
        verbose=1,
        validation_data=ds_test)

model_name = f'{dataset} trained classifier.h5'
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
