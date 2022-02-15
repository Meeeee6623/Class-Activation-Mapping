# imports
import os

from matplotlib import gridspec
from tensorflow import keras
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import cv2
from skimage.transform import resize
import scipy

# format matplotlib
plt.rcParams['figure.figsize'] = (8.0, 16.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

# store path to trained model
path = os.getcwd()
model_path = os.path.join(path, 'Trained Fruits Model')
# match to ResNet 50 input size
img_width, img_height = 224, 224

test_data_dir = os.path.join(path, r'fruits-360\test')

# load pre-trained model
model_name = 'trained_fruits_classifier.h5'
model_path_new = os.path.join(model_path, model_name)
loaded_model = tf.keras.models.load_model(model_path_new)


# batch_size = 1000
#
# # match pre-processing for training data
# test_datagen = ImageDataGenerator(rescale=1. / 255)
# test_generator = test_datagen.flow_from_directory(
#     test_data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical')


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


def get_class_activation_map(model, predictions, img):
    """
    this function computes the class activation map

    Inputs:
        1) model (tensorflow model) : trained model
        2) img (numpy array of shape (224, 224, 3)) : input image
    """

    # expand dimension to fit the image to a network accepted input size
    img = np.expand_dims(img, axis=0)

    label_index = np.argmax(predictions)

    # Get the 2048 input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]

    # get the final conv layer
    final_conv_layer = model.get_layer("conv5_block3_out")

    # create a function to fetch the final conv layer output maps (should be shape (1, 7, 7, 2048))
    get_output = K.function([model.layers[0].input], [final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])

    # squeeze conv map to shape image to size (7, 7, 2048)
    conv_outputs = np.squeeze(conv_outputs)

    # bilinear upsampling to resize each filtered image to size of original image
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (32, 32, 1), order=1)  # dim: 224 x 224 x 2048

    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((224 * 224, 2048)), class_weights_winner).reshape(224,
                                                                                                 224)  # dim: 224 x 224

    # return class activation map
    return final_output, label_index


def plot_class_activation_map(CAM, image, show=True, data_path=test_data_dir, class_index=None, class_name=None, classification=None,
                              file_name=None):
    """
    this function plots the activation cmap_map

    Inputs:
        1) CAM (numpy array of shape (224, 224)) : class activation cmap_map containing the trained heat cmap_map
        2) img (numpy array of shape (224, 224, 3)) : input image
        3) label (uint8) : index of the winning class
        4) data_path (string) : path to the images -> used to extract the class labels by extracting all local subdirectories
        5) ax (matplotlib axes) : axes where current CAM should be plotted
    """
    fig, axes = plt.subplots(2)
    # plt.tick_params(left=False, right=False, labelleft=False,
    #                 labelbottom=False, bottom=False)
    # axes[1].set_axis_off()

    cmap_map = mpl.cm.get_cmap("jet")
    # cmap_map.set_under(alpha=0)

    # plot image
    axes[0].imshow(image, alpha=1)
    axes[1].imshow(image, alpha=1)

    # normalize CAM values
    CAM /= np.max(CAM)

    # plot class activation map
    axes[1].imshow(CAM, cmap=cmap_map, alpha=.5)
    # get string for classified class
    class_list = extract_labels(data_path)
    if class_name is None:
        class_name = class_list[class_index]

    # add network's classification label
    axes[0].set_title(f'Classified as:\n{classification}', fontsize=28)
    axes[1].set_title('CAM Data', fontsize=28)

    # disable ticks
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([], minor=True)
        ax.grid(True)

    # true class label
    fig.suptitle(class_name, fontsize=40)
    if file_name is not None:
        plt.savefig(file_name)
    if show:
        plt.show()


def plot_single_CAM(model, class_name=None, class_index=None, image_path=None):
    """
    Chooses two random images from the specified class, classifies them with the given model, and saves them along with their CAM data in a 2x2 grid.
    Inputs:
        1) model (loaded keras model) : used to classify images and get CAM data
        2) class_name (string) : class to get images from to classify
        4) file_name (string) : name of file to save plot as
        5) image_path (string) : (optional) path to an image. Used to classify a specific image instead of one chosen at random from the given class.

    """
    if image_path is not None:
        img_path = os.path.join(os.getcwd(), image_path)
        class_name = os.path.abspath(os.path.join(img_path, os.pardir))
        class_list = extract_labels(os.path.join(r'fruits-360\test'))
    else:
        # get` test path and class names
        test_path = os.path.join(os.getcwd(), r'fruits-360\test')
        class_list = extract_labels(test_path)
        if class_name is None:
            class_name = class_list[class_index]
        class_path = os.path.join(test_path, class_name)
        img_path = os.path.join(class_path, np.random.choice(os.listdir(class_path)))

    img = mpl.image.imread(img_path)

    # fit to right size
    img = resize(img, (224, 224))

    # expand dimension to fit the image to a network accepted input size of (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    assert img.shape == (1, 224, 224, 3)

    # make prediction using the already trained model
    predictions = loaded_model.predict(img)

    classification = class_list[np.argmax(predictions)]

    # set image dimensions back to (224, 224, 3)
    img = np.squeeze(img)

    class_activation_map = get_class_activation_map(loaded_model, predictions, img)[0]  # don't need label index

    plot_class_activation_map(class_activation_map, img, class_name=class_name, classification=classification, file_name='test.png')


def save_class_CAM_2x2(model, class_name, file_name='test.png', show=False):
    """
    Chooses two random images from the specified class, classifies them with the given model, and saves them along with their CAM data in a 2x2 grid.
    Inputs:
        1) model (loaded keras model) : used to classify images and get CAM data
        2) class_name (string) : class to get images from to classify
        4) file_name (string) : name of file to save plot as
        5) show (bool) : (optional) show plots as they are made

    """
    # get test path and class names
    test_path = os.path.join(os.getcwd(), r'fruits-360\test')
    class_path = os.path.join(test_path, class_name)
    class_list = extract_labels(test_path)
    cam_data = []
    images = []
    # save all classifications
    classes = []
    for _ in range(2):
        img_path = os.path.join(class_path, np.random.choice(os.listdir(class_path)))
        img = mpl.image.imread(img_path)

        # fit to right size
        img = resize(img, (224, 224))

        # expand dimension to fit the image to a network accepted input size of (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)
        assert img.shape == (1, 224, 224, 3)

        # make prediction using the already trained model
        predictions = loaded_model.predict(img)
        classes.append(class_list[np.argmax(predictions)])
        if class_list[np.argmax(predictions)] != class_name:
            with open(r'cam_data\incorrect_classifications.txt', 'a') as f:
                f.write(f'{class_name}: {img_path}\n')

        # set image dimensions back to (224, 224, 3)
        img = np.squeeze(img)
        images.append(img)

        class_activation_map = get_class_activation_map(loaded_model, predictions, img)[0]  # don't need label index
        cam_data.append(class_activation_map)

    # get color map
    cmap_map = mpl.cm.get_cmap("jet")

    fig = plt.figure(figsize=(16, 16))
    # specifications for 2x2 grid
    gs = gridspec.GridSpec(2, 2, wspace=0.02, hspace=0.02, top=0.92, bottom=0.08, left=0.08, right=0.92)

    # plot images in a 2x2 grid
    for i in range(2):
        for j in range(2):
            ax = plt.subplot(gs[i, j])

            # plot image
            ax.imshow(images[j])

            # show CAM in bottom 2 images
            if i == 1:
                # normalize CAM values
                cam_data[j] /= np.max(cam_data[j])

                # plot class activation map
                ax.imshow(cam_data[j], cmap=cmap_map, alpha=.5)
            # if classified incorrectly, show classified class name above image
            else:
                if classes[i] != class_name:
                    ax.set_title(classes[j], fontsize=16)

            # set line width
            for child in ax.get_children():
                if isinstance(child, mpl.spines.Spine):
                    child.set_linewidth(2)
                    # give incorrect classifications red borders
                    if classes[j] != class_name:
                        child.set_color('#ff0000')

            # disable ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([], minor=True)

    # title and save plot
    fig.suptitle(class_name, fontsize=48)
    plt.savefig(file_name)
    if show:
        plt.show()
    plt.close(fig)


# runtime logic
plot_single_CAM(loaded_model, image_path=r"C:\Users\bchauhan22\Downloads\download.png")
exit()

# clear list
with open(r'cam_data\incorrect_classifications.txt', 'w') as f:
    f.write('')

for class_name in os.listdir(test_data_dir):
    save_class_CAM_2x2(loaded_model, class_name, file_name=rf'cam_data\{class_name}.png', show=False)
    print(f'Saved: {class_name}')
