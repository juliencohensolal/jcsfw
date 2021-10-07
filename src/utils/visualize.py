import math

import numpy as np

from matplotlib import pyplot as plt

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def display_one_image(image, title, subplot, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(np.clip(image, 0, 1))
    if len(title) > 0:
        plt.title(
            title, fontsize=int(titlesize), color='black', fontdict={'verticalalignment':'center'}, 
            pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:
        # If no labels, only image IDs, return None for labels (this is the case for test data)
        numpy_labels = [None for _ in enumerate(numpy_images)]
    return numpy_images, numpy_labels


def save_first_images(experiment_dir, databatch):
    images, labels = batch_to_numpy_images_and_labels(databatch)

    # Auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # Size and spacing
    FIGSIZE = 13
    SPACING = 0.1
    subplot=(rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))

    # Display
    for image in images[:rows*cols]:
        # Magic formula tested to work from 1x1 to 10x10 images
        dynamic_titlesize = FIGSIZE * SPACING / max(rows,cols) * 40 + 3
        subplot = display_one_image(image, "", subplot, titlesize=dynamic_titlesize)

    # Layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    # Save
    plt.savefig(experiment_dir + "first_images.pdf", bbox_inches='tight')
