import random

import tensorflow as tf

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def add_augmentations(image, label, conf):
    seed = tf.random.uniform((2,), minval=0, maxval=100, dtype=tf.int32)

    if conf.train_random_flip_left_right:
        image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    if conf.train_random_flip_up_down:
        image = tf.image.stateless_random_flip_up_down(image, seed=seed)
    if conf.train_rot90:
        proba = random.random()
        if proba > .75:
            image = tf.image.rot90(image, k=3) # Rotate 270º
        elif .5 > proba > .25:
            image = tf.image.rot90(image, k=1) # Rotate 90º
    if conf.train_random_brightness != 0:
        image = tf.image.stateless_random_brightness(image, max_delta=conf.train_random_brightness, seed=seed)
    if conf.train_random_hue != 0:
        image = tf.image.stateless_random_hue(image, max_delta=conf.train_random_hue, seed=seed)
    if conf.train_random_saturation != [0, 0]:
        image = tf.image.stateless_random_saturation(
            image, lower=conf.train_random_saturation[0], upper=conf.train_random_saturation[1], seed=seed)
    if conf.train_random_contrast != [0, 0]:
        image = tf.image.stateless_random_contrast(
            image, lower=conf.train_random_contrast[0], upper=conf.train_random_contrast[1], seed=seed)
    if conf.train_random_crop != 0:
        image = tf.image.stateless_random_crop(
            image, size=[conf.train_random_crop, conf.train_random_crop, conf.channels], seed=seed)
    if conf.train_central_crop != 0:
        image = tf.image.central_crop(
            image, central_fraction=conf.train_central_crop)

    return image, label


def add_cutmix(image, label, conf, conf_proj):
    # See https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu/comments#759542
    # input : image - batch of images of size [n,dim,dim,3], not a single image of [dim,dim,3]
    # output : batch of images with cutmix applied
    imgs, labs = [], []
    for j in range(conf.batch_size):
        # Define actual probability of doing cutmix (i.e. 0 or 1 integer)
        actual_proba = tf.cast(tf.random.uniform([], 0, 1) <= conf.train_cutmix_proba, tf.int32)

        # Randomly choose image in batch to apply cutmix with
        cutwith_img_idx = tf.cast(tf.random.uniform([], 0, conf.batch_size), tf.int32)

        # Choose location to cut in image
        x = tf.cast(tf.random.uniform([], 0, conf.img_size), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, conf.img_size), tf.int32)
        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0
        cutwidth = tf.cast(conf.img_size * tf.math.sqrt(1 - b), tf.int32) * actual_proba
        ya = tf.math.maximum(0, y - cutwidth // 2)
        yb = tf.math.minimum(conf.img_size, y + cutwidth // 2)
        xa = tf.math.maximum(0, x - cutwidth // 2)
        xb = tf.math.minimum(conf.img_size, x + cutwidth // 2)

        # Assemble cutmix image
        one = image[j, ya:yb, 0:xa, :]
        two = image[cutwith_img_idx, ya:yb, xa:xb, :]
        three = image[j, ya:yb, xb:conf.img_size, :]
        middle = tf.concat([one, two, three], axis=1)
        img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:conf.img_size,:, :]], axis=0)
        imgs.append(img)

        # Update label of cutmix image
        # Warning : labels will be one hot encoded
        proportion = tf.cast((xb-xa) * (yb-ya) / conf.img_size / conf.img_size, tf.float32)
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j], conf_proj.n_classes)
            lab2 = tf.one_hot(label[cutwith_img_idx], conf_proj.n_classes)
        else:
            lab1 = label[j, ]
            lab2 = label[cutwith_img_idx, ]
        labs.append((1 - proportion)*lab1 + proportion*lab2)

    # Only useful for TPU? Not sure
    image2 = tf.reshape(tf.stack(imgs), (conf.batch_size, conf.img_size, conf.img_size, conf.channels))
    label2 = tf.reshape(tf.stack(labs), (conf.batch_size, conf_proj.n_classes))
    return image2, label2


def add_mixup(image, label, conf, conf_proj):
    # See https://www.kaggle.com/cdeotte/cutmix-and-mixup-on-gpu-tpu/comments#759542
    # input : image - batch of images of size [n,dim,dim,3], not a single image of [dim,dim,3]
    # output : batch of images with cutmix applied
    imgs, labs = [], []
    for j in range(conf.batch_size):
        # Define actual probability of doing mixup
        actual_proba = tf.cast(tf.random.uniform([], 0, 1) <= conf.train_mixup_proba, tf.float32)

        # Randomly choose image in batch to apply mixup with
        mixwith_img_idx = tf.cast(tf.random.uniform([], 0, conf.batch_size), tf.int32)

        # Set intensity of 2nd image
        mixwith_intensity = tf.random.uniform([], 0, 1) * actual_proba

        # Assemble mixup image
        img1 = image[j, ]
        img2 = image[mixwith_img_idx, ]
        imgs.append((1 - mixwith_intensity)*img1 + mixwith_intensity*img2)

        # Update label of mixup image
        # Warning : labels will be one hot encoded
        if len(label.shape)==1:
            lab1 = tf.one_hot(label[j], conf_proj.n_classes)
            lab2 = tf.one_hot(label[mixwith_img_idx], conf_proj.n_classes)
        else:
            lab1 = label[j, ]
            lab2 = label[mixwith_img_idx, ]
        labs.append((1 - mixwith_intensity)*lab1 + mixwith_intensity*lab2)

    # Only useful for TPU? Not sure
    image2 = tf.reshape(tf.stack(imgs), (conf.batch_size, conf.img_size, conf.img_size, conf.channels))
    label2 = tf.reshape(tf.stack(labs), (conf.batch_size, conf_proj.n_classes))
    return image2, label2
