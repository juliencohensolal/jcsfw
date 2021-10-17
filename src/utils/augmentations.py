import random

import albumentations as alb
import numpy as np
import tensorflow as tf

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def add_augmentations(image, label, conf, conf_proj, is_train=True):
    def alb_aug(image):
        data = {conf_proj.tfrec_img: image}
        aug_data  = transforms(**data)
        aug_img  = aug_data[conf_proj.tfrec_img]
        aug_img = tf.cast(aug_img, tf.float32)
        return aug_img

    if is_train:
        # Train

        if conf.train_blur != [0, 0]:
            LOG.info("Applying train_blur")
            transforms = alb.Compose([alb.Blur(
                blur_limit=conf.train_blur[0], p=conf.train_blur[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_centercrop != [0, 0]:
            LOG.info("Applying train_centercrop")
            transforms = alb.Compose([alb.CenterCrop(
                height=int(conf.img_size - (conf.train_centercrop[0] * conf.img_size / 100)),
                width=int(conf.img_size - (conf.train_centercrop[0] * conf.img_size / 100)),
                p=conf.train_centercrop[1]),
                alb.Resize(conf.img_size, conf.img_size)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        '''if conf.train_clahe != [0, 0, 0]:
            LOG.info("Applying train_clahe")
            transforms = alb.Compose([alb.CLAHE(
                clip_limit=conf.train_clahe[0],
                tile_grid_size=[conf.train_clahe[1], conf.train_clahe[1]],
                p=conf.train_clahe[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)'''

        if conf.train_coarse_dropout != [0, 0, 0, 0]:
            LOG.info("Applying train_coarse_dropout")
            transforms = alb.Compose([alb.CoarseDropout(
                min_holes=conf.train_coarse_dropout[0],
                max_holes=conf.train_coarse_dropout[1],
                max_height=conf.train_coarse_dropout[2],
                max_width=conf.train_coarse_dropout[2],
                p=conf.train_coarse_dropout[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_crop != [0, 0, 0, 0, 0]:
            LOG.info("Applying train_crop")
            transforms = alb.Compose([alb.Crop(
                x_min=conf.train_crop[0],
                x_max=conf.train_crop[1],
                y_min=conf.train_crop[2],
                y_max=conf.train_crop[3],
                p=conf.train_crop[4]),
                alb.Resize(conf.img_size, conf.img_size)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_horizontal_flip != 0:
            LOG.info("Applying train_horizontal_flip")
            transforms = alb.Compose([alb.HorizontalFlip(p=conf.train_horizontal_flip)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_hue_sat_value != [0, 0, 0, 0]:
            LOG.info("Applying train_hue_sat_value")
            transforms = alb.Compose([alb.HueSaturationValue(
                hue_shift_limit=conf.train_hue_sat_value[0],
                sat_shift_limit=conf.train_hue_sat_value[1],
                val_shift_limit=conf.train_hue_sat_value[2],
                p=conf.train_hue_sat_value[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_random_brightness != [0, 0]:
            LOG.info("Applying train_random_brightness")
            transforms = alb.Compose([alb.RandomBrightness(
                limit=conf.train_random_brightness[0], p=conf.train_random_brightness[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_random_contrast != [0, 0]:
            LOG.info("Applying train_random_contrast")
            transforms = alb.Compose([alb.RandomContrast(
                limit=conf.train_random_contrast[0], p=conf.train_random_contrast[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_random_fog != [0, 0, 0, 0]:
            LOG.info("Applying train_random_fog")
            transforms = alb.Compose([alb.RandomFog(
                fog_coef_lower=conf.train_random_fog[0],
                fog_coef_upper=conf.train_random_fog[1],
                alpha_coef=conf.train_random_fog[2],
                p=conf.train_random_fog[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_random_grid_shuffle != [0, 0, 0]:
            LOG.info("Applying train_random_grid_shuffle")
            transforms = alb.Compose([alb.RandomGridShuffle(
                grid=(conf.train_random_grid_shuffle[0], conf.train_random_grid_shuffle[1]),
                p=conf.train_random_grid_shuffle[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_random_resized_crop != [0, 0, 0]:
            LOG.info("Applying train_random_resized_crop")
            transforms = alb.Compose([alb.RandomResizedCrop(
                height=conf.img_size,
                width=conf.img_size,
                ratio=(1., 1.),
                scale=(conf.train_random_resized_crop[0], conf.train_random_resized_crop[1]),
                p=conf.train_random_resized_crop[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_rotate != [0, 0, 0]:
            LOG.info("Applying train_rotate")
            transforms = alb.Compose([alb.Rotate(
                limit=conf.train_rotate[0],
                p=conf.train_rotate[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.train_vertical_flip != 0:
            LOG.info("Applying train_vertical_flip")
            transforms = alb.Compose([alb.VerticalFlip(p=conf.train_vertical_flip)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)
    else:
        # Predict

        if conf.test_blur != [0, 0]:
            LOG.info("Applying test_blur")
            transforms = alb.Compose([alb.Blur(
                blur_limit=conf.test_blur[0], p=conf.test_blur[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_centercrop != [0, 0]:
            LOG.info("Applying test_centercrop")
            transforms = alb.Compose([alb.CenterCrop(
                height=int(conf.img_size - (conf.test_centercrop[0] * conf.img_size / 100)),
                width=int(conf.img_size - (conf.test_centercrop[0] * conf.img_size / 100)),
                p=conf.test_centercrop[1]),
                alb.Resize(conf.img_size, conf.img_size)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        '''if conf.test_clahe != [0, 0, 0]:
            LOG.info("Applying test_clahe")
            transforms = alb.Compose([alb.CLAHE(
                clip_limit=conf.test_clahe[0],
                tile_grid_size=[conf.test_clahe[1], conf.test_clahe[1]],
                p=conf.test_clahe[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)'''

        if conf.test_coarse_dropout != [0, 0, 0, 0]:
            LOG.info("Applying test_coarse_dropout")
            transforms = alb.Compose([alb.CoarseDropout(
                min_holes=conf.test_coarse_dropout[0],
                max_holes=conf.test_coarse_dropout[1],
                max_height=conf.test_coarse_dropout[2],
                max_width=conf.test_coarse_dropout[2],
                p=conf.test_coarse_dropout[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_crop != [0, 0, 0, 0, 0]:
            LOG.info("Applying test_crop")
            transforms = alb.Compose([alb.Crop(
                x_min=conf.test_crop[0],
                x_max=conf.test_crop[1],
                y_min=conf.test_crop[2],
                y_max=conf.test_crop[3],
                p=conf.test_crop[4]),
                alb.Resize(conf.img_size, conf.img_size)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_horizontal_flip != 0:
            LOG.info("Applying test_horizontal_flip")
            transforms = alb.Compose([alb.HorizontalFlip(p=conf.test_horizontal_flip)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_hue_sat_value != [0, 0, 0, 0]:
            LOG.info("Applying test_hue_sat_value")
            transforms = alb.Compose([alb.HueSaturationValue(
                hue_shift_limit=conf.test_hue_sat_value[0],
                sat_shift_limit=conf.test_hue_sat_value[1],
                val_shift_limit=conf.test_hue_sat_value[2],
                p=conf.test_hue_sat_value[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_random_brightness != [0, 0]:
            LOG.info("Applying test_random_brightness")
            transforms = alb.Compose([alb.RandomBrightness(
                limit=conf.test_random_brightness[0], p=conf.test_random_brightness[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_random_contrast != [0, 0]:
            LOG.info("Applying test_random_contrast")
            transforms = alb.Compose([alb.RandomContrast(
                limit=conf.test_random_contrast[0], p=conf.test_random_contrast[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_random_fog != [0, 0, 0, 0]:
            LOG.info("Applying test_random_fog")
            transforms = alb.Compose([alb.RandomFog(
                fog_coef_lower=conf.test_random_fog[0],
                fog_coef_upper=conf.test_random_fog[1],
                alpha_coef=conf.test_random_fog[2],
                p=conf.test_random_fog[3])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_random_grid_shuffle != [0, 0, 0]:
            LOG.info("Applying test_random_grid_shuffle")
            transforms = alb.Compose([alb.RandomGridShuffle(
                grid=(conf.test_random_grid_shuffle[0], conf.test_random_grid_shuffle[1]),
                p=conf.test_random_grid_shuffle[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_random_resized_crop != [0, 0, 0]:
            LOG.info("Applying test_random_resized_crop")
            transforms = alb.Compose([alb.RandomResizedCrop(
                height=conf.img_size,
                width=conf.img_size,
                ratio=(1., 1.),
                scale=(conf.test_random_resized_crop[0], conf.test_random_resized_crop[1]),
                p=conf.test_random_resized_crop[2])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_rotate != [0, 0, 0]:
            LOG.info("Applying test_rotate")
            transforms = alb.Compose([alb.Rotate(
                limit=conf.test_rotate[0],
                p=conf.test_rotate[1])])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

        if conf.test_vertical_flip != 0:
            LOG.info("Applying test_vertical_flip")
            transforms = alb.Compose([alb.VerticalFlip(p=conf.test_vertical_flip)])
            image = tf.numpy_function(func=alb_aug, inp=[image], Tout=tf.float32)

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
