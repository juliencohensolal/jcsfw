from collections import Counter

import numpy as np
import pandas as pd
import re
import tensorflow as tf

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)

AUTO = tf.data.AUTOTUNE


### ALL ###

def get_img_batches(conf, dataset, shuffle=False, buffer_size=1000):
    if shuffle:
        return dataset.shuffle(buffer_size).batch(conf.batch_size).prefetch(buffer_size=AUTO)
    else:
        return dataset.batch(conf.batch_size).prefetch(buffer_size=AUTO)


def show_data_shape(train_batches, title):
    LOG.info(title + " " + str(train_batches))
    LOG.info(title + " data shapes:")
    for image, label in train_batches.take(1):
        LOG.info(str(image.numpy().shape) + " - " +  str(label.numpy().shape))


def one_hot_encode_labels(image, label, conf, conf_proj):
    if len(label.shape)==1:
        label2 = tf.one_hot(label, conf_proj.n_classes)
    else:
        label2 = label
    return image, label2


def get_balanced_weights(train_dataset):
    images, labels = tuple(zip(*train_dataset))
    train_labels = np.array(labels)
    label_counter = Counter(train_labels)
    max_count = float(max(label_counter.values()))
    balanced_weights = {class_id : max_count/id_count for class_id, id_count in label_counter.items()}
    return balanced_weights


### TFRECORDS ###

def load_tfrecords_from_filenames(conf, conf_proj, filenames, phase, labeled=True, ordered=False):
    # Count data
    n_images = count_tfrecords_items(filenames)
    LOG.info("Number of " + phase + " images : " + str(n_images))

    # Automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    
    # Uses data as soon as it streams in, rather than in its original order, increases speed
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
    dataset = dataset.with_options(ignore_order)
    if labeled:
        dataset = dataset.map(lambda x: read_labeled_tfrecord(
            x, conf.img_size, conf.channels, conf_proj), num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(lambda x: read_unlabeled_tfrecord(
            x, conf.img_size, conf.channels, conf_proj), num_parallel_calls=AUTO)
    return dataset, n_images


def count_tfrecords_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def read_labeled_tfrecord(example, img_size, channels, conf_proj):
    LABELED_TFREC_FORMAT = {
        conf_proj.tfrec_img: tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        conf_proj.tfrec_class: tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example[conf_proj.tfrec_img], img_size, channels)
    label = tf.cast(example[conf_proj.tfrec_class], tf.int32)
    return image, label # returns a dataset of (image, label) pairs


def read_unlabeled_tfrecord(example, img_size, channels, conf_proj):
    UNLABELED_TFREC_FORMAT = {
        conf_proj.tfrec_img: tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example[conf_proj.tfrec_img], img_size, channels)
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)


def decode_image(image_data, img_size, channels):
    image = tf.image.decode_jpeg(image_data, channels=channels)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*[img_size, img_size], channels])
    return image


### CSV ###

'''def load_csv(conf_proj, verbose=True):
    csvpath = "../_data/" + conf_proj.project + "/train.csv"
    train = pd.read_csv(csvpath)
    if verbose:
        LOG.info("Train dimensions : " + str(train.shape))
        LOG.info(train.head(2))
    return train'''


'''def create_img_dataset(df, column_files, column_labels):
    img_dataset = tf.data.Dataset.from_tensor_slices((df[column_files].values, df[column_labels].values))
    img_dataset = img_dataset.map(load_image_and_label_from_path, num_parallel_calls=AUTO)
    return img_dataset'''


'''def load_image_and_label_from_path(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img, label'''
