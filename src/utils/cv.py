from sklearn.model_selection import train_test_split
import tensorflow as tf

import utils.data as data
import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def split_csv_data(conf, df, col_label, verbose=True):
    stratify = None
    if conf.stratified:
        stratify = df[col_label]

    train, val = train_test_split(
        df, 
        test_size=conf.val_size, 
        random_state=conf.seed, 
        stratify=stratify)
    if verbose:
        LOG.info("CV train size : " + str(train.shape))
        LOG.info("CV val size : " + str(val.shape))

    return train, val


def split_tfrec_data(conf, conf_proj):
    train_filenames, val_filenames = train_test_split(
        tf.io.gfile.glob(conf_proj.train_path + "*.tfrec"),
        test_size=conf.val_size, 
        random_state=conf.seed)

    train_dataset, nb_train_images = data.load_tfrecords_from_filenames(
        conf, conf_proj, train_filenames, "train", labeled=True)
    val_dataset, nb_val_images = data.load_tfrecords_from_filenames(
        conf, conf_proj, val_filenames, "val", labeled=True)

    return train_dataset, nb_train_images, val_dataset, nb_val_images
