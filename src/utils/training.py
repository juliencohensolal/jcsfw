import os
import sys

import tensorflow as tf

import utils.lr_handler as lr_handler
import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def set_callbacks(conf, experiment_dir, fold_idx):
    # Set up early stopping
    callbacks = []
    if conf.early_stopping_epochs > 0:
        LOG.debug("Set up early stopping")
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=conf.early_stopping_epochs, 
            restore_best_weights=conf.restore_best_after_training)
        callbacks.append(es_callback)

    # Set up learning rate
    if conf.lr is not None:
        LOG.debug("Set up learning rate")
        lr = lr_handler.get_learning_rate(conf)
        if lr is None:
            sys.exit()
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr)
        callbacks.append(lr_callback)

    # Plug to Tensorboard
    if conf.tensorboard:
        LOG.debug("Plug Tensorboard")
        tensorboard_dir = experiment_dir + "tensorboard/fold_" + str(fold_idx) + "/"
        os.makedirs(tensorboard_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
        callbacks.append(tensorboard_callback)

    # Set up checkpoints
    if conf.checkpoints:
        LOG.debug("Set up checkpoints")
        checkpoint_dir = experiment_dir + "checkpoints/fold_" + str(fold_idx) + "/"
        os.makedirs(checkpoint_dir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + conf.base + "-{epoch:02d}-{val_loss:.4f}.ckpt",
            monitor="val_loss", 
            save_best_only=True,
            save_weights_only=True)
        callbacks.append(cp_callback)

    return callbacks
