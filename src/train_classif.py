from argparse import ArgumentParser
import os
import sys
from time import time

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf

from projects import cassava, flowers
from utils import augmentations, c_logging, config, cv, data, loss_handler, lr_handler
from utils import metric_handler, network_handler, optimizer_handler, startup, visualize


LOG = c_logging.getLogger(__name__)

# Reduce Matplotlib log level
c_logging.getLogger("matplotlib").setLevel(20)

AUTO = tf.data.AUTOTUNE


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, help="Config file for experiment in folder 'cfg'")
    parser.add_argument(
        "--cfg_proj", type=str, help="Config file for project in folder 'cfg'")
    args = parser.parse_args()
    return args


if __name__ == '__main__' :
    # Load configuration files
    args = parse_args()
    conf = config.load_config("cfg/" + args.cfg)
    conf_proj = config.load_config("cfg/" + args.cfg_proj)

    # Init logging
    experiment_id = int(time())
    experiment_dir = "./experiments/train/" + conf_proj.project + "_" + conf_proj.task + \
            "_" + str(experiment_id) + "/"
    c_logging.config(
        project=conf_proj.project, 
        task=conf_proj.task, 
        experiment_id=experiment_id, 
        experiment_dir=experiment_dir,
        log_level=conf.log_level)

    # Setup everything
    LOG.info("Setup everything")
    startup.allow_gpu_ram_growth()
    startup.set_mixed_precision()
    startup.seed_everything(conf.seed)
    config.save_config("cfg/", args.cfg, experiment_dir)
    config.save_config("cfg/", args.cfg_proj, experiment_dir)

    # Load dataset
    LOG.info("Load dataset")
    if conf_proj.tfrecords:
        LOG.info("TFRecords data format")
        
        if conf_proj.val_path != "":
            # External validation set
            LOG.info("External validation set")
            train_dataset, nb_train_images = data.load_tfrecords(conf, conf_proj, "train")
            val_dataset, nb_val_images = data.load_tfrecords(conf, conf_proj, "val")
        else:
            # No exteral validation set, must create ourselves
            LOG.info("No exteral validation set, must create ourselves")
            train_dataset, nb_train_images, val_dataset, nb_val_images = cv.split_tfrec_data(conf, conf_proj)
    else:
        LOG.info("CSV data format NOT HANDLED")
        sys.exit()
        '''train = data.load_csv(conf_proj)

        # Shuffle dataset
        LOG.info("Shuffle dataset")
        train = shuffle(train, random_state=conf.seed)
        train.reset_index(drop=True, inplace=True)

        # Run project-specific processing
        if conf_proj.project == "Cassava":
            train = cassava.preprocessing(train)
            [imgpath_col, label_col] = cassava.get_dataset_names(train)
        else:
            LOG.error("Unknown project, quitting")
            sys.exit()

        # Split train and validation
        LOG.info("Split train and validation")
        train, val = cv.split_csv_data(conf, train, label_col)

        # Create image datasets
        LOG.info("Create image datasets")
        train_dataset = data.create_img_dataset(train, imgpath_col, label_col)
        val_dataset = data.create_img_dataset(val, imgpath_col, label_col)'''

    # Add augmentations
    LOG.info("Add augmentations")
    train_dataset = train_dataset.map(
        lambda image, label: augmentations.add_augmentations(image, label, conf), num_parallel_calls=AUTO)

    if conf.balanced_weights:
        # Get balanced weights
        LOG.info("Get balanced weights")
        balanced_weights = data.get_balanced_weights(train_dataset)

    # The training dataset must repeat for several epochs
    train_dataset = train_dataset.repeat()

    # Get batches
    LOG.info("Get batches")
    train_batches = data.get_img_batches(conf, train_dataset, shuffle=True)
    data.show_data_shape(train_batches, "Training")
    val_batches = data.get_img_batches(conf, val_dataset)
    data.show_data_shape(val_batches, "Validation")

    if conf.train_cutmix_proba > 0:
        # Apply cutmix
        LOG.info("Apply cutmix")
        train_batches = train_batches.repeat().map(
            lambda image, label: augmentations.add_cutmix(image, label, conf, conf_proj), num_parallel_calls=AUTO)
        data.show_data_shape(train_batches, "Training")
        val_batches = val_batches.map(
            lambda image, label: data.one_hot_encode_labels(image, label, conf, conf_proj), num_parallel_calls=AUTO)
        data.show_data_shape(val_batches, "Validation")
    elif conf.train_mixup_proba > 0:
        # Apply mixup
        LOG.info("Apply mixup")
        train_batches = train_batches.repeat().map(
            lambda image, label: augmentations.add_mixup(image, label, conf, conf_proj), num_parallel_calls=AUTO)
        data.show_data_shape(train_batches, "Training")
        val_batches = val_batches.map(
            lambda image, label: data.one_hot_encode_labels(image, label, conf, conf_proj), num_parallel_calls=AUTO)
        data.show_data_shape(val_batches, "Validation")

    # Save first augmented images
    if conf.nb_saved_augmented_img > 0:
        LOG.info("Save first augmented images")
        first_batch = next(iter(train_batches.unbatch().batch(conf.nb_saved_augmented_img)))
        visualize.save_first_images(experiment_dir, first_batch)

    # Get base model
    LOG.info("Get model")
    x, inputs, nb_layers = network_handler.get_base_model(conf)

    # Add head
    LOG.info("Add head")
    if conf_proj.project == "Cassava":
        model = cassava.add_classif_head(conf, conf_proj, x, inputs)
    elif conf_proj.project == "Flowers":
        model = flowers.add_classif_head(conf, conf_proj, x, inputs)
    else:
        LOG.error("Unknown project, quitting")
        sys.exit()

    # Print model summary
    if conf.model_summary:
        LOG.info("Print model summary")
        model.summary(print_fn=LOG.info)

    if conf.warmup_epochs > 0:
        # WARM UP

        # Compile warm-up model
        LOG.info("Compile warm-up model")
        optimizer = optimizer_handler.get_optimizer(conf, is_warmup=True)
        loss = loss_handler.get_loss(conf)
        metrics = metric_handler.get_metrics(conf)
        if (optimizer is None) | (loss is None) | (metrics is None):
            sys.exit()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Train model
        LOG.info("Train warm-up model")
        if conf.balanced_weights:
            history = model.fit(
                x=train_batches, 
                validation_data=val_batches, 
                epochs=conf.warmup_epochs, 
                steps_per_epoch=nb_train_images//conf.batch_size, 
                class_weight=balanced_weights)
        else:
            history = model.fit(
                x=train_batches, 
                validation_data=val_batches, 
                epochs=conf.warmup_epochs, 
                steps_per_epoch=nb_train_images//conf.batch_size)

    # Unfreeze layers if needed
    LOG.info("Unfreeze layers if needed")
    model = network_handler.unfreeze(conf, model, nb_layers)

    # Compile model
    LOG.info("Compile model")
    optimizer = optimizer_handler.get_optimizer(conf)
    loss = loss_handler.get_loss(conf)
    metrics = metric_handler.get_metrics(conf)
    if (optimizer is None) | (loss is None) | (metrics is None):
        sys.exit()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Set up early stopping
    callbacks = []
    if conf.early_stopping_epochs > 0:
        LOG.info("Set up early stopping")
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=conf.early_stopping_epochs, 
            restore_best_weights=conf.restore_best_after_training)
        callbacks.append(es_callback)

    # Set up learning rate
    if conf.lr is not None:
        LOG.info("Set up learning rate")
        lr = lr_handler.get_learning_rate(conf)
        if lr is None:
            sys.exit()
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr)
        callbacks.append(lr_callback)

    # Plug to Tensorboard
    if conf.tensorboard:
        LOG.info("Plug Tensorboard")
        tensorboard_dir = experiment_dir + "tensorboard/"
        os.makedirs(tensorboard_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)
        callbacks.append(tensorboard_callback)

    # Set up checkpoints
    if conf.checkpoints:
        LOG.info("Set up checkpoints")
        checkpoint_dir = experiment_dir + "checkpoints/"
        os.makedirs(checkpoint_dir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + conf.base + "-{epoch:02d}-{val_loss:.4f}.ckpt",
            monitor="val_loss", 
            save_best_only=True,
            save_weights_only=True)
        callbacks.append(cp_callback)

    # Train model
    LOG.info("Train model")
    if conf.balanced_weights:
        history = model.fit(
            x=train_batches, 
            validation_data=val_batches, 
            epochs=conf.epochs, 
            steps_per_epoch=nb_train_images//conf.batch_size, 
            callbacks=callbacks, 
            class_weight=balanced_weights)
    else:
        history = model.fit(
            x=train_batches, 
            validation_data=val_batches, 
            epochs=conf.epochs, 
            steps_per_epoch=nb_train_images//conf.batch_size, 
            callbacks=callbacks)

