from argparse import ArgumentParser
import os
import sys
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf

from projects import cassava, flowers
from utils import c_logging, config, data, network_handler, startup


LOG = c_logging.getLogger(__name__)

# Reduce Matplotlib log level
c_logging.getLogger("matplotlib").setLevel(20)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--project", type=str, help="Name of the project")
    parser.add_argument(
        "--train_dir", type=str, help="Folder where to find the train experiment")
    args = parser.parse_args()
    return args


if __name__ == '__main__' :
    # Load configuration files
    args = parse_args()
    experiment_dir_train = './experiments/train/' + args.train_dir + "/"
    conf_train = config.load_config(experiment_dir_train + "cfg_experiment.yml")
    conf_proj_train = config.load_config(experiment_dir_train + "cfg_proj_" + args.project + ".yml")

    # Init logging
    experiment_id = int(time())
    experiment_dir = './experiments/predict/' + args.train_dir + "_pred_" + str(experiment_id) + "/"
    task = args.train_dir.split("_")[1]
    c_logging.config(
        project=args.project, 
        task=task, 
        experiment_id=experiment_id, 
        experiment_dir=experiment_dir,
        log_level=conf_train.log_level)

    # Setup everything
    LOG.info("Setup everything")
    startup.allow_gpu_ram_growth()
    startup.set_mixed_precision()
    startup.seed_everything(conf_train.seed)
    config.save_config(
        experiment_dir_train, "cfg_experiment.yml", experiment_dir, suffix="_pred")
    config.save_config(
        experiment_dir_train, "cfg_proj_" + args.project + ".yml", experiment_dir, suffix="_pred")

    # Load dataset
    LOG.info("Load dataset")
    if conf_proj_train.tfrecords:
        LOG.info("TFRecords data format")
        test_dataset, nb_test_images = data.load_tfrecords(conf_train, conf_proj_train, "test")
    else:
        LOG.info("CSV data format - NOT HANDLED")
        sys.exit()

    # Get batches
    LOG.info("Get batches")
    test_batches = data.get_img_batches(conf_train, test_dataset)
    data.show_data_shape(test_batches, "Test")

    # Get base model
    LOG.info("Get model")
    x, inputs, nb_layers = network_handler.get_base_model(conf_train)

    # Add head
    LOG.info("Add head")
    if conf_proj_train.project == "Cassava":
        model = cassava.add_classif_head(conf_train, conf_proj_train, x, inputs)
    elif conf_proj_train.project == "Flowers":
        model = flowers.add_classif_head(conf_train, conf_proj_train, x, inputs)
    else:
        LOG.error("Unknown project, quitting")
        sys.exit()

    # Print model summary
    if conf_train.model_summary:
        LOG.info("Print model summary")
        model.summary(print_fn=LOG.info)

    # Load pretrained weights
    LOG.info("Load pretrained weights")
    weights = ""
    checkpoint_dir = experiment_dir_train + "checkpoints/"
    weights = tf.train.latest_checkpoint(checkpoint_dir)
    LOG.info("Weights : " + weights)
    model.load_weights(weights)

    # Get image IDs from test set
    LOG.info("Get image IDs from test set")
    test_ids_ds = test_batches.map(lambda img, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(nb_test_images))).numpy().astype('U')

    # Predict test images
    LOG.info("Predict test images")
    probas = model.predict(test_batches)
    df_probas = pd.DataFrame(np.vstack(probas))
    col_names = df_probas.columns.values.tolist()
    df_probas["id"] = test_ids
    df_probas = df_probas[["id"] + col_names]
    df_probas.to_csv(experiment_dir + "probas.csv", header=True, index=False)

    # Generate predictions from probas
    LOG.info("Generate predictions from probas")
    predictions = np.argmax(probas, axis=-1)
    df_probas = pd.DataFrame({'id': test_ids, 'label': predictions})
    df_probas.to_csv(experiment_dir + "preds.csv", header=True, index=False)
