from urllib.parse import urlparse

import mlflow
from mlflow.entities import Param
import mlflow.tensorflow
from mlflow.tracking import MlflowClient

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


client = MlflowClient()


def setup(mlf_exp_name, mlf_run_name):
    mlflow.set_tracking_uri('http://localhost:5000')
    exp_id = mlflow.set_experiment(mlf_exp_name)
    active_run = mlflow.start_run(experiment_id=exp_id, run_name=mlf_run_name)
    mlflow.tensorflow.autolog(every_n_iter=1, log_models=False)
    return active_run


def log_run(conf):
    # Log params
    params = {
        "_mixed_precision": conf.mixed_precision,
        "_seed": conf.seed,
        "_xla": conf.xla,
        "_batch_size": conf.batch_size,
        "_img_size": conf.img_size,
        "_first_fold_only": conf.first_fold_only,
        "_n_folds": conf.n_folds,
        "_base": conf.base,
        "_init_weights": conf.init_weights,
        "_unfreeze_layers": conf.unfreeze_layers,
        "_dropout": conf.dropout,
        "_optimizer": conf.optimizer,
        "_weight_decay": conf.weight_decay,
        "_lr": conf.lr,
        "_lr_start": conf.lr_start,
        "_lr_alpha": conf.lr_alpha,
        "_lr_decay_epochs": conf.lr_decay_epochs,
        "_lr_restart_epoch_multiplier": conf.lr_restart_epoch_multiplier,
        "_lr_restart_multiplier": conf.lr_restart_multiplier,
        "_lr_custom_boundaries": conf.lr_custom_boundaries,
        "_lr_custom_values": conf.lr_custom_values,
        "_lr_rampup_decay_factor": conf.lr_rampup_decay_factor,
        "_lr_rampup_epochs": conf.lr_rampup_epochs,
        "_lr_rampup_max": conf.lr_rampup_max,
        "_lr_rampup_min": conf.lr_rampup_min,
        "_lr_rampup_sustain_epochs": conf.lr_rampup_sustain_epochs,
        "_warmup_epochs": conf.warmup_epochs,
        "_warmup_lr": conf.warmup_lr,
        "_balanced_weights": conf.balanced_weights,
        "_early_stopping_epochs": conf.early_stopping_epochs,
        "_epochs": conf.epochs,
        "_label_smoothing": conf.label_smoothing,
        "_loss": conf.loss,
        "_metrics": conf.metrics,
        "_onehot": conf.onehot,
        "_train_blur": conf.train_blur,
        "_train_centercrop": conf.train_centercrop,
        "_train_clahe": conf.train_clahe,
        "_train_coarse_dropout": conf.train_coarse_dropout,
        "_train_crop": conf.train_crop,
        "_train_horizontal_flip": conf.train_horizontal_flip,
        "_train_hue_sat_value": conf.train_hue_sat_value,
        "_train_random_brightness": conf.train_random_brightness,
        "_train_random_contrast": conf.train_random_contrast,
        "_train_random_fog": conf.train_random_fog,
        "_train_random_grid_shuffle": conf.train_random_grid_shuffle,
        "_train_random_resized_crop": conf.train_random_resized_crop,
        "_train_rotate": conf.train_rotate,
        "_train_vertical_flip": conf.train_vertical_flip,
    }
    mlflow.log_params(params)


def end_run(history):
    # Log metrics
    metrics = {}
    for i, (key, value) in enumerate(history):
        if i == 0:
            metrics["_best_loss"] = min(value)
        elif i == 1:
            metrics["_best_metric"] = max(value)
        elif i == 2:
            metrics["_best_val_loss"] = min(value)
        elif i == 3:
            metrics["_best_val_metric"] = max(value)
    mlflow.log_metrics(metrics)
    mlflow.end_run()