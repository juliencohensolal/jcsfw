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
        #"lr_custom_boundaries": conf.lr_custom_boundaries,
        #"lr_custom_values": conf.lr_custom_values,
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
        "_train_central_crop": conf.train_central_crop,
        "_train_cutmix_proba": conf.train_cutmix_proba,
        "_train_mixup_proba": conf.train_mixup_proba,
        "_train_random_brightness": conf.train_random_brightness,
        #"train_random_contrast": conf.train_random_contrast,
        "_train_random_crop": conf.train_random_crop,
        "_train_random_flip_left_right": conf.train_random_flip_left_right,
        "_train_random_flip_up_down": conf.train_random_flip_up_down,
        "_train_random_hue": conf.train_random_hue,
        #"train_random_saturation": conf.train_random_saturation,
        "_train_rot90": conf.train_rot90,
    }
    mlflow.log_params(params)


def end_run():
    mlflow.end_run()