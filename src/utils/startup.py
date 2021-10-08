import numpy as np
import os
import random
import tensorflow as tf
from tensorflow.keras import mixed_precision

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1' # Tensorflow 2.6.0 bug issues/51596


def allow_gpu_ram_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            LOG.info(str(len(gpus)) + " Physical GPUs - " + str(len(logical_gpus)) + " Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            LOG.info(e)


def set_mixed_precision():
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    LOG.info('Compute dtype: %s' % policy.compute_dtype)
    LOG.info('Variable dtype: %s' % policy.variable_dtype)


def enable_XLA(conf):
    os.environ['XLA_FLAGS'] = "--xla_gpu_cuda_data_dir='" + conf.path_to_cuda + "'"
    tf.config.optimizer.set_jit("autoclustering")
