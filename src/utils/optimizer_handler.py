import utils.c_logging as c_logging

from tensorflow.keras import optimizers, mixed_precision
import tensorflow_addons as tfa

LOG = c_logging.getLogger(__name__)


def get_optimizer(conf, is_warmup=False):
    if is_warmup:
        lr = conf.warmup_lr
    else:
        lr = conf.lr_start

    optimizer = None
    if conf.optimizer == "Adam": # = RMSprop with momentum
        optimizer = optimizers.Adam(learning_rate=lr)
    elif conf.optimizer == "AdamW": # = Adam with weight decay
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=conf.weight_decay)
    elif conf.optimizer == "LAMB": # = Layer-wise Adaptive Moments
        optimizer = tfa.optimizers.LAMB(learning_rate=lr, weight_decay_rate=conf.weight_decay)
    elif conf.optimizer == "Nadam": # = Adam with Nesterov momentum
        optimizer = optimizers.Nadam(learning_rate=lr)
    elif conf.optimizer == "RectifiedAdam": # = Adam with rectified adaptive learning rate to have a consistent variance
        optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr, weight_decay=conf.weight_decay)
    elif conf.optimizer == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=lr)
    elif conf.optimizer == "SGD":
        optimizer = optimizers.SGD(learning_rate=lr)
    else:
        LOG.error("Unknown optimizer, quitting")
    
    # Set mixed precision for custom training loops
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer
