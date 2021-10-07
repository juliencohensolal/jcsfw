import utils.c_logging as c_logging

from tensorflow.keras import optimizers, mixed_precision

LOG = c_logging.getLogger(__name__)


def get_optimizer(conf, is_warmup=False):
    if is_warmup:
        lr = conf.warmup_lr
    else:
        lr = conf.lr_start

    optimizer = None
    if conf.optimizer == "Adam": # = RMSprop with momentum
        optimizer = optimizers.Adam(learning_rate=lr)
    elif conf.optimizer == "Nadam": # = Adam with Nesterov momentum
        optimizer = optimizers.Nadam(learning_rate=lr)
    else:
        LOG.error("Unknown optimizer, quitting")
    
    # Set mixed precision for custom training loops
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    return optimizer
