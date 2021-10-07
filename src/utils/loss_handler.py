import utils.c_logging as c_logging

from tensorflow.keras import losses

LOG = c_logging.getLogger(__name__)


def get_loss(conf):
    loss = None
    if conf.loss == "SparseCategoricalCrossentropy":
        loss = losses.SparseCategoricalCrossentropy(from_logits=False)
    elif conf.loss == "CategoricalCrossentropy":
        loss = losses.CategoricalCrossentropy(from_logits=False)
    else:
        LOG.error("Unknown loss, quitting")
    return loss
