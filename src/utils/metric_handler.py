import utils.c_logging as c_logging

from tensorflow.keras import metrics

LOG = c_logging.getLogger(__name__)


def get_metrics(conf):
    metrics_ = None
    if conf.metrics == "Accuracy":
        metrics_ = metrics.Accuracy()
    elif conf.metrics == "CategoricalAccuracy":
        metrics_ = metrics.CategoricalAccuracy()
    elif conf.metrics == "CategoricalCrossentropy":
        metrics_ = metrics.CategoricalCrossentropy(from_logits=False)
    elif conf.metrics == "SparseCategoricalAccuracy":
        metrics_ = metrics.SparseCategoricalAccuracy()
    elif conf.metrics == "SparseCategoricalCrossentropy":
        metrics_ = metrics.SparseCategoricalCrossentropy(from_logits=False)
    else:
        LOG.error("Unknown metric, quitting")
    return metrics_
