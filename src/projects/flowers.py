from tensorflow.keras import layers, Model

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def add_classif_head(conf, conf_proj, x, inputs):
    x = layers.GlobalAveragePooling2D()(x)
    if conf.dropout > 0:
        x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(conf_proj.n_classes)(x)
    predictions = layers.Activation('softmax', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model
