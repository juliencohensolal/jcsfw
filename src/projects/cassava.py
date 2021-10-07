from tensorflow.keras import layers, Model

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def preprocessing(train, verbose=True):
    LOG.info("Cassava preprocessing")

    # Add image path to dataframe
    train["imagepath"] = "../_data/Cassava/train_images/" + train["image_id"]

    if verbose:
        LOG.info(train.head(2))

    return train


def get_dataset_names(df):
    return "imagepath", "label"


def add_classif_head(conf, conf_proj, x, inputs):
    x = layers.GlobalAveragePooling2D()(x)
    if conf.dropout > 0:
        x = layers.Dropout(conf.dropout)(x)
    x = layers.Dense(conf_proj.n_classes)(x)
    predictions = layers.Activation('softmax', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model
