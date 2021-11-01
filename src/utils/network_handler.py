from classification_models.tfkeras import Classifiers # qubvel
from tensorflow.keras import Input
from tensorflow.keras.applications import densenet, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import efficientnet, EfficientNetB0, EfficientNetB1, EfficientNetB2
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4, EfficientNetB5
from tensorflow.keras.applications import inception_resnet_v2, InceptionResNetV2
from tensorflow.keras.applications import inception_v3, InceptionV3
from tensorflow.keras.applications import mobilenet_v2, MobileNetV2
from tensorflow.keras.applications import resnet_v2, ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications import xception, Xception

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def get_base_model(conf):
    # Format inputs
    inputs = Input(shape=(*[conf.img_size_y, conf.img_size_x], conf.channels))

    # Retrieve base model
    base_model, x = define_base(conf, inputs)

    # Freeze base by default
    base_model.trainable = False
    nb_layers = len(base_model.layers)

    model = base_model(x, training=False)
    return model, inputs, nb_layers


def define_base(conf, inputs):
    # Retrieve base model
    if conf.base == "DenseNet121":
        x = densenet.preprocess_input(inputs)
        base_model = DenseNet121(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "DenseNet169":
        x = densenet.preprocess_input(inputs)
        base_model = DenseNet169(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "DenseNet201":
        x = densenet.preprocess_input(inputs)
        base_model = DenseNet201(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB0":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB0(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB1":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB1(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB2":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB3":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB3(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB4":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB4(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "EfficientNetB5":
        x = efficientnet.preprocess_input(inputs)
        base_model = EfficientNetB5(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "InceptionResNetV2":
        x = inception_resnet_v2.preprocess_input(inputs)
        base_model = InceptionResNetV2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "InceptionV3":
        x = inception_v3.preprocess_input(inputs)
        base_model = InceptionV3(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "MobileNetV2":
        x = mobilenet_v2.preprocess_input(inputs)
        base_model = MobileNetV2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "ResNet50V2":
        x = resnet_v2.preprocess_input(inputs)
        base_model = ResNet50V2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "ResNet101V2":
        x = resnet_v2.preprocess_input(inputs)
        base_model = ResNet101V2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "ResNet152V2":
        x = resnet_v2.preprocess_input(inputs)
        base_model = ResNet152V2(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "ResNeXT50":
        ResNeXT50, preprocess_input = Classifiers.get('resnext50')
        x = preprocess_input(inputs)
        base_model = ResNeXT50(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "ResNeXT101":
        ResNeXT101, preprocess_input = Classifiers.get('resnext101')
        x = preprocess_input(inputs)
        base_model = ResNeXT101(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "SeResNeXT50":
        SeResNeXT50, preprocess_input = Classifiers.get('seresnext50')
        x = preprocess_input(inputs)
        base_model = SeResNeXT50(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "SeResNeXT101":
        SeResNeXT101, preprocess_input = Classifiers.get('seresnext101')
        x = preprocess_input(inputs)
        base_model = SeResNeXT101(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "VGG16":
        x = vgg16.preprocess_input(inputs)
        base_model = vgg16.VGG16(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    elif conf.base == "Xception":
        x = xception.preprocess_input(inputs)
        base_model = Xception(
            input_shape=(*[conf.img_size_y, conf.img_size_x], conf.channels),
            include_top=conf.include_top,
            weights=conf.init_weights)
    else:
        LOG.error("Unknown network base, quitting")

    return base_model, x


def unfreeze(conf, model, nb_layers):
    # Unfreeze base if needed
    if conf.unfreeze_layers > 0:
        for i in range(nb_layers - conf.unfreeze_layers):
            model.layers[nb_layers - 1 - i].trainable = True
    elif conf.unfreeze_layers == -1:
        # Unfreeze all
        model.trainable = True

    return model
