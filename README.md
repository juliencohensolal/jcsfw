# JCSFW : JCS Framework

## About

This is a personal project to allow for quick protoyping and iteration on **image classification problems**. It is still very much a *WORK IN PROGRESS*.

## Features

- Experiment configuration via YAML files
- Logging traces of each execution into a text file
- Tensorboard binding
- MLFlow experiment tracking
- Various classification architectures available, including EfficientNet & SeResNeXT
- Various compilers available, including RectifiedAdam & LAMB
- Various augmentations available, mostly (but not only) relying on albumentations package
- Various learning rates available, including CosineDecayRestarts & PiecewiseConstantDecay
- RampUp & WarmUp options for learning rate
- Mixed-precision training & XLA compilation for faster training
- Possibility to freeze a custom number of layers
- Label smoothing

## Usage

There are currently 2 scripts available:
- *train_classif.py*: to train a classification model
- *predict_classif.py*: to perform inference using a previously trained classification model

The *cfg_experiment.yml* file contains all of the experiment settings you want to se before running an experiment


**TO BE CONTINUED**