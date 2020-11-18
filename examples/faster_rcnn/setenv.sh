#!/bin/bash

#
# Copyright 2018 Google LLC. This software is provided as-is, without warranty 
# or representation for any use or purpose. Your use of it is subject to your 
# agreements with Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

###########################################################
# Shared environment variables for Tensorflow Object Detection 
###########################################################


set -a # assign bash variables to environment variables

### How many training steps to take
TRAINING_STEPS=4000

### TF Runtime version for CMLE
TF_RUNTIME_VERSION=1.9

### Which model checkpoint and config to use for training. Choose from
### the model zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md#coco-trained-models
MODEL=faster_rcnn_resnet101_coco_2018_01_28

### Model config name (match this with the model name)
### choose from https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
MODEL_CONFIG=faster_rcnn_resnet101_coco.config

### Local installation directory for tensorflow models and object detection.
### Installed from https://github.com/tensorflow/models/
PROJECT_HOME=`pwd`/../..
BASE_TMP=`pwd`/tmp
TF_MODULE_PATH=$BASE_TMP/tensorflow-models

### Location to copy the model config file 
MODEL_CONFIG_PATH=object_detection/config

### Label map file
LABEL_MAP_FILE=my_label_map.pbtxt

### Bucket location of images and xml files
# set the value here
GCS_IMAGES_DIR=
REGION=us-central1
IMAGES_ZIP=images-for-training.zip
ANNOTATIONS_ZIP=annotations.zip

### GCS Bucket to store initial model checkpoint, config, and tfrecords
### the initial files get written to $GCS_ML_BUCKET/data and the checkpoints
### are written to $GCS_ML_BUCKET/train during training
GCS_ML_BUCKET=

### Set the PYTHONPATH
export PYTHONPATH=.:$TF_MODULE_PATH/models/research:$TF_MODULE_PATH/models/research/slim:$TF_MODULE_PATH/models/research/object_detection

set +a # stop exporting bash variables to environment variables

