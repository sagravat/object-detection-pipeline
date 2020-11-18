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

set -eu
source mask_rcnn_resnet101_atrous_coco.sh

$PROJECT_HOME/check_env_vars.sh

echo $TF_MODULE_PATH
echo $MODEL_CONFIG
gcloud ml-engine local train \
    --job-dir=$GCS_ML_BUCKET/train \
    --package-path $TF_MODULE_PATH/models/research/object_detection \
    --module-name object_detection.legacy.train \
    -- \
    --train_dir=$GCS_ML_BUCKET/train \
    --pipeline_config_path=$GCS_ML_BUCKET/data/$MODEL_CONFIG

