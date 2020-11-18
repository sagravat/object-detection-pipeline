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


cd tmp
rm -rf ${MODEL}/
mkdir -p object_detection/config
NUM_CLASSES=$(grep item object_detection/annotations/$LABEL_MAP_FILE | wc -l)
echo $NUM_CLASSES
echo "Upload pretrained COCO Model for Transfer Learning..."
wget https://storage.googleapis.com/download.tensorflow.org/models/object_detection/${MODEL}.tar.gz
wget -O object_detection/config/$MODEL_CONFIG https://github.com/tensorflow/models/raw/master/research/object_detection/samples/configs/${MODEL_CONFIG}  
cd object_detection/config
sed -i "s|PATH_TO_BE_CONFIGURED|"${GCS_ML_BUCKET}"/data|g" $MODEL_CONFIG
sed -i -E "s/mscoco_train.record-[^\s]{5}-of-[^\s]{5}/train.record/g" $MODEL_CONFIG
sed -i -E "s/mscoco_val.record-[^\s]{5}-of-[^\s]{5}/val.record/g" $MODEL_CONFIG
sed -i -E "s/num_classes: ([0-9]+)/num_classes: $NUM_CLASSES/g" $MODEL_CONFIG
sed -i "s|mscoco_label_map.pbtxt|${LABEL_MAP_FILE}|g" $MODEL_CONFIG
cd ../..
	    
tar -xf ${MODEL}.tar.gz
echo "Copy model checkpoints to GCS..."
gsutil cp $MODEL/model.ckpt.* $GCS_ML_BUCKET/data/

rm -rf ${MODEL}.tar.gz
echo "Copy modified model config to GCS..."
echo "PWD $PWD"
gsutil cp $MODEL_CONFIG_PATH/$MODEL_CONFIG  $GCS_ML_BUCKET/data/$MODEL_CONFIG

