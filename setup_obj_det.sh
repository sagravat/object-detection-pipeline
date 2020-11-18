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

TF_MODEL_DIR=tmp/tensorflow-models


##################################################
# Configuring TF research models
# Based on this tutorial: https://cloud.google.com/solutions/creating-object-detection-application-tensorflow
##################################################
setup_models() {
    echo "Setting up Proto and TensorFlow Models..."

	if [ -d "$TF_MODEL_DIR" ]; then
	    rm -rf $TF_MODEL_DIR
	fi
	mkdir -p $TF_MODEL_DIR
	cd $TF_MODEL_DIR

    ### Configure dev environment - pull down TF models
    git clone https://github.com/tensorflow/models.git
    cd models
    # checking out a commit we have verified that works
    git reset --hard 256b8ae622355ab13a2815af326387ba545d8d60
    cd ..

    PROTO_V=3.3
    PROTO_SUFFIX=0-linux-x86_64.zip

	if [ -d "protoc_${PROTO_V}" ]; then
	    rm -rf protoc_${PROTO_V}
	fi
    mkdir protoc_${PROTO_V}
    cd protoc_${PROTO_V}

    echo "Download PROTOC..."
    wget https://github.com/google/protobuf/releases/download/v${PROTO_V}.0/protoc-${PROTO_V}.${PROTO_SUFFIX}
    chmod 775 protoc-${PROTO_V}.${PROTO_SUFFIX}
    unzip protoc-${PROTO_V}.${PROTO_SUFFIX}
    rm -rf protoc-${PROTO_V}.${PROTO_SUFFIX}

    cd ..
    echo "Compiling protos..."
    cd models/research
    bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
    python setup.py sdist
    (cd slim && python setup.py sdist)
    echo $PWD

    PROTOC=../../protoc_${PROTO_V}/bin/protoc
    $PROTOC object_detection/protos/*.proto --python_out=.
}

setup_models
