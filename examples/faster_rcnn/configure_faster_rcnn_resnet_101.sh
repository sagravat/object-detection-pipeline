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

source setenv.sh
set -eu

$PROJECT_HOME/check_env_vars.sh
$PROJECT_HOME/reset_buckets.sh
$PROJECT_HOME/setup_obj_det.sh
$PROJECT_HOME/download_dataset.sh
$PROJECT_HOME/create_tf_records.sh  
$PROJECT_HOME/upload_model.sh  

