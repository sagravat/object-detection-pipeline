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

rm -rf tmp/object_detection
mkdir -p tmp/object_detection
mkdir -p tmp/object_detection/images
mkdir -p tmp/object_detection/annotations/xmls

cd tmp/object_detection
gsutil cp $GCS_IMAGES_DIR/$IMAGES_ZIP .
gsutil cp $GCS_IMAGES_DIR/$ANNOTATIONS_ZIP .
unzip -q -j $ANNOTATIONS_ZIP -d annotations/xmls
unzip -q -j $IMAGES_ZIP -d images

rm $ANNOTATIONS_ZIP
rm $IMAGES_ZIP
cd ../..

