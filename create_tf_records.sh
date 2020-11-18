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

declare -A LABELS
xml_files=( $(ls tmp/object_detection/annotations/xmls/*.xml) )
NUM_INSTANCES=${#xml_files[@]}
INSTANCES=()

TRAIN_INSTANCES=()
TEST_INSTANCES=()
TRAIN_SIZE=$(echo "$NUM_INSTANCES*.8" | bc)
TRAIN_SIZE=${TRAIN_SIZE%.*}

class_id=1
for i in "${!xml_files[@]}"; do 
	filename=$(basename ${xml_files[$i]})
	instance=(${filename//.xml/ })
	class_label=$(echo $instance | sed -E "s/_[0-9]+//g")

	INSTANCES+=("$instance ${LABELS[$class_label]} 1 1")
	if ! test "${LABELS[$class_label]+isset}"
	then
		LABELS[$class_label]=$class_id
		echo $class_id, $class_label
		((class_id++))
	fi;

	if [ "$i" -le "$TRAIN_SIZE" ]
	then
		TRAIN_INSTANCES+=("$instance ${LABELS[$class_label]} 1 1")
	else
		TEST_INSTANCES+=("$instance ${LABELS[$class_label]} 1 1")
	fi
done

## write instances to list.txt
printf "%s\n" "${INSTANCES[@]}" > tmp/object_detection/annotations/list.txt
printf "%s\n" "${TRAIN_INSTANCES[@]}" > tmp/object_detection/annotations/trainval.txt
printf "%s\n" "${TEST_INSTANCES[@]}" > tmp/object_detection/annotations/test.txt

rm -f tmp/object_detection/annotations/$LABEL_MAP_FILE 
for K in "${!LABELS[@]}"; 
do 
cat << EOF >> tmp/object_detection/annotations/$LABEL_MAP_FILE

item {
    id: ${LABELS[$K]}
    name: "$K"
}

EOF
 
done

python $PROJECT_HOME/python/create_tf_record.py \
	--images_dir=tmp/object_detection/images \
	--label_map_path=tmp/object_detection/annotations/$LABEL_MAP_FILE \
	--annotations_dir=tmp/object_detection/annotations \
	--output_path=tmp/object_detection

cd tmp/object_detection
echo "Removing existing objects and bucket '$GCS_ML_BUCKET' from GCS..."
gsutil -m rm -r $GCS_ML_BUCKET/* | true
gsutil rb $GCS_ML_BUCKET | true # ignore the error if bucket does not exist
  
echo "Upload dataset to GCS..."
gsutil mb -l $REGION -c regional $GCS_ML_BUCKET
gsutil cp train.record $GCS_ML_BUCKET/data/train.record
gsutil cp val.record $GCS_ML_BUCKET/data/val.record
gsutil cp annotations/$LABEL_MAP_FILE $GCS_ML_BUCKET/data/$LABEL_MAP_FILE
