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

r"""
python object_detection/dataset_tools/rectlabel_create_pascal_tf_record.py \
    --images_dir="${IMAGES_DIR}" \
    --label_map_path="${LABEL_MAP_PATH}" \
    --output_path="${OUTPUT_PATH}" \
    --include_masks
"""
import hashlib
import io
import os
import glob
import random
import logging

from lxml import etree
import numpy as np
np.set_printoptions(threshold=np.nan)
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_boolean('include_masks', False, 'Add image/object/mask to TFRecord using png images in annotations folder')
flags.DEFINE_string('images_dir', '', 'Full path to the images directory.')
flags.DEFINE_string('annotations_dir', 'annotations', '(Relative) path to annotations directory.')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
FLAGS = flags.FLAGS


def getClassId(name, label_map_dict):
    class_id = -1
    for item_name, item_id in label_map_dict.items():
        if name in item_name:
            class_id = item_id
            break
    if class_id < 0:
        raise ValueError(name + ' not found in the label map file')
    return class_id

def dict_to_tf_example(data, annotations_dir, images_dir, label_map_dict, include_masks, ignore_difficult_instances):
    image_path = os.path.join(images_dir, data['filename'])
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')
    key = hashlib.sha256(encoded_jpg).hexdigest()
    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    masks = []
    if 'object' in data:
        for idx, obj in enumerate(data['object']):
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue
            difficult_obj.append(int(difficult))
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            class_id = getClassId(obj['name'], label_map_dict)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(class_id)
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

            if include_masks:
                mask_path = os.path.join(annotations_dir, "masks", os.path.splitext(data['filename'])[0] + '_' + str(idx) + '.png')
                with tf.gfile.GFile(mask_path, 'rb') as fid:
                    encoded_mask_png = fid.read()
                encoded_png_io = io.BytesIO(encoded_mask_png)
                mask = PIL.Image.open(encoded_png_io)
                if mask.format != 'PNG':
                    raise ValueError('Mask format not PNG')
                mask_np = np.asarray(mask)
                mask_remapped = (mask_np == 255).astype(np.uint8)
                masks.append(mask_remapped)
    feature_dict = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(
            data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }

    if include_masks:
        encoded_mask_png_list = []
        for mask in masks:
            img = PIL.Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (dataset_util.bytes_list_feature(encoded_mask_png_list))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

def create_tf_record(output_filename,
                     label_map_dict,
                     annotations_dir,
                     images_dir,
                     examples,
                     include_masks=True):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes.  Otherwise
      generates bounding boxes (as well as segmentations).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
      xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      #mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')
      annotation_path = os.path.join(annotations_dir, 'xmls', example + '.xml')

      if not os.path.exists(annotation_path):
          print('Could not find %s, ignoring example.', annotation_path)
          continue
      with tf.gfile.GFile(annotation_path, 'r') as fid:
          xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
      tf_example = dict_to_tf_example(data, annotations_dir, images_dir, label_map_dict, include_masks, FLAGS.ignore_difficult_instances)
      writer.write(tf_example.SerializeToString())
  writer.close()

def main(_):
    images_dir = FLAGS.images_dir
    annotations_dir = os.path.join(FLAGS.annotations_dir) 
    print("include masks", FLAGS.include_masks)

    if FLAGS.include_masks:
        train_output_path = os.path.join(FLAGS.output_path,
                                                 'train_with_masks.record')
        val_output_path = os.path.join(FLAGS.output_path,
                                                 'val_with_masks.record')
    else:
        train_output_path = os.path.join(FLAGS.output_path,
                                                 'train.record')
        val_output_path = os.path.join(FLAGS.output_path,
                                                 'val.record')

    writer = tf.python_io.TFRecordWriter(train_output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    # SBA
    examples_path = os.path.join("tmp/object_detection/annotations", 'trainval.txt')
    examples_list = dataset_util.read_examples_list(examples_path)

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))
    ## END


    #os.chdir(images_dir)
    #file_types = ('*.jpg', '*.jpeg')
    #image_files = []
    #for file_type in file_types:
        #image_files.extend(glob.glob(file_type))
    #print("image_file: %s", image_files)
    create_tf_record(
      train_output_path,
      label_map_dict,
      annotations_dir,
      FLAGS.images_dir,
      train_examples,
      FLAGS.include_masks)
    create_tf_record(
      val_output_path,
      label_map_dict,
      annotations_dir,
      FLAGS.images_dir,
      val_examples,
      FLAGS.include_masks)



if __name__ == '__main__':
    tf.app.run()
