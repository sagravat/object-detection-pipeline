# Tensorflow Object Detection on Google Cloud with Deep Learning VM Images
These scripts demonstrate a pipeline to automate object detection based on
the [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

## Pre-requisites
Either use the [Deep Learning VM](https://cloud.google.com/deep-learning-vm/) on GCP or manually install Tensorflow with a GPU.

The naming convention of the .jpg and .xml files should be 
`<classlabel>_<numericalindex>.[jpg|xml]` and each class should be stored in a separate
subfolder. The classlabel can also have an underscore in the name.

The images and xml annotation files should be uploaded in GCS in
separate .zip files. Mask files should also be included in a zip file if applicable.

Modify the `setenv.sh` file to set the GCS bucket locations and model name for 
object detection.

## Installation
Run the following to authenticate to gcloud:

`$ gcloud auth application-default login`

Install necessary packages for performing calculation in unix shell:

`$ sudo apt-get install bc`

## Setup Object Detection with custom dataset
Go to one of the example directories run the configuration script, for example:

`$ cd examples/faster_rcnn`

### Edit the setenv.sh file  
update the GCS_IMAGES_DIR and GCS_ML_BUCKET and any other variables as needed.
The GCS_IMAGES_DIR is the bucket that contains the custom images and annotations.
The GCS_ML_BUCKET is the bucket used for the training job
`GCS_IMAGES_DIR=gs://my_images`
`GCS_ML_BUCKET=gs://my_ml_training`

### Run the setup script
`$ ./configure_faster_rcnn_resnet_101.sh`

## Local training
Run local training:

`./local_training.sh`


