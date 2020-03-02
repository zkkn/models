#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
tf.config.gpu.set_per_process_memory_growth(True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


def get_args():
  parser = argparse.ArgumentParser(
    description="This script outputs the object detection inference result in image and coordinates",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--image_dir_path", 
    type=str,
    default='/home/intern1/library/models/research/object_detection/test_images/counter_0221',
    help="set target image directory path")
  parser.add_argument(
    "--model_name",
    type=str,
    default='mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28',
    help="set object detection model. google tensorflow object detection model zoo.")
  parser.add_argument(
    "--labels_path",
    type=str,
    default='/home/intern1/library/models/research/object_detection/data/mscoco_label_map.pbtxt',
    help="set label path")
  args = parser.parse_args()
  return args

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  model = tf.saved_model.load(str(model_dir))
  model = model.signatures['serving_default']

  return model


def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  output_dict = model(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

def show_inference(model, image_path, result_image_dir_path, category_index):
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = np.array(Image.open(image_path))
    # Actual detection.
    output_dict = run_inference_for_single_image(model, image_np)

    print(output_dict['detection_boxes'])
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        max_boxes_to_draw=50)
        #min_score_thresh=0.2)
    
    coordinates = vis_util.return_coordinates(
                        image_np,
                        output_dict['detection_boxes'],
                        output_dict['detection_classes'],
                        # (output_dict['detection_classes']).astype(np.int32),
                        output_dict['detection_scores'],
                        category_index,
                        instance_masks=output_dict.get('detection_masks_reframed', None),
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=50)
                        #min_score_thresh=0.2)

    # output bbox image
    image_basename = os.path.basename(str(image_path))
    images_output_dir = os.path.join(result_image_dir_path, "images")
    if not os.path.exists(images_output_dir):
        os.mkdir(images_output_dir)
    Image.fromarray(image_np).save(os.path.join(images_output_dir, image_basename))

    # output coordinate txt
    # 座標の出力形式を以下のようにするような処理
    #(class id) (bboxの中心のx座標) (bboxの中心のy座標) (bboxの横幅) (bboxの縦幅)
    # だからあえてリストを空白で区切った

    label_basename = os.path.splitext(image_basename)[0]+".txt"
    labels_output_dir = os.path.join(result_image_dir_path, "labels")
    if not os.path.exists(labels_output_dir):
        os.mkdir(labels_output_dir)
    f = open(os.path.join(labels_output_dir, label_basename), 'w')
    for x in coordinates:
        f.write(' '.join(map(str, x)) + "\n")
    f.close()


def main():
    args = get_args()
    image_dir_path = pathlib.Path(args.image_dir_path)
    model_name = args.model_name
    labels_path = args.labels_path
    TEST_IMAGE_PATHS = sorted(list(image_dir_path.glob("*.jpg")))
    print(TEST_IMAGE_PATHS)
    # model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
    detection_model = load_model(model_name)
    #print(detection_model.inputs)
    #detection_model.output_dtypes
    #detection_model.output_shapes

    # List of the strings that is used to add correct label for each box.
    _category_index = label_map_util.create_category_index_from_labelmap(labels_path, use_display_name=True)

    PATH_TO_OBJECT_DETECTION_DIR = '/home/intern1/library/models/research/object_detection/'
    _result_image_dir_path = os.path.join(PATH_TO_OBJECT_DETECTION_DIR, "result", os.path.basename(str(image_dir_path)))
    if not os.path.exists(_result_image_dir_path):
        # os.mkdir(os.path.join(PATH_TO_OBJECT_DETECTION_DIR, "result"))
        os.mkdir(_result_image_dir_path)

    for _image_path in TEST_IMAGE_PATHS:
      show_inference(detection_model, _image_path, _result_image_dir_path, _category_index)

if __name__ == '__main__':
    main()
