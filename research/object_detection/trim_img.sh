#!/bin/bash
BASE_DIR='/home/intern1/Asset/arashi_turningup'

#python object_detection_demo.py --base_dir_path $BASE_DIR
python trim_image_bbox.py --base_dir $BASE_DIR
