import argparse
import json
import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import time
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

"""
Input: label_pathとimage_path
Output: bboxをトリミングしたもの
"""

def get_args():
  parser = argparse.ArgumentParser(
    description="This script inputs the bbox coordinate and outputs trimmed image",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
    "--dir_path", 
    type=str,
    #default='/home/intern1/library/models/research/object_detection/test_images/counter_0223',
    default='/home/intern1/library/models/research/object_detection/result/hamada_hideaki',
    help="set target image directory path")
  args = parser.parse_args()
  return args

def create_path_list(target_path, label_or_image):
    if label_or_image == "label":
        target_paths = sorted(list(target_path.glob("*.txt")))
    elif label_or_image == "image":
        target_paths = sorted(list(target_path.glob("*.jpg")))
    else:
        return(Error)
    return target_paths

def check_correspondence(label_paths, image_paths):
    #check whether the image and label direcotry has the same name of files
    label_basenames = [os.path.splitext(os.path.basename(str(label_path)))[0] for label_path in label_paths]
    image_basenames = [os.path.splitext(os.path.basename(str(image_path)))[0] for image_path in image_paths]
    return  label_basenames == image_basenames

# def hogehoge():
    # labelにあったjpgを拾ってくる
    # txtからbboxの座標を取る
    # トリミングする
    # 画像をアウトプットする

def main():
    args = get_args()
    dir_path = args.dir_path
    print("\ndir_path: {}".format(dir_path))

    LABEL_PATH = pathlib.Path(os.path.join(dir_path, "labels"))
    IMAGE_PATH = pathlib.Path(os.path.join(dir_path, "images"))
    if not os.path.exists(str(LABEL_PATH) or str(IMAGE_PATH)):
        return(Error)
    LABEL_PATHS = create_path_list(LABEL_PATH, "label") 
    IMAGE_PATHS = create_path_list(IMAGE_PATH, "image")
    if check_correspondence(LABEL_PATHS, IMAGE_PATHS):
        for label_path in LABEL_PATHS:
            f = open(str(label_path))
            lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            f.close()
            # lines2: リスト。要素は1行の文字列データ
            coordinates = []
            for line in lines2:
                coordinates.append(line.split()[1:])
            print(coordinates)
            #該当するjpgファイルを見つけて、以下coordinateの情報を元にトリミング処理をする
    else:
        return(Error("Basenames don't correspond. Checkout whether the labels and images directory has the same basenames."))

    # print(label_base_names)
    # print(image_base_names)

if __name__ == '__main__':
    main()