import argparse
import cv2
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
    "--base_dir", 
    type=str,
    default = '/home/intern1/Asset/arashi_trim',
    help="set target image directory path")
  args = parser.parse_args()
  return args


def create_path_list(target_path, label_or_image):
    if label_or_image == "label":
        target_path = pathlib.Path(target_path)
        target_paths = sorted(list(target_path.glob("*.txt")))
    elif label_or_image == "image":
        #TODO: jpgにしか現在対応してないので、拡張子対応を増やす
        target_path = pathlib.Path(target_path)
        target_paths = sorted(list(target_path.glob("*.jpg")))
    else:
        return(Error)
    return target_paths

def check_correspondence(label_paths, image_paths):
    #check whether the image and label direcotry has the same name of files
    label_basenames = [os.path.splitext(os.path.basename(str(label_path)))[0] for label_path in label_paths]
    image_basenames = [os.path.splitext(os.path.basename(str(image_path)))[0] for image_path in image_paths]
    return label_basenames == image_basenames

def main():
    args = get_args()
    BASE_DIR = args.base_dir
    LABEL_DIR = os.path.join(BASE_DIR, "result", "labels")
    IMAGE_DIR = os.path.join(BASE_DIR, "src")
    TRIM_DIR = os.path.join(BASE_DIR, 'result', 'trim_img')

    print("\nimage_dir: {}".format(IMAGE_DIR))

    if not os.path.exists(str(LABEL_DIR) or str(IMAGE_DIR)):
        return(Error)
    LABEL_DIR_PATH_LIST = create_path_list(LABEL_DIR, "label") 
    IMAGE_DIR_PATH_LIST = create_path_list(IMAGE_DIR, "image")
    if check_correspondence(LABEL_DIR_PATH_LIST, IMAGE_DIR_PATH_LIST):
        for label_path in LABEL_DIR_PATH_LIST:
            f = open(str(label_path))
            lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            f.close()
            # lines2: リスト。要素は1行の文字列データ
            # (bboxの中心のx座標) (bboxの中心のy座標) (bboxの横幅) (bboxの縦幅)
            coordinates = []
            for line in lines2:
                coordinates_before_convert = [int(num) for num in line.split()[1:]]
                coordinates.append(coordinates_before_convert)
            image_coresspond_path = os.path.join(str(IMAGE_DIR), os.path.basename(str(label_path)).split('.')[0])+".jpg"
            #該当するjpgファイルを見つけて、以下coordinateの情報を元にトリミング処理をする
            im = cv2.imread(image_coresspond_path)
            for i, (x,y,w,h) in enumerate(coordinates):
                x -= int(w/2)
                y -= int(h/2)
                dst = im[y:y+h, x:x+w]
                if not os.path.exists(os.path.join(TRIM_DIR)):
                    os.mkdir(os.path.join(TRIM_DIR))
                cv2.imwrite(os.path.join(TRIM_DIR, os.path.basename(str(label_path)).split('.')[0]+"_"+str(i)+".jpg"),dst)
    else:
        return(Error("Basenames don't correspond. Checkout whether the labels and images directory has the same basenames."))

if __name__ == '__main__':
    main()