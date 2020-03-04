import cv2
import numpy as np
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(
        description="This script divides video into frames", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--video_path", type=str, default=None,
                        help="target video path")
    args = parser.parse_args()
    return args


def save_all_frames(video_path, ext='jpg'):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    dir_path = os.path.splitext(video_path)[0]
    os.makedirs(dir_path, exist_ok=True)
    basename = "frame"
    base_path = os.path.join(dir_path, basename)

    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    print(digit)

    n = 0

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(
                base_path, str(n).zfill(digit), ext), frame)
            n += 1
            print("{} done\\".format(n))
        else:
            return


def main():
    args = get_args()
    _video_path = args.video_path
    save_all_frames(_video_path)


if __name__ == '__main__':
    main()
