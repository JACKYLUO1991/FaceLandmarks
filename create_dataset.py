"""
Create training dataset
"""
import cv2
import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd

from common.landmark_helper import LandmarkHelper
from common.landmark_utils import LandmarkImageCrop
import time
from pprint import pprint

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--landmark_txt", type=str, default='./test_dataset/landmarks.txt',
                help="path to landmarks txt")
ap.add_argument("-b", "--base_dir", type=str, default='./test_dataset',
                help="base dataset dir")
ap.add_argument("-s", "--output_size", type=int, default=64,
                help="output image size")
ap.add_argument("-n", "--new_path", type=str, default='./demo_test_dataset',
                help="new save image file")
args = vars(ap.parse_args())


def main():

    if not os.path.exists(args['new_path']):
        os.mkdir(args['new_path'])

    with open(args['landmark_txt']) as f:

        samples_list = []

        for line in f.readlines():
            # Parse txt file
            img_path, landmarks, poses = LandmarkHelper.parse(line)
            image_path = os.path.join(args['base_dir'], img_path)
            samples_list.append([image_path, landmarks, poses])

        boxes, ldmarks, poses = LandmarkImageCrop().mini_crop_by_landmarks(
            samples_list, scale=(1.2, 1.5), output_size=args['output_size'], is_vis=False)

        # Save image , new landmarks and poses
        mix_dict = dict()

        for box, ldmark, pose, num in tqdm(zip(boxes, ldmarks, poses, np.arange(len(samples_list)))):
            cv2.imwrite("{}.png".format(
                os.path.join(args['new_path'], str(num).zfill(5))), box)
            mix_dict["{}.png".format(str(num).zfill(5))] = np.concatenate(
                (ldmark, pose), axis=0)
            # print(np.concatenate((ldmark, pose), axis=0))

        df = pd.DataFrame(mix_dict).T
        df.to_csv("{}/face_mixed.csv".format(args['new_path']),
                  encoding="utf-8", header=None)

        pprint("Complete conversion!!!")


if __name__ == "__main__":
    main()
