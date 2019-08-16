# This algorithm is limited to algorithm verification

import argparse
import cv2
import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
from skimage import transform
from pprint import pprint

from mtcnn.mtcnn import MTCNN

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

# from common.landmark_utils import LandmarkImageCrop
# from common.landmark_helper import LandmarkHelper

ap = argparse.ArgumentParser()
ap.add_argument("-l", "--landmark_txt", type=str, default='./new_dataset/landmarks.txt',
                help="path to landmarks txt")
ap.add_argument("-c", "--landmark_csv", type=str, default='./new_dataset/face_landmarks.csv',
                help="exist landmarks csv")
ap.add_argument("-b", "--base_dir", type=str, default='./new_dataset',
                help="base dataset dir")
ap.add_argument("-s", "--output_size", type=int, default=112,
                help="output image size")
ap.add_argument("-n", "--new_path", type=str, default='./align_new_dataset',
                help="new save image file")
args = vars(ap.parse_args())


REFERENCE_FACIAL_POINTS = [[38.453125, 28.139446],
                           [70.8962, 27.549734],
                           [54.171013, 50.283226]]


# def scale_and_shift(image, landmarks, scale_range, output_size):
#     '''
#     Auto generate bbox and then random to scale and shift it.
#     Args:
#         image: a numpy type
#         landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
#         scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
#         output_size: output size of image
#     Returns:
#         an image and landmarks will be returned
#     Raises:
#         No
#     '''
#     (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h) = LandmarkImageCrop.get_bbox_of_landmarks(
#         image, landmarks, scale_range, shift_rate=0.3)
#     box_image = image[y1:y2, x1:x2]
#     if need_pad:
#         box_image = np.lib.pad(
#             box_image, ((p_y, p_h), (p_x, p_w), (0, 0)), 'constant')
#     box_image = cv2.resize(box_image, (output_size, output_size))
#     landmarks = (landmarks - (x1 - p_x, y1 - p_y))

#     return box_image, landmarks

class FaceAlign(object):
    '''Align face with MTCNN'''

    def __init__(self, out_size):
        self.detector = MTCNN()
        self.out_size = out_size

    def face_aligned_mtcnn(self, im):
        '''
        Function: Alignment with MTCNN Prior box
        im: BGR image array
        '''
        try:
            wrapper = self.detector.detect_faces(im[:, :, ::-1])[0]
        except:
            raise ValueError("No face...")

        points = wrapper['keypoints']
        values = list(points.values())
        gt_array = np.array(values).reshape((-1, 2))[:2]
        ref_array = np.array(REFERENCE_FACIAL_POINTS[:2], dtype=np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(gt_array, ref_array)
        tfm = tform.params[0: 2, :]

        return cv2.warpAffine(
            im, tfm, (self.out_size, self.out_size))

    def face_aligned(self, im, ldmarks):
        '''
        im: BGR array
        ldmarks: [(x0, y0), ...]
        '''
        gt_array = np.array(ldmarks)[:2]
        ref_array = np.array(REFERENCE_FACIAL_POINTS[:2], dtype=np.float32)

        tform = transform.SimilarityTransform()
        tform.estimate(gt_array, ref_array)
        tfm = tform.params[0: 2, :]

        return cv2.warpAffine(
            im, tfm, (self.out_size, self.out_size)), tform


if __name__ == '__main__':

    # with open('./dataset/landmarks.txt') as f:

    #     samples_list = []

    #     for line in f.readlines():
    #         # Parse txt file
    #         img_path, landmarks = LandmarkHelper.parse(line)
    #         image_path = os.path.join("./dataset", img_path)

    #         im = cv2.imread(image_path)
    #         image, landmarks = scale_and_shift(
    #             im, landmarks, scale_range=(1.1, 1.5), output_size=112)

    #         cv2.imshow("image", image)
    #         cv2.waitKey(0)

    if not os.path.exists(args['new_path']):
        os.mkdir(args['new_path'])

    root_dir = args['base_dir']
    df = pd.read_csv(args['landmark_csv'], header=None)

    ldmarks = np.array(df.iloc[:, 1:])
    ldmarks = ldmarks.reshape((-1, 106, 2)) * \
        (args['output_size'], args['output_size'])

    ref_leftpupil = np.mean(ldmarks[:, 34], axis=0)
    ref_rightpupil = np.mean(ldmarks[:, 92], axis=0)
    ref_nose = np.mean(ldmarks[:, 86], axis=0)
    ref_array = np.stack(
        [ref_leftpupil, ref_rightpupil, ref_nose], axis=0).astype(np.float32)

    boxes = np.empty(
        (df.shape[0], args['output_size'], args['output_size'], 3), dtype=np.uint8)
    landmarks = np.empty((df.shape[0], 212))

    for idx in tqdm(range(df.shape[0])):

        im = cv2.imread(os.path.join(root_dir, df.iloc[idx, 0]))
        im = cv2.resize(im, (args['output_size'], args['output_size']))
        gt_ldmarks = ldmarks[idx]

        gt = np.array(df.iloc[idx, 1:], dtype=np.float32).reshape(
            (-1, 2)) * (args['output_size'], args['output_size'])
        gt_leftpupil = gt[34]
        gt_rightpupil = gt[92]
        gt_nose = gt[86]
        gt_array = np.stack(
            [gt_leftpupil, gt_rightpupil, gt_nose], axis=0).astype(np.float32)

        # M = cv2.getAffineTransform(gt_array, ref_array)
        # Similar transformation
        tform = transform.SimilarityTransform()
        tform.estimate(gt_array, ref_array)
        tfm = tform.params[0: 2, :]
        dst = cv2.warpAffine(
            im, tfm, (args['output_size'], args['output_size']))

        b = np.ones((gt_ldmarks.shape[0], 1))
        d = np.concatenate((gt_ldmarks, b), axis=1)
        gt_ldmarks = np.dot(d, np.transpose(tfm))

        boxes[idx] = dst
        landmarks[idx] = (gt_ldmarks / (args['output_size'])).flatten()

        # for ldmark in gt_ldmarks:
        #     cv2.circle(
        #         dst, (int(ldmark[0]), int(ldmark[1])), 2, (255, 0, 0), -1)
        # cv2.imshow("image", dst)
        # cv2.waitKey(0)

    # Save image and new landmarks
    ldmark_dict = dict()

    for box, ldmark, num in tqdm(zip(boxes, landmarks, np.arange(df.shape[0]))):
        cv2.imwrite("{}.png".format(
            os.path.join(args['new_path'], str(num).zfill(5))), box)
        ldmark_dict["{}.png".format(str(num).zfill(5))] = ldmark

    df = pd.DataFrame(ldmark_dict).T
    df.to_csv("{}/face_landmarks.csv".format(args['new_path']),
              encoding="utf-8", header=None)

    pprint("Complete conversion!!!")
