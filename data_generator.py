# https: // stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

from keras.utils import Sequence
import numpy as np
import pandas as pd
import sys
import os
import cv2
from pprint import pprint

from common.landmark_utils import LandmarkImageCrop
from common.landmark_helper import LandmarkHelper


class DataGenerator(Sequence):
    '''
    Generates data for Keras
    '''

    def __init__(self, batch_size, root_dir, csv_file, output_size=112,
                 shuffle=False, max_angle=45, transformer=None):

        self.landmarks_frame = pd.read_csv(csv_file, header=None)
        self.batch_size = batch_size
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.max_angle = max_angle

        assert isinstance(output_size, int)
        self.output_size = output_size
        self.transformer = transformer
        self.on_epoch_end()

    def __getitem__(self, index):
        '''Generate one batch of data'''

        indexes = self.indexes[index *
                               self.batch_size: (index+1) * self.batch_size]
        list_frames = self.landmarks_frame.iloc[indexes, :]
        X, y_ld, y_p = self.__data_generation(list_frames)

        if self.transformer:

            X_imgs = np.empty(
                (self.batch_size, self.output_size, self.output_size, 3), dtype=np.uint8)
            y_ldmarks = np.empty((self.batch_size, 212), dtype=np.float32)
            y_poses = np.empty((self.batch_size, 3), dtype=np.float32)

            for idx, img, ldmark, pose in zip(np.arange(len(indexes)), X, y_ld, y_p):
                ldmark = ldmark.reshape((-1, 2)) * \
                    (self.output_size, self.output_size)

                # Data Augmentationm you can custom parameter
                img, ldmark, pose = self.__flip(img, ldmark, pose)
                img, ldmark, pose = self.__rotate(
                    img, ldmark, pose, max_angle=self.max_angle)

                # Do not need to modified pose...
                img, ldmark, pose = self.__scale_and_shift(
                    img, ldmark, pose, (1.1, 1.5), output_size=self.output_size)
                img, ldmark, pose = self.__occlusion(img, ldmark, pose)

                X_imgs[idx] = img
                y_ldmarks[idx] = (ldmark / (self.output_size,
                                            self.output_size)).flatten()
                y_poses[idx] = pose
            # Image normalization
            return X_imgs.astype(np.float32) / 255., [y_ldmarks, y_poses]

        return X.astype(np.float32) / 255., [y_ld, y_p]

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.landmarks_frame) / self.batch_size))

    def on_epoch_end(self):
        '''Updates indexes after each epoch'''
        self.indexes = np.arange(len(self.landmarks_frame))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @property
    def data_predict(self):
        '''Predict a batch size data'''
        return self.__data_generation()

    def __data_generation(self, list_frames):
        '''Producing batches of data'''

        X_imgs = np.empty((self.batch_size, self.output_size,
                           self.output_size, 3), dtype=np.uint8)
        y_ldmarks = np.empty((self.batch_size, 212), dtype=np.float32)
        y_poses = np.empty((self.batch_size, 3), dtype=np.float32)

        for i in range(len(list_frames)):
            image_path = os.path.join(
                self.root_dir, list_frames.iloc[i, 0])
            X_imgs[i] = cv2.imread(image_path)
            y_ldmarks[i] = list_frames.iloc[i, 1:-3]
            y_poses[i] = list_frames.iloc[i, -3:]

        return X_imgs, y_ldmarks, y_poses

    def __flip(self, image, landmarks, poses, run_prob=0.5):
        '''
        Do image flip. Only for horizontal

        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), (x2, y2), ...]
            run_prob: probability to do this operate. 0.0-1.0
        Returns:
            an image and landmarks will be returned
        Raises:
            Unsupport count of landmarks
        '''
        if np.random.rand() < run_prob:
            return image, landmarks, poses
        image = np.fliplr(image)
        landmarks[:, 0] = image.shape[1] - landmarks[:, 0]
        landmarks = LandmarkHelper.flip(landmarks)

        # pitch, roll, yaw...
        poses[1] = -poses[1]
        poses[2] = -poses[2]

        return image, landmarks, poses

    def __rotate(self, image, landmarks, poses, max_angle, run_prob=0.5):
        '''
        Do image rotate.

        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            max_angle: random to rotate in [-max_angle, max_angle]. range is 0-180.
        Returns:
            an image and landmarks will be returned
        Raises:
            No
        '''
        if np.random.rand() < run_prob:
            return image, landmarks, poses

        c_x = (min(landmarks[:, 0]) + max(landmarks[:, 0])) / 2
        c_y = (min(landmarks[:, 1]) + max(landmarks[:, 1])) / 2
        h, w = image.shape[:2]
        angle = np.random.randint(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((c_x, c_y), angle, 1)
        image = cv2.warpAffine(image, M, (w, h))
        b = np.ones((landmarks.shape[0], 1))
        d = np.concatenate((landmarks, b), axis=1)
        landmarks = np.dot(d, np.transpose(M))

        # Adjustment roll angle
        poses[1] += angle

        return image, landmarks, poses

    def __occlusion(self, image, landmarks, poses, sl=0.05, sh=0.2, r1=0.3, mean=[0, 0, 0], run_prob=0.2):
        '''
        Do image part occlusion

        sl: min erasing area
        sh: max erasing area
        r1: min aspect ratio
        mean: erasing value
        https://github.com/zhunzhong07/Random-Erasing/blob/master/transforms.py
        '''

        if np.random.rand() < run_prob:
            return image, landmarks, poses

        for attempt in range(50):
            area = image.shape[0] * image.shape[1]
            target_area = np.random.uniform(sl, sh) * area
            aspect_ratio = np.random.uniform(r1, 1/r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < image.shape[1] and h < image.shape[0]:
                x1 = np.random.randint(0, image.shape[0] - h)
                y1 = np.random.randint(0, image.shape[1] - w)
                image[x1: x1+h, y1: y1+w, 0] = mean[0]
                image[x1: x1+h, y1: y1+w, 1] = mean[1]
                image[x1: x1+h, y1: y1+w, 2] = mean[2]

                return image, landmarks, poses

        return image, landmarks, poses

    def __scale_and_shift(self, image, landmarks, poses, scale_range, output_size, run_prob=0.5):
        '''
        Auto generate bbox and then random to scale and shift it.

        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale_range: scale bbox in (min, max). eg: (1.3, 1.5)
            output_size: output size of image
        Returns:
            an image and landmarks will be returned
        Raises:
            No
        '''
        if np.random.rand() < run_prob:
            return image, landmarks, poses

        (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h) = LandmarkImageCrop.get_bbox_of_landmarks(
            image, landmarks, scale_range, shift_rate=0)
        box_image = image[y1:y2, x1:x2]
        if need_pad:
            box_image = np.lib.pad(
                box_image, ((p_y, p_h), (p_x, p_w), (0, 0)), 'constant')
        box_image = cv2.resize(box_image, (output_size, output_size))
        landmarks = (landmarks - (x1 - p_x, y1 - p_y))

        return box_image, landmarks, poses
