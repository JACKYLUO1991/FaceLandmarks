# Normalized Mean Error
# Created by Jacky LUO
# https://github.com/MarekKowalski/DeepAlignmentNetwork

import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.utils import custom_object_scope
import pandas as pd
import os
import cv2 as cv
from tqdm import tqdm
from scipy.integrate import simps

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

sys.path.append("../")
from loss import *
from model import relu6, hard_swish


class LandmarkNme(object):
    """Measure normalized mean error"""

    failure_threshold = 0.10

    def __init__(self, model_path, nb_points=106, output_dim=112):

        with custom_object_scope({'normalized_mean_error': normalized_mean_error,
                                  'wing_loss': wing_loss, 'smoothL1': smoothL1,
                                  'relu6': relu6, 'hard_swish': hard_swish}):
            self.model = load_model(model_path)

        self.output_dim = output_dim
        self.nb_points = nb_points

        self.__gt_landmarks = None
        self.__pred_landmarks = None
        self.__image_names = None

    @property
    def gt_landmarks(self):
        return self.__gt_landmarks

    @gt_landmarks.setter
    def gt_landmarks(self, landmarks_csv):
        '''Get Groundtruth landmarks'''
        df = pd.read_csv(landmarks_csv, header=None)
        self.__image_names = df.iloc[:, 0].values
        self.__gt_landmarks = df.iloc[:, 1:-
                                      3].values.reshape((-1, self.nb_points, 2)) 
    
    @property
    def pred_landmarks(self):
        return self.__pred_landmarks

    @pred_landmarks.setter
    def pred_landmarks(self, prefix):
        """Get pred landmarks"""
        marks_list = []
        for image_name in tqdm(self.__image_names):
            image_path = os.path.join(prefix, image_name)
            # Resize image to specific size like 112, 64...
            img = cv.resize(cv.imread(image_path),
                             (self.output_dim, self.output_dim))
            if self.output_dim == 64:
                img_normalized = img.astype(np.float32)
            else:
                img_normalized = img.astype(np.float32) / 255.
            face_img = img_normalized.reshape(
                1, self.output_dim, self.output_dim, 3)
            if self.output_dim == 64:
                marks = self.model.predict_on_batch(face_img)
            else:
                marks = self.model.predict_on_batch(face_img)[0]
            # marks = self.model.predict_on_batch(face_img)
            # marks = np.reshape(marks, (-1, 2))
            marks_list.append(marks)
            # print(marks)
        self.__pred_landmarks = np.array(
            marks_list, dtype=np.float32).reshape((-1, self.nb_points, 2))

    def landmark_error(self, normalization='centers'):
        """Get landmarks error between gt and pred"""
        errors = []
        n_imgs = len(self.__gt_landmarks)

        for i in tqdm(range(n_imgs)):
            gt_ldmarks = self.__gt_landmarks[i]
            pred_ldmarks = self.__pred_landmarks[i]

            if normalization == 'centers':
                normDist = np.linalg.norm(
                    gt_ldmarks[38] - gt_ldmarks[92])
            error = np.mean(np.sqrt(np.sum((gt_ldmarks -
                                            pred_ldmarks) ** 2, axis=1))) / normDist
            errors.append(error)

        return errors

    @classmethod
    def plot_ced(cls, errors_lists, step=0.0001, fontsize=18, labels=None, colors=None,
                 showCurve=True):
        '''Plot CED curve'''
        ced_list = []
        xAxis_list = []

        for errors in errors_lists:
            nErrors = len(errors)
            xAxis = list(np.arange(0., cls.failure_threshold + step, step))
            ced = [float(np.count_nonzero([errors <= x])) /
                   nErrors for x in xAxis]
            # AUC = simps(ced, x=xAxis) / cls.failure_threshold
            # failureRate = 1. - ced[-1]
            ced_list.append(ced)
            xAxis_list.append(xAxis)

        if showCurve:
            if labels is not None and colors is not None:
                plt.grid()
                plt.axis([0.0, cls.failure_threshold, 0, 1.0])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                for i in range(len(errors_lists)):
                    plt.plot(xAxis_list[i], ced_list[i], color=colors[i],
                             label=labels[i])
                plt.legend()
                plt.xlabel('Mean Normalized Error', fontsize=fontsize)
                plt.ylabel('Proportion of facial landmarks', fontsize=fontsize)
                plt.show()


if __name__ == "__main__":

    # Pipline
    errors_lists = []
    # PFLD network
    ln = LandmarkNme("../checkpoints/pfld.h5")
    ln.gt_landmarks = "../new_test_dataset/face_mixed.csv"
    ln.pred_landmarks = "../new_test_dataset"
    errors = ln.landmark_error()
    errors_lists.append(errors)

    # Mobilenetv3 network
    ln2 = LandmarkNme("../checkpoints/mobilenetv3.h5")
    ln2.gt_landmarks = "../new_test_dataset/face_mixed.csv"
    ln2.pred_landmarks = "../new_test_dataset"
    errors2 = ln2.landmark_error()
    errors_lists.append(errors2)

    # Basenet network
    ln3 = LandmarkNme("../checkpoints/model.h5", output_dim=64)
    ln3.gt_landmarks = "../new_test_dataset/face_mixed.csv"
    ln3.pred_landmarks = "../new_test_dataset"
    errors3 = ln3.landmark_error()
    errors_lists.append(errors3)

    # CED curve show
    LandmarkNme.plot_ced(errors_lists, showCurve=True, \
        labels=['Plfd', 'Mobilenetv3', 'Basenet'], colors=['blue', 'green', 'red'])
