"""
Detect faces and add landmarks based on them
"""
import numpy as np
import tensorflow as tf
import cv2
import os
from keras.models import load_model
from keras.utils import custom_object_scope
from loss import normalized_mean_error, wing_loss, smoothL1

import sys
sys.path.append("../")
from model import relu6, hard_swish
from detection import retinaface
# from detection import detect_face


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    # def __init__(self, threshold=[0.6, 0.7, 0.7], factor=0.709, minsize=20, mark_model=None):

    def __init__(self, model_path, gpuid=-1, thresh=0.95, scales=[384, 512], mark_model=None):

        self.gpuid = -1 if gpuid < 0 else gpuid
        self.thresh = thresh

        if isinstance(scales, (list, tuple)) and len(scales) == 2:
            self.scales = scales
        else:
            raise Exception("scales set is error...")

        try:
            self.detector = retinaface.RetinaFace(
                model_path, 0, self.gpuid, 'net3')
        except:
            raise Exception("Detector loading error...")

        # if isinstance(threshold, list) and len(threshold) == 3:
        #     self.threshold = threshold
        #     self.factor = factor
        #     self.minsize = minsize

            # with tf.Graph().as_default():
            #     sess = tf.Session()
            #     with sess.as_default():
            #         self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(
            #             sess, None)

        if mark_model.split(".")[-1] == 'h5':
            with custom_object_scope({'normalized_mean_error': normalized_mean_error,
                                      'wing_loss': wing_loss, 'smoothL1': smoothL1, \
                                          'relu6': relu6, 'hard_swish': hard_swish}):
                self.sess = load_model(mark_model)
        else:
            raise Exception("model should be given...")

        # else:
        #     raise ValueError("error occur in threshold params!")

    def detect_marks_keras(self, image_np):
        """Detect marks from image"""
        predictions = self.sess.predict_on_batch(image_np)

        # Convert predictions to landmarks.
        marks = np.array(predictions[0]).flatten()
        marks = np.reshape(marks, (-1, 2))

        return marks

    @staticmethod
    def move_box(box, offset, scale=1.1):
        """Move the box to direction specified by vector offset"""
        if scale < 1. or scale >= 2.:
            raise ValueError("scale should be between 1 and 2...")

        left_x = box[0] + offset[0] * scale
        top_y = box[1] + offset[1] * scale
        right_x = box[2] + offset[0] * scale
        bottom_y = box[3] + offset[1] * scale

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""

        left_x = int(box[0])
        top_y = int(box[1])
        right_x = int(box[2])
        bottom_y = int(box[3])

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff / 2))

        if diff == 0:                   # Already a square.
            return [left_x, top_y, right_x, bottom_y]
        elif diff > 0:                  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:                           # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        # Make sure box is always square.
        assert (right_x - left_x) == (bottom_y - top_y)

        return [left_x, top_y, right_x, bottom_y]

    def extract_cnn_facebox(self, image):
        """Extract face area from image."""
        faceboxes = []
        # scores = []
        face_ldmarks = []

        im_shape = image.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)

        # Prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        # bboxs, landmarks = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
        #                                            self.threshold, self.factor)

        faces, landmarks = self.detector.detect(
            image, self.thresh, scales=[im_scale])

        for box, ldmarks in zip(faces, landmarks):
            # Box: (x1, y1, x2, y2)
            diff_height_width = (box[2] - box[0]) - (box[3] - box[1])
            offset = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset])
            facebox = self.get_square_box(box_moved)
            ldmarks = ldmarks - (facebox[0], facebox[1])
            faceboxes.append(facebox)
            face_ldmarks.append(ldmarks)

        return faceboxes, face_ldmarks

        # if bboxs.shape[0] == 0:
        #     landmarks_reshape = landmarks
        # else:
        #     landmarks_reshape = landmarks.reshape((-1, 5, 2), order='F')

        # for bbox, ldmarks in zip(bboxs, landmarks_reshape):
        #     box, score = bbox[0: 4], bbox[4]
        #     # Move down
        #     # box coordinate: (x1, y1, x2, y2)
        #     diff_height_width = (box[2] - box[0]) - (box[3] - box[1])
        #     offset = int(abs(diff_height_width / 2))
        #     box_moved = self.move_box(box, [0, offset])
        #     # Make box square and landmarks alignment
        #     facebox = self.get_square_box(box_moved)
        #     ldmarks = ldmarks - (facebox[0], facebox[1])
        #     faceboxes.append(facebox)
        #     face_ldmarks.append(ldmarks)
        #     scores.append(score)

        # return faceboxes, face_ldmarks, scores

    @staticmethod
    def draw_marks(image, marks, color=(255, 0, 255), thick=1):
        """Draw mark points on image"""
        for idx, mark in enumerate(marks):
            cv2.circle(image, (int(mark[0]), int(mark[1])),
                       thick, color, -1, cv2.LINE_AA)
        # Visualization cropped image
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
