"""
For static image landmarks detection
"""
import numpy as np
import cv2
import time
import glob
# import imutils
import sys
import os
sys.path.append("../")
# from transformer import FaceAlign
from detection.detector import MarkDetector

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

CNN_INPUT_SIZE = 112
# face_align = FaceAlign(out_size=CNN_INPUT_SIZE)


def main(images_dir, savePath):

    if not os.path.isdir(savePath):
        os.makedirs(savePath)

    images_path = os.listdir(images_dir)

    # mark_detector = MarkDetector(threshold=[0.7, 0.6, 0.95],
    #                              mark_model='../checkpoints/pfld.h5')
    mark_detector = MarkDetector(model_path='../detection/model/mnet.25',
                                 gpuid=-1, thresh=0.9, scales=[224, 384],
                                     mark_model='../checkpoints/pfld.h5')

    for idx, image_path in enumerate(images_path):

        img = cv2.imread(os.path.join(images_dir, image_path))
        # img = imutils.resize(img, width=512)
        h, w, _ = img.shape
        img_copy = img.copy()
        # img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        s1 = time.time()
        faceboxes, face_ldmarks = mark_detector.extract_cnn_facebox(
            img_copy)
        print(f"Detection: {time.time() - s1}s")
        pad_type = -1

        if len(faceboxes) == 0:
            print("Not detected face...")
            continue

        else:
            for facebox, ldmarks in zip(faceboxes, face_ldmarks):

                facebox = list(map(int, facebox))
                x_min, y_min, x_max, y_max = facebox[0], facebox[1], facebox[2], facebox[3]

                if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
                    pad_type = 1
                    absTmp = np.minimum(
                        (x_min, y_min, w - x_max, h - y_max), 0)
                    pad = np.max(np.abs(absTmp))
                    # Entire image op
                    img = cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    y_min, y_max, x_min, x_max = y_min + pad, y_max + pad, x_min + pad, x_max + pad

                face_img_crop = img[y_min: y_max, x_min: x_max]
                face_img_align_uint = cv2.resize(
                    face_img_crop, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                # ldmarks = [(x / (face_img_crop.shape[0] / CNN_INPUT_SIZE), y /
                #             (face_img_crop.shape[1] / CNN_INPUT_SIZE)) for (x, y) in ldmarks]
                #####################################################################
                # 5 points alignment code
                # face_img_align_uint, tform = face_align.face_aligned(
                #     face_img_align_uint, ldmarks)
                #####################################################################
                if CNN_INPUT_SIZE == 64:
                    face_img_align = face_img_align_uint.astype(np.float32)
                else:
                    face_img_align = face_img_align_uint.astype(np.float32) / 255.
                face_img0 = face_img_align.reshape(
                    1, CNN_INPUT_SIZE, CNN_INPUT_SIZE, 3)

                s2 = time.time()
                marks = mark_detector.detect_marks_keras(face_img0)
                print(f"Landmarks: {time.time() - s2}s")
                #############################################################################
                # Inverse similarity transformation Matrix
                # marks *= CNN_INPUT_SIZE
                # b = np.ones((marks.shape[0], 1))
                # d = np.concatenate((marks, b), axis=1)
                # M = cv2.invertAffineTransform(tform.params[:2, :])
                # marks = np.dot(d, M.T)
                # marks /= CNN_INPUT_SIZE

                marks *= (x_max - x_min)
                #############################################################################
                marks[:, 0] += x_min
                marks[:, 1] += y_min

                # Draw Predicted Landmarks
                MarkDetector.draw_marks(img, marks, thick=2)

                if pad_type == 1:
                    pad_type = -1
                    img = img[pad: pad+h, pad: pad+w]

        print("[INFO] Finished {} pictures".format(idx+1))

        cv2.imwrite(os.path.join(savePath, "result_%d.jpg" % (idx)), img)


if __name__ == "__main__":
    main(images_dir='./images', savePath='./result112')
