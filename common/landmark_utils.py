import numpy as np
import cv2
import os
from tqdm import tqdm
np.random.seed(2018)


class LandmarkImageCrop(object):
    '''
    Facial 106 landmarks augmentation.
    '''

    def __init__(self):
        pass

    def __visualize(self, image, landmarks, output_size):
        '''
        Visualize images and corresponding landmarks
        '''
        try:
            image.shape
        except:
            raise ValueError("read image error...")

        for (x, y) in landmarks:
            cv2.circle(image, (int(x * output_size), int(y * output_size)),
                       1, (0, 0, 255), -1)

        cv2.imshow("image", image)
        cv2.waitKey(0)

    def mini_crop_by_landmarks(self, sample_list, scale, output_size=112, is_vis=False):
        '''
        Crop full image to mini. Only keep vaild image to save
        Args:
            sample_list: (image, landmarks)
            scale: up scale rate
            value: color value
            output_size: output image size
        Returns:
            new sample list
        '''

        boxes = np.empty((len(sample_list), output_size,
                          output_size, 3), dtype=np.uint8)
        ldmarks = np.empty((len(sample_list), 212))
        poses = np.empty((len(sample_list), 3))

        for idx, sample in tqdm(enumerate(sample_list)):
            image = cv2.imread(sample[0])
            landmarks = sample[1]
            pose = sample[2]
            try:
                (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h) = LandmarkImageCrop.get_bbox_of_landmarks(
                    image, landmarks, scale, 0.5)
            except:
                print(sample[0])
            # Extract roi image
            box_image = image[y1:y2, x1:x2]
            if need_pad:
                box_image = np.lib.pad(
                    box_image, ((p_y, p_h), (p_x, p_w), (0, 0)), 'constant')
            box_image = cv2.resize(box_image, (output_size, output_size))
            landmarks = (landmarks - (x1 - p_x, y1 - p_y)) / \
                (new_size, new_size)
            if is_vis:
                self.__visualize(box_image, landmarks, output_size)
            # Convert to (212,)
            landmarks = landmarks.flatten()

            boxes[idx] = box_image
            ldmarks[idx] = landmarks
            poses[idx] = pose

        return boxes, ldmarks, poses

    @staticmethod
    def get_bbox_of_landmarks(image, landmarks, scale, shift_rate=0.3):
        '''
        According to landmark box to generate a new bigger bbox
        Args:
            image: a numpy type
            landmarks: face landmarks with format [(x1, y1), ...]. range is 0-w or h in int
            scale: scale bbox in (min, max). eg: (1.3, 1.5)
            shift_rate: up, down, left, right to shift
        Returns:
            return new bbox and other info
        Raises:
            No
        '''
        ori_h, ori_w = image.shape[:2]

        x = int(min(landmarks[:, 0]))
        y = int(min(landmarks[:, 1]))
        w = int(max(landmarks[:, 0]) - x)
        h = int(max(landmarks[:, 1]) - y)
        if type(scale) == float:
            scale = scale
        else:
            scale = np.random.randint(
                int(scale[0] * 100.0), int(scale[1] * 100.0)) / 100.0
        new_size = int(max(w, h) * scale)
        if shift_rate >= 0.5:
            x1 = x - (new_size - w) // 2
            y1 = y - (new_size - h) // 2
        else:
            x1 = x - np.random.randint(int((new_size-w) * shift_rate),
                                       int((new_size-w) * (1.0-shift_rate)))
            y1 = y - np.random.randint(int((new_size-h) * shift_rate),
                                       int((new_size-h) * (1.0-shift_rate)))
        x2 = x1 + new_size
        y2 = y1 + new_size
        need_pad = False
        p_x, p_y, p_w, p_h = 0, 0, 0, 0
        if x1 < 0:
            p_x = -x1
            x1 = 0
            need_pad = True
        if y1 < 0:
            p_y = -y1
            y1 = 0
            need_pad = True
        if x2 > ori_w:
            p_w = x2 - ori_w
            x2 = ori_w
            need_pad = True
        if y2 > ori_h:
            p_h = y2 - ori_h
            y2 = ori_h
            need_pad = True

        return (x1, y1, x2, y2), new_size, need_pad, (p_x, p_y, p_w, p_h)
