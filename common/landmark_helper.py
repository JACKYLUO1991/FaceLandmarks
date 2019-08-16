# coding=utf-8
'''
Bob.Liu in 20171114
'''
import numpy as np
import cv2
import sys


class LandmarkHelper(object):
    '''
    Helper for different landmark type
    '''

    def __init__(self):
        pass

    @classmethod
    def parse(cls, line):
        '''
        use for parse txt line to get file path and landmarks and so on
        Args:
            cls: this class
            line: line of input txt
            landmark_type: len of landmarks
        Return:
            see child parse
        Raises:
            unsupport type
        '''
        return cls.__landmark106_txt_parse(line)

    @staticmethod
    def __landmark106_txt_parse(line):
        '''
        [1] image path
        [2:5] bounding box
        [6:] 106 landmarks
        '''
        a = line.split()
        landmarks = list(map(int, a[1:-3]))[4:]
        pose = list(map(float, a[1:]))[-3:]

        return a[0], np.array(landmarks).reshape((-1, 2)), np.array(pose, dtype=np.float32)

    @staticmethod
    def flip(a):
        '''
        use for flip landmarks. Because we have to renumber it after flip
        Args:
            a: original landmarks
            landmark_type: type of landmarks(106)
        Returns:
            landmarks: new landmarks
        Raises:
            unsupport type
        '''
        landmarks = np.concatenate((
            a[0:1], a[17:33], a[1:17],
            a[87:89], a[93:94], a[91:92], a[90:91], a[92:93], a[89:90],
            a[94:95], a[96:97], a[95:96], a[99:102][::-
                                                    1], a[97:99], a[104:106][::-1], a[102:104],
            a[61:62], a[53:54], a[57:60], a[54:57], a[60:61], a[52:
                                                                53], a[62:63], a[67:71], a[63:67], a[71:72],
            a[72:75], a[81:86], a[80:81], a[75:80], a[86:87],
            a[33:35], a[39:40], a[37:38], a[36:37], a[38:39], a[35:36],
            a[40:41], a[42:43], a[41:42], a[46:47], a[47:48], a[45:46], a[44:45], a[43:44],
            a[50:51], a[51:52], a[49:50], a[48:49]
        ), axis=0)

        return landmarks.reshape((-1, 2))
