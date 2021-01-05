import cv2
import numpy as np
import dlib

import logging
import threading
import time
# pip install face_recognition  -i https://pypi.tuna.tsinghua.edu.cn/simple

'''安装openCV pip install opencv-contrib-python

安装cmake pip install cmake

安装boost pip install boost

安装dlib pip install dlib

安装face_recognition pip install face_recognition
'''

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../res/shape_predictor_68_face_landmarks.dat')

def FacePosition(img):
    """人脸定位"""
    dets = detector(img, 0)
    if not dets:
        return None
    return max(dets, key=lambda det: (det.right() - det.left()) * (det.bottom() - det.top()))

def GetDot(img, facePosition):
    """提取关键点"""
    landmark_shape = predictor(img, facePosition)
    dot = []
    for i in range(68):
        pos = landmark_shape.part(i)
        dot.append(np.array([pos.x, pos.y], dtype=np.float32))
    return dot

def MakeDot(dot):
    """生成构造点"""
    def Center(index):
        return sum([dot[i] for i in index]) / len(index)
    leftBrow = [18, 19, 20, 21]
    rightBrow = [22, 23, 24, 25]
    chew = [6, 7, 8, 9, 10]
    nose = [29, 30]
    return Center(leftBrow + rightBrow), Center(chew), Center(nose)

def MakeFeatures(dot):
    """生成特征"""
    brow, chew, nose = dot
    centerLine = brow - chew
    arrLine = brow - nose
    xSpin = np.cross(centerLine, arrLine) / np.linalg.norm(centerLine)**2
    ySpin = centerLine @ arrLine / np.linalg.norm(centerLine) ** 2
    return np.array([xSpin, ySpin])

def GetImgFeatures(img):
    """提取图片特征"""
    faceFacePosition = FacePosition(img)
    if not faceFacePosition:
        return None
    dot = GetDot(img, faceFacePosition)
    dotMake = MakeDot(dot)
    spin = MakeFeatures(dotMake)
    return spin

#def GetLoop():
    """捕捉循环"""
