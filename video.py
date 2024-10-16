#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import numpy as np
import cv2
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import threading

from util import *

#
#
#
class Video:
    def __init__(self, video_path, video_fmt = ''):
        
        if video_fmt == '':
            video_fmt = os.path.splitext(video_path)[1]
        video_fmt_c = video_fmt.lower()
        
        self.bVideo = (video_fmt_c == '.mov') or (video_fmt_c == '.mp4') or (video_fmt_c == '.avi') or (video_fmt_c == '.asf')
        self.video_path = video_path
        self.video_fmt = video_fmt
        self.total_names = []
        self.v = []
        self.lock = threading.Lock()

        self.counter = 0

        if self.bVideo:
            self.v = cv2.VideoCapture(self.video_path)
            self.n = int(self.v.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_names = sorted([f for f in os.listdir(video_path) if f.endswith(video_fmt)])
            self.n = len(self.total_names)
            
    def release(self):
        self.v.release()

    def getNumFrames(self):
        return self.n

    def setFrame(self, frame):
        self.counter = frame % self.n
        self.v.set(cv2.CAP_PROP_POS_FRAMES, self.counter)

    def getNextFrame(self, bBGR=True):
        if self.bVideo:
            success, frame_cv = self.v.read()
            if success:
                frame = fromNPtoPIL(fromVideoFrameToNP(frame_cv, bBGR))
            else:
                frame = []
        else:
            path = os.path.join(self.video_path, self.total_names[self.counter])
            frame = imread(path)
            self.counter = (self.counter + 1) % self.n
            success = True

        return success, frame
        
    #
    #
    #
    def getNextFrameWithIndex(self, index, bBGR=True, bPIL = True):
        index = index % self.n
        
        if self.bVideo:
            self.lock.acquire(blocking=True)
            self.setFrame(index)
            success, frame_cv = self.v.read()
            self.lock.release()
            
            if success:
                frame = fromVideoFrameToNP(frame_cv, bBGR)
                if bPIL:
                    frame = fromNPtoPIL(frame)
            else:
                frame = []
        else:
            path = os.path.join(self.video_path, self.total_names[index])
            frame = Image.open(path)
            if not bPIL:
                frame = fromPILtoNP(frame)
            success = True

        return success, frame
        
    #
    #
    #
    def readBlockFromVideo(self, index, use_transform, differential, fps = 30):
        X = []
        for i in range(0, fps):
            success, frame = self.getNextFrameWithIndex(index + i)
            if success:
                if use_transform is not None:
                    frame = use_transform(frame)
                else:
                    frame = to_tensor(frame)
                X.append(frame)
 
        if(differential == 1):
            for i in range(0, fps - 1):
                X[i] = X[i + 1] - X[i]
            X = torch.stack(X[0:(fps - 1)], dim=0)
        else:
            X = torch.stack(X[0:fps], dim=0)
            
        return X

    def getName(self, index):
        if self.bVideo:
            return 'frame_' + str(index)
        else:
            return self.total_names[index]
