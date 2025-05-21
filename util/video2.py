#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import cv2
from preprocess_gt.util_ip import *

#
#
#
def createVideo(filename, width, height, fps = 30.0):
    print([width, height])
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    return out

#
#
#
class Video:
    
    #
    #
    #
    def __init__(self, video_path, video_fmt = ' '):
    
        if video_fmt == ' ':
            video_fmt = os.path.splitext(video_path)[1]
            
        video_fmt_l = video_fmt.lower()
        self.bVideo = (video_fmt_l == '.mov') or (video_fmt_l == '.mp4') or (video_fmt_l == '.avi') or (video_fmt_l == '.asf')
        self.video_path = video_path
        self.video_fmt = video_fmt
        self.total_names = []
        self.v = []

        self.counter = 0

        if self.bVideo:
            self.v = cv2.VideoCapture(self.video_path)
            self.n = int(self.v.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            self.total_names = sorted([f for f in os.listdir(video_path) if f.endswith(video_fmt)])
            self.n = len(self.total_names)

    #
    #
    #
    def release(self):
        if self.v != []:
            self.v.release()

    #
    #
    #
    def getNumFrames(self):
        return self.n

    #
    #
    #
    def setFrame(self, frame):
                    
        if (frame > -1):
            frame = frame % self.n
            
            self.counter = frame
            
            if (self.v != []):
                self.v.set(cv2.CAP_PROP_POS_FRAMES, frame)

    #
    #
    #
    def getNextFrame(self, frame = -1, bBGR = True):
        print("A")
        self.setFrame(frame)
        
        counter = self.counter
    
        if self.bVideo: #from a video stream
            success, frame_cv = self.v.read()
            
            if success and bBGR:
                frame = np.zeros(frame_cv.shape, dtype = np.float32)
                frame[:,:,0] = frame_cv[:,:,2]
                frame[:,:,1] = frame_cv[:,:,1]
                frame[:,:,2] = frame_cv[:,:,0]
            else:
                frame = frame_cv
            self.counter = (self.counter + 1) % self.n

        else: #from a sequence of images in a folder
            path = os.path.join(self.video_path, self.total_names[self.counter])
            frame = readCV2(path, bBGR)
            self.counter = (self.counter + 1) % self.n
            success = True

        return success, frame, counter
