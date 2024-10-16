#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from util import *
from sklearn.model_selection import train_test_split
import sys
import pandas as pd

#
#
#
def split_data(fq_vec, group=1, fps = 30):
    
    if group > 1:
        index_v = []
        fq_v = []
        group_v = []
        for i in range(0, len(fq_vec)):
            fq_i = fq_vec[i]
            index = int(i * fps)
        
            for j in range(0, group):
                fq_v.append(fq_i)
                index_v.append(index)
                group_v.append(j)
    
        d = {'Fq': fq_v, 'Index': index_v, 'Group': group_v}
        data = pd.DataFrame(data=d)
        data = [data[i:i + group] for i in range(0, len(data), group)]
    else:
        index_v = []
        fq_v = []
        for i in range(0, len(fq_vec)):
            fq_i = fq_vec[i]
            index = int(i * fps)

            fq_v.append(fq_i)
            index_v.append(index)
    
        d = {'Fq': fq_v, 'Index': index_v}
        data = pd.DataFrame(data=d)

    train, val = train_test_split(data, test_size=0.2, random_state=42)

    if (group > 1):
        train = pd.concat(train)
        val = pd.concat(val)

    return train, val

#
#
#
def ReadFQFile(fn):
    fq_vec = []
    with open(fn) as f:
        for line in f: # read rest of lines
            value = [float(x) for x in line.split()]
            fq_vec.append(value[0])
    return fq_vec
    
#
#
#
def ReadImageFileNames(data):
    frames_names_tmp = [f for f in os.listdir(data) if f.endswith('.jpg')]
    
    for i in range(0, len(frames_names_tmp)):
        frames_names_tmp[i] = os.path.join(data, frames_names_tmp[i])
    
    frames_names_tmp = sorted(frames_names_tmp)
    return frames_names_tmp

#
#
#
def ReadDataset(data, group = 1, method = 'our', fps = 30):
    video_folders = os.listdir(data)
    video_folders = sorted(video_folders)
    
    fq_vec = []
    img_vec =[]
    for i in range(0, len(video_folders)):
        v = video_folders[i]
    
        if(os.path.isfile(os.path.join(data, v))):
            continue

        if(v.startswith(".")):
            continue
            
        path_fq_v = os.path.join(os.path.join(data, v), 'fq_' + method + '.txt')
        if not os.path.exists(path_fq_v):
            continue
            
        fq_vec_v = ReadFQFile(path_fq_v)
        
        max_frames = int(fps * len(fq_vec_v))
        
        fq_vec += fq_vec_v
        
        path_img_fn = os.path.join(data, os.path.join(v, 'data_pre'))
        img_vec_v = ReadImageFileNames(path_img_fn)
        print(v + " - Usable Max Frames: " + str(max_frames) + " Frames: " + str(len(img_vec_v)))
        img_vec_v = img_vec_v[0:max_frames]

        img_vec += img_vec_v
    return fq_vec, img_vec

#
#
#
class DatasetModelVideo(data.Dataset):

    #
    #
    #
    def __init__(self, data, img_vec, fps = 30, transform = None, group = 1, differential = 0):
        self.differential = differential
        self.fps = int(fps)
        self.transform = transform
        self.data = data
        self.img_vec = img_vec
        self.group = group

    #
    #
    #
    def __len__(self):
        return len(self.data)

    #
    #
    #
    def read_images(self, index):

        sample = self.data.iloc[index]

        #
        j = int(sample.Index)
        X = [] 
        for i in range(0, self.fps):
            name = self.img_vec[i + j]
            image = Image.open(name)

            if self.group > 1:
                image = dataAugmentation(image, sample.Group % self.group)
            else:
                image = dataAugmentation(image, 0)

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = to_tensor(image)

            X.append(image)

        #differential?
        if(self.differential == 1):
            for i in range(0, self.fps - 1):
                X[i] = X[i + 1] - X[i]
            X = torch.stack(X[0:(self.fps - 1)], dim=0)
        else:
            X = torch.stack(X[0:self.fps], dim=0)
        
        y = torch.from_numpy(np.array(sample.Fq))
        y = y.to(torch.float32)
        
        return X, y

    #
    #
    #
    def __getitem__(self, index):
        X, y = self.read_images(index)      

        return X, y
