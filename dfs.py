#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import time
import argparse
from shutil import copyfile

import torch
import numpy as np

from model_video import *
from dataloader import *
from util.util import *
from util.video import *
from util.util_ip import checkLaplaicanBluriness
from util.clean_video import cleanSequence

#
#
#
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Eval video regressor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='Path to the video or data dir to be tested')
    parser.add_argument('-c', '--copy', type=int, default=1, help='Copy (0 --> no copy; 1 --> copy')
    parser.add_argument('-r', '--removeBlurred', type=int, default=0, help='Copy (0 --> no blurred frame removal; 1 --> blurred frame removal')
    parser.add_argument('-f', '--format', type=str, default = '.jpg', help='If the video is a folder with images format is the image file format; e.g., .jpg')
    args = parser.parse_args()

    #results from the paper
    args.differential = 1
    args.samplescompute = 30
    args.runs = 'dfs_weights.pth'
    
    bRegular = (args.removeBlurred == 0)
        
    ext = os.path.splitext(args.data)[1]
    bVideo = True

    if (ext == ''):
        bVideo = False

        if(args.format == ''):
            sys.exit()
        else:
            ext = args.format
            
    res_size_x, res_size_y = getResolution()

    #CPU or CUDA?
    bCuda = torch.cuda.is_available()                   # do we have a CUDA GPU?
    device = torch.device("cuda" if bCuda else "cpu")   # use CPU or GPU

    print('Differential: ' + str(args.differential))
    print('Movie Ext:' + ext)
    model = ModelVideo(args.runs, device)
    
    print([res_size_x, res_size_y])

    #how to convert input frames
    transform = getTransform(res_size_x, res_size_y, args.data, args.differential, False)

    fps = 30
    vec = []
    bCopy = (args.copy >= 1)
    
    if bVideo:
        video_obj = Video(args.data, ext)
    else:
        video_obj = Video(args.data + '/data_pre', ext)
        print(args.data + '/data_pre')
        
    lst = cleanSequence(video_obj, folder_out = os.path.splitext(args.data)[0] + '_clean', bWrite = False)
        
    
    if bVideo:
        name_wo_ext = os.path.splitext(args.data)[0]
        lp = localPath(name_wo_ext)
        args_data = lp + '_' + ext
    else:
        args_data = args.data
        lp = localPath(args_data)
    
    fps = args.samplescompute

    n = len(lst)
    n = ((n // fps) * fps)
    
    folder_dfs = os.path.splitext(args.data)[0] + '_dfs'
    mkdir_s(folder_dfs)

    fid = open(os.path.join(folder_dfs, 'clean.txt'), 'w')
    for i in lst:
        fid.write(str(i) + '\n')
    fid.close()
    
    #
    #compute network
    #
    indices_out = []
    with open(os.path.join(folder_dfs, 'net_fq_est.txt'), 'w') as fq_pred_file:
        for i in range(0, n, fps):
            X = video_obj.readBlockFromVideo(i, transform, args.differential, fps, lst)

            y_out = model.predict(X)
    
            frames = np.round(y_out * fps)
            vec.append(frames)
            indices_out.append(i)

            print(lp + ' ' + str(lst[i]) + ' ' + str(frames))
            fq_pred_file.write(str(y_out) + '\n')
    
    if bCopy:
       output_dir = os.path.join(folder_dfs, "data_selected/")
       if not os.path.exists(output_dir):
          os.mkdir(output_dir)
    else:
        sys.exit()

    fid = open(os.path.join(folder_dfs, 'selected.txt'), 'w')
    
    #copy selected frames in data_selected folder
    for i in range(0, len(vec)):
        n_frames = vec[i]
        if n_frames > 0:
           sampling_factor = round(fps / n_frames)
           print([sampling_factor, fps, n_frames])
           
           if bRegular:
               for j in range(0, fps, sampling_factor):
                   index = lst[indices_out[i] + j]
                   
                   fid.write(str(index) + '\n')
                   
                   if bVideo:
                        success, frame = video_obj.getNextFrameWithIndex(index, False, False)
                                            
                        if success:
                            fn = 'frame_' + '{0:06d}'.format(index) + '.png';
                            fn_full = os.path.join(output_dir, fn)
                            cv2.imwrite(fn_full, frame * 255.0)
                   else:
                        name_index = video_obj.getName(index)
                        src = os.path.join(args_data  + '/data_pre/', name_index)
                        dst = os.path.join(output_dir, name_index)
                        copyfile(src, dst)
           else:
               #first pass
               lap_vec = []
               indices_i = []
               for j in range(0, fps):
                   index = lst[indices_out[i] + j]
                   success, frame = video_obj.getNextFrameWithIndex(index, False, False)
                   if success:
                       success, value = checkLaplaicanBluriness(frame, 0.0009, True)
                       lap_vec.append(value)
                       indices_i.append(index)
                
               #second pass
               indices = np.argsort(lap_vec)
                
               n = len(lap_vec)
               start = int(n - n_frames)
               for j in range(start, n):
                   index = indices_i[indices[j]]
                   fid.write(str(index) + '\n')
                   
                   success, frame = video_obj.getNextFrameWithIndex(index, False, False)
                   if success:
                       fn = 'frame_' + '{0:06d}'.format(index) + '.png';
                       fn_full = os.path.join(output_dir, fn)
                       cv2.imwrite(fn_full, frame * 255.0)
                
    fid.close()

