#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys

from util.util_ip import *
from util.video import *

#
#
#
def cleanSequence(video_obj, shift_threshold = 16, bBGR = False, bWrite = True, folder_out = 'images', name_base = 'frame'):

    if bWrite:
        mkdir_s(folder_out)

    c = 0
    n = video_obj.n
    print('Total frames: ' + str(n))
   
    lst = []

    if shift_threshold < 0:
        shape_max = np.max(img.shape)
        shift_threshold = np.max([16, np.round(shape_max / 40.0)])

    print('Threshold: ' + str(shift_threshold))
    
    j = 0
    
    success, img, k = video_obj.getNextFrame(bPIL = False)
    
    while(j < (n - 1)):
        lap = False
        while (lap == False):
            success, img_n, k = video_obj.getNextFrame(bPIL = False)
            if (success == False) or (k >= (n - 1)):
                j = n + 1
                break
                
            lap, value = checkKeyPointBluriness(img_n)
            
            if lap == False:
                j += 1
                print('Removed ' + format(j, '06d'))

        if success:
            removed_str = ' removed '
            bTest1, ssim = checkSimilarity(img, img_n, 0.925, bBGR)
            tmp = " "
            
            if(bTest1 == False):
        
                bTest2, shift = checkMTB(img, img_n, shift_threshold, bBGR)
                tmp += "Shift: " + str(shift) + " "

                if(bTest2 == False):
                    img = img_n
                    j = k
                    lst.append(k)
                    removed_str = ' kept '
                    
                    if bWrite:
                        writeCV2(img_n / 255.0, folder_out + '/'+ name_base+'_' + format(j,'06d') + '.png')

            print('Ref: ' + str(j) + ' Cur: ' + str(k) + removed_str + " SSIM: " + str(ssim) + tmp)

    return lst
    
#
#
#
def processOneVideo(name_video, folder_out = [], sampling = -1, iTarget = -1):
    
    if folder_out == []:
        folder_out = os.path.join(os.path.dirname(name_video), 'images')
    
    print('Processing video ' + name_video)

    name = os.path.splitext(name_video)[0]
    name_base = os.path.basename(name_video)
    extension = os.path.splitext(name_base)[1]
    name_base = os.path.splitext(name_base)[0]
    
    
    v = Video(name_video, extension)

    mkdir_s(folder_out)
        
    if sampling == -1:
        bSave = (iTarget == -1)
        lst = cleanSequence(v, 16, False, bSave, folder_out, name_base)

        if bSave == False:
            n = len(lst)
            print(n)
            
            index = []
            if iTarget > n:
                index = np.linspace(0, n - 1, n)
                index = index.astype(int)
            else:
                index = np.round(np.linspace(0, n - 1, iTarget))
                index = index.astype(int)
                index = np.unique(index)
            index = index.tolist()
            
            for i in index:
                j = lst[i]
                success, frame, j_k = v.getNextFrame(j, True)
                if success:
                    writeCV2(frame / 255.0, folder_out + '/' + name_base + '_' + format(j,'06d') + '.png')
    else:
        n = v.getNumFrames()
        for i in range(0, (n - sampling), sampling):
            success, frame, i_k = v.getNextFrame(i, True)
            counter = 0
            frame /= 255.0

            if iTarget == 1:
                for j in range(1, sampling):
                    success_j, frame_j, j_k = v.getNextFrame(i + j, True)
                    if success_j:
                        frame += (frame_j / 255.0)
                        counter += 1
                    
                if counter > 1:
                    frame /= float(counter)
                    frame = np.clip(frame, 0.0, 1.0)
        
            if success:
                writeCV2(frame, folder_out + '/' + name_base + '_' + format(i,'06d') + '.png')
        
    v.release()
    
#
#
#
if __name__ == "__main__":
    folder = sys.argv[1]
    
    if len(sys.argv) > 2:
        sampling = int(sys.argv[2])
    else:
        sampling = -1

    if len(sys.argv) > 3:
        target = int(sys.argv[3])
    else:
        target = -1

    list = ['.mp4', '.mov', '.asf']
    if os.path.splitext(folder)[1].lower() in list:
        processOneVideo(folder, [], sampling, target)
    else:
        videos = [v for v in os.listdir(folder) if v.lower().endswith('.mp4')]
        for v in videos:
            processOneVideo(os.path.join(folder, v), [], sampling, target)
