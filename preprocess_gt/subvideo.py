#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import cv2

import util_ip as ipt
from video import *

#
#
#
def processOneVideo(name_video, sampling):
    print('Processing video ' + name_video)

    name = os.path.splitext(name_video)[0]
    extension = os.path.splitext(name_video)[1]
    print(name)
    v_in = Video(name_video, extension)
    n = v_in.getNumFrames()
    
    bFirst = True
    for i in range(0, (n - sampling), sampling):
        success, frame, i_k = v_in.getNextFrame(i, False)
        
        frame = frame.astype('float32')
        
        if success:
            name_out = name + '_s_' + str(sampling) + '_' + '{0:06d}'.format(i) + '.png'

            success, value = ipt.checkLaplaicanBluriness(frame, 99.0)
            
            if success:
                cv2.imwrite(name_out, frame)
        
    v_in.release()


#
#
#
if __name__ == "__main__":

    if len(sys.argv) < 3:
        print('---------------------------------------------------------')
        print('---------------------------------------------------------')
        print('subvideo subsamples videos and store them as .png files:')
        print('subvideo video_file_name sampling_rate')
        print('')
        print('Example:')
        print('subvideo video.mp4 10')
        print('---------------------------------------------------------')
        print('---------------------------------------------------------')
        print('\n')
        sys.exit()
        
    folder = sys.argv[1]
    sampling = int(sys.argv[2])

    list = ['.mp4', '.mov', '.asf']
    fmt = os.path.splitext(folder)[1].lower()

    if fmt in list:
        processOneVideo(folder, sampling)
    else:
        videos = [v for v in os.listdir(folder) if v.lower().endswith('.mp4')]
        for v in videos:
            processOneVideo(os.path.join(folder, v),sampling)
    

