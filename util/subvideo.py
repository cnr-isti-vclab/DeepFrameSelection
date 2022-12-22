#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
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
    for i in range(0, (n-sampling), sampling):
        success, frame, i_k = v_in.getNextFrame(i, False)
        frame = frame.astype(float)
        if success:
            if bFirst:
                shape = frame.shape
                name_out = name + '_s_' + str(sampling) + '.mp4'
                print(name_out)
                v_out = createVideo(name_out, shape[1], shape[0])
                bFirst = False
            for j in range(0, sampling):
                success_j, frame_j, j_k = v_in.getNextFrame(i+j, False)
                frame_j = frame_j.astype(float)

                frame += frame_j
            
            if sampling > 1:
                frame /= sampling
                
            v_out.write(frame.astype(np.uint8))
        
    v_out.release()
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

    list = ['.mp4', '.MP4', '.mov', '.MOV', '.asf', '.ASF']
    fmt = os.path.splitext(folder)[1]

    if fmt in list:
        processOneVideo(folder, sampling)
    else:
        videos = [v for v in os.listdir(folder) if v.endswith('.MP4')]
        for v in videos:
            processOneVideo(os.path.join(folder, v),sampling)
    

