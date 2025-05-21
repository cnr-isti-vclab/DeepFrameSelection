#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim

#
#
#
def readCV2(filename, bBGR = False):
    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if bBGR:
        out = np.zeros(img.shape, dtype = np.float32)
        out[:,:,0] = img[:,:,2]
        out[:,:,1] = img[:,:,1]
        out[:,:,2] = img[:,:,0]
        return out
    else:
        return img
        
#
#
#
def writeCV2(img, filename, bBGR = True):
    if bBGR:
        out = np.zeros(img.shape, dtype = np.float32)
        out[:,:,0] = img[:,:,2]
        out[:,:,1] = img[:,:,1]
        out[:,:,2] = img[:,:,0]
        img = out

    cv2.imwrite(filename, fromFloatToUint8(img))

#
#
#
def luma(img, bBGR = False):
        
    r, c, col = img.shape
    if col == 3:
        if bBGR:
            return 0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]
        else:
            return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        return []

#
#
#
def fromFloatToUint8(img):
    formatted = (img * 255).astype('uint8')
    return formatted

#
#
#
def checkMTB(img1, img2, thr = 4, bBGR = False):
    gray1_u8 = fromFloatToUint8(luma(img1, bBGR))
    gray2_u8 = fromFloatToUint8(luma(img2, bBGR))
    mtb = cv2.createAlignMTB()
    shift = mtb.calculateShift(gray1_u8, gray2_u8)
    len = np.sqrt(shift[0] * shift[0] + shift[1] * shift[1])
    return len < thr, len

#
#
#
def checkKeyPointBluriness(img, thr = 16, bBGR = False):
    gray = luma(img, bBGR)
    gray_u8 = fromFloatToUint8(gray)
    orb = cv2.ORB_create()
    kp = orb.detect(gray_u8, None)
    len_kp = len(kp)
    kp, des = orb.compute(img, kp)
    return (len_kp >= thr), len_kp

#
#
#
def checkLaplaicanBluriness(img, thr = 100.0, bBGR = False):
    L = luma(img, bBGR)
    value = cv2.Laplacian(L, cv2.CV_32F).var()
    return (value > thr), value

#
#
#
def npMedianThreshold(img):
    thr = np.median(img);
    img_wrk = np.copy(img)
    idx = img[:,:,1] > thr
    img_wrk[idx,0] = 1.0
    img_wrk[idx,1] = 1.0
    img_wrk[idx,2] = 1.0

    idx = img[:,:,1] <= thr
    img_wrk[idx,0] = 0.0
    img_wrk[idx,1] = 0.0
    img_wrk[idx,2] = 0.0
    
    return img_wrk

#
#
#
def checkSimilarity(img1, img2, thr = 0.925, bBGR = False):

    if(thr < 0.0):
        thr = 0.925

    ssim_none = ssim(luma(img1, bBGR), luma(img2, bBGR), data_range=1.0, multichannel = False)
    return (ssim_none >= thr), ssim_none

#
#
#
if __name__ == "__main__":

    if len(argv) < 2:
       sys.exit()
       
    img = readCV2(sys.argv[1])
    img1 = readCV2(sys.argv[2])
    
    print(checkSimilarity(img, img1))

