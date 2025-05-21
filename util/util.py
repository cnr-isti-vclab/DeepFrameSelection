#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms

from PIL import Image

#
#
#
def mkdir_s(output_dir):
    if os.path.isdir(output_dir) == False:
       os.mkdir(output_dir)
       
#
#
#
def fromVideoFrameToNP(frame, BGR = False):
    frame = frame.astype(dtype = np.float32)
    frame = frame / 255.0
    if BGR:
        out = np.zeros(frame.shape, dtype = np.float32)
        out[:,:,0] = frame[:,:,2]
        out[:,:,1] = frame[:,:,1]
        out[:,:,2] = frame[:,:,0]
    else:
        out = frame
    return out

#
#
#
def fromNPtoPIL(img):
    out = np.clip(img, 0.0, 1.0)
    formatted = (out * 255).astype('uint8')
    img_pil = Image.fromarray(formatted)
    return img_pil

#
# getResolution: get the video resolution
#
def getResolution(res = 2):
    return 910, 512#return (1920 // res), (1080 // res)

#
# getTransform: compute frame transformation
#
def getTransform(res_size_x, res_size_y, data_dir, differential = 0, bSimple = False):
    #
    if(differential == 0):
        name_file = os.path.join(data_dir, "data_pre/color_mean_std_dataset.txt")
        if os.path.isfile(name_file):
            array = []
            with open(name_file) as f:
                for line in f: # read rest of lines
                    array.append([float(x) for x in line.split()])

            transform = transforms.Compose([transforms.Resize([res_size_x, res_size_y]),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[array[0][0], array[0][1], array[0][2]],
                                                                 std =[array[1][0], array[1][1], array[1][2]])])
        else:
            if(data_dir == "ResNet"):
                print("ResNet")
                transform = transforms.Compose([transforms.Resize([224,224]),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            else:
                if(data_dir == "ResNetCol"):
                    print("ResNetCol")
                    transform = transforms.Compose([transforms.Resize([res_size_x, res_size_y]),
                                                    transforms.CenterCrop(res_size_y),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                else:
                    print('Basic')
                    transform = transforms.Compose([transforms.Resize([res_size_x, res_size_y]),
                                                    transforms.ToTensor()])
    else:
        if bSimple:
            transform = transforms.ToTensor()
        else:
            transform = transforms.Compose([transforms.Resize([res_size_x, res_size_y]),
                                            transforms.CenterCrop(res_size_y),
                                            transforms.ToTensor()])
    return transform
    
#
# readBlockFromVideo: reads a fps frames into a segment
#
def readBlockFromVideo(base_dir, frame_names, index, use_transform, differential, fps = 30):
    X = []
    for i in range(0, fps):
        name = frame_names[index + i]
        image = Image.open(os.path.join(base_dir, name))
   
        if use_transform is not None:
            image = use_transform(image)
        X.append(image)
   
    if(differential == 1):
        for i in range(0, fps - 1):
            X[i] = X[i + 1] - X[i]
        X = torch.stack(X[0:(fps - 1)], dim=0)
    else:
        X = torch.stack(X[0:fps], dim=0)
  
    return X
    
#
#
#
def dataAugmentation(img, j):
    
    img_out = []
    
    if(j == 0):
        img_out = img
    elif (j == 1):
        img_out = img.rotate(90)
    elif (j == 2):
        img_out = img.rotate(180)
    elif (j == 3):
        img_out = img.rotate(270)
    elif (j == 4):
        img_out = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    elif (j == 5):
        img_out = img.rotate(90)
        img_out = img_out.transpose(method=Image.FLIP_LEFT_RIGHT)
    else:
        img_out = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    
    return img_out

#
#
#
def plotGraph(array1, array2, folder, bLocal = False):
    # plot
    fig = plt.figure(figsize=(10, 4))
    n = min(len(array1), len(array2))
    plt.plot(np.arange(1, n + 1), array1[0:n])  # train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array2[0:n])  # train loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc="upper left")
    title = os.path.join(folder, "plot.png")
    plt.savefig(title, dpi=600)
    if bLocal:
        plt.savefig("plot.png", dpi=600)
    plt.close(fig)

#
#
#
def localPath(path):
    rev = path[::-1]
    t = rev.find('/')
    if t == 0:
        rev = rev[1:-1]
        t = rev.find('/')

    if t > 0:
        rev = rev[0:t]
    rev = rev[::-1]
    return rev

#
#
#
def mkdir_s(output_dir):
    if os.path.isdir(output_dir) == False:
       os.mkdir(output_dir)

#
#
#
def batchResizeLDRFolder(images_dir, scale, format_in, format_out):
    total_names = [f for f in os.listdir(images_dir) if f.endswith('.' + format_in)]
    total_names = sorted(total_names)
    
    folder_out = images_dir + '_s_' + str(scale)
    mkdir_s(folder_out)
    
    for filename in total_names:
        print(filename)
        filename_full = os.path.join(images_dir, filename)
        fn, fe = os.path.splitext(filename)
        img = Image.open(filename_full)
        (width, height) = (img.width // scale, img.height // scale)
        img = img.resize((width, height), resample = Image.LANCZOS)
        
        img.save(os.path.join(folder_out, fn + '.' + format_out))

#
#
#
if __name__ == '__main__':

    images_dir = sys.argv[1]
    scale = sys.argv[2]
    fmt_in = sys.argv[3]
    fmt_out = sys.argv[4]
    batchResizeLDRFolder(images_dir, int(scale), fmt_in, fmt_out)
