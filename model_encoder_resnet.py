#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from util import fromNPtoPIL
import numpy as np

#
#ENCODER using ResNet-18
#
class EncoderResNet(nn.Module):

    #
    #
    #
    def __init__(self, s1 = 1024, s2 = 768, p_drop_out = 0.2, s_out = 512, differential = 0):
        super(EncoderResNet, self).__init__()
        
        #load resnet18 weights
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        s0 = resnet.fc.in_features
               
        #do we have a differential network?
        if(differential == 0):
            #freeze resnet parameters if it is not differential
            for param in resnet.parameters():
                param.requires_grad = False
                
        self.differential = differential
                
        #self.resnet = resnet
        blocks = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*blocks)     
                
        self.features_net = nn.Sequential(
            nn.Linear(s0, s1),
            nn.Dropout(p = p_drop_out),
            nn.ReLU(),
            nn.Linear(s1, s2),
            nn.Dropout(p = p_drop_out),
            nn.ReLU(),
            nn.Dropout(p = p_drop_out),
            nn.Linear(s2, s_out)
        )
        
        self.s_out = s_out

    #
    #
    #
    def forward(self, x):
            
        n_batches = x.size(0)
        n_frames = x.size(1)
        
        y = torch.zeros((n_batches, n_frames, self.s_out))
        device_index = x.get_device()
        if device_index >= 0:
            y = y.to(device_index)

        for t in range(0, n_frames):
            #apply resnet to each frame
            r_x = self.resnet(x[:, t, :, :, :])
            r_x = r_x.view(r_x.size(0), -1)

            y[:,t,:] = self.features_net(r_x)
            
        return y
