#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import torch
import glob2
import re
import torch.nn as nn

from model_decoder_lstm import DecoderLSTM
from model_encoder_resnet import EncoderResNet

#
#
#
class ModelVideo(nn.Module):

    #
    #
    #
    def __init__(self, run, device, differential = 0):
    
        super(ModelVideo, self).__init__()
        
        #create the model
        self.cnn_encoder = EncoderResNet(1024, 768, 0.2, 512, differential).to(device)
            
        self.lstm_decoder = DecoderLSTM().to(device)
        
        if run != '':
            print('Resume ModelVideo')
            try:
                ext = os.path.splitext(run)[1]

                if (ext == ''):
                    #load the model
                    self.run = run
                    ckpt_dir = os.path.join(run, 'ckpt')
    
                    ckpt_name = os.path.join(ckpt_dir, '*.pth')
                    print(ckpt_dir)
                    print(ckpt_name)
                    ckpts = glob2.glob(ckpt_name)
                    assert ckpts, "No checkpoints to resume from!"

                    def get_epoch(ckpt_url):
                        s = re.findall("ckpt_e(\d+).pth", ckpt_url)
                        epoch = int(s[0]) if s else -1
                        return epoch, ckpt_url
    
                    print(ckpts)
                    start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
                    print('Checkpoint:', ckpt)
                else:
                    ckpt = run
                
                bCuda = torch.cuda.is_available() # do we have a CUDA GPU?
                device = torch.device("cuda" if bCuda else "cpu")
                ckpt = torch.load(ckpt, map_location = device)

                c0 = ckpt['cnn_model']
                
                try:
                    c1 = ckpt['lstm_model']
                except:
                    c1 = ckpt['rnn_model']
                    
                self.cnn_encoder.load_state_dict(c0)
                self.lstm_decoder.load_state_dict(c1)
            except:
                print('No model to resume')
        
        self.cnn_encoder.eval()
        self.lstm_decoder.eval()

        self.device = device
     
    #
    #
    #
    def forward(self, x):
        output_cnn = self.cnn_encoder(x)
        output = self.lstm_decoder(output_cnn)
        return output

    #
    #
    #
    def predict(self, X):
        self.eval()
        
        X = X.to(self.device)        
        if (len(X.shape) == 4):
            X = torch.unsqueeze(X, 0)
        
        return self.predictSimple(X)

    #
    #
    #
    def predictSimple(self, X):
        with torch.no_grad():
            output_cnn = self.cnn_encoder(X)
            output = self.lstm_decoder(output_cnn)
            return output.data.cpu().item()
