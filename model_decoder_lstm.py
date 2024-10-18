#
#Copyright (C) 2020-2021 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
#DECODER a classic LSTM
#
class DecoderLSTM(nn.Module):

    #
    #
    #
    def __init__(self, features_size = 512, hidden_layers = 2, hidden_nodes = 128, fc_size = 128, p_drop_out = 0.2):
        super(DecoderLSTM, self).__init__()

        self.LSTM = nn.LSTM(
            input_size = features_size,
            hidden_size = hidden_nodes,
            num_layers = hidden_layers,
            batch_first = True
        )
                        
        self.f = nn.Sequential(
            nn.Linear(hidden_nodes, fc_size),
            nn.ReLU(),
            nn.Dropout(p = p_drop_out),
            nn.Linear(fc_size, 1)
        )

    #
    #
    #
    def forward(self, x):
    
        #run the LSTM
        self.LSTM.flatten_parameters()
        out, (h_n, h_c) = self.LSTM(x, None)
        
        #run the regressor
        z = out[:, -1, :]
        y = self.f(z)

        if not self.training:
            y = y.clamp(0.0, 1.0)

        y = y.squeeze(0)
            
        return y
