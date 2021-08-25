import os
import logging
import math
import copy
import sys
import numpy as np
import re

import torch
import torch.nn as nn
import pdb
from networks.revgrad import GradientReversal

def gen_connector(num_chan, use_BN, use_leakyReLU):
    connectors = []
    if use_BN:
        connectors.append(nn.BatchNorm2d(num_chan))
    if use_leakyReLU:
        connectors.append(nn.LeakyReLU(0.2))
    else:
        connectors.append(nn.ReLU())
    return connectors
        
class Discriminator(nn.Module):
    def __init__(self, num_in_chan, num_classes=2, do_avgpool=True, do_revgrad=True, num_base_chan=32):
        super(Discriminator, self).__init__()
        self.num_in_chan    = num_in_chan
        self.num_base_chan  = num_base_chan
        self.num_classes    = num_classes
        self.use_BN = True
        self.use_LeakyReLU = True
        self.do_revgrad = do_revgrad
        self.do_avgpool = do_avgpool
        
        layers =     [
                        nn.Conv2d(self.num_in_chan,         self.num_base_chan,
                                  kernel_size=4, stride=2, padding=1, bias=False),
                        *gen_connector(self.num_base_chan, self.use_BN, self.use_LeakyReLU),
                        
                        nn.Conv2d(self.num_base_chan,       2 * self.num_base_chan,
                                  kernel_size=4, stride=2, padding=1, bias=False),
                        *gen_connector(2 * self.num_base_chan, self.use_BN, self.use_LeakyReLU),

                        nn.Conv2d(2 * self.num_base_chan,   4 * self.num_base_chan,
                                  kernel_size=4, stride=2, padding=1, bias=False),
                        *gen_connector(4 * self.num_base_chan, self.use_BN, self.use_LeakyReLU),

                        nn.Conv2d(4 * self.num_base_chan,   8 * self.num_base_chan,
                                  kernel_size=4, stride=2, padding=1, bias=False),
                        *gen_connector(8 * self.num_base_chan, self.use_BN, self.use_LeakyReLU),

                        nn.Conv2d(8 * self.num_base_chan,   num_classes,
                                  kernel_size=4, stride=2, padding=1, bias=False),
                    ]
        if self.do_avgpool:
            tail =  [          
                        nn.AdaptiveAvgPool2d(1),
                        # nn.Sigmoid()  # use BCEWithLogitsLoss(), no need to do sigmoid.
                        # By default, Flatten() starts from dim=1. So output shape is [Batch, Classes].
                        nn.Flatten()
                    ]
            self.tail = nn.Sequential(*tail)
        else:
            self.tail = None
                    
        # Do not insert RevGrad layer when doing ADDA training.
        if self.do_revgrad:
            layers.insert(0, GradientReversal())
            
        self.model = nn.Sequential(*layers)       
        
    def forward(self, x):
        scores = self.model(x)
        if self.tail is None:
            num_tail_feat = scores.shape[2:].numel()
            tail = [          
                        # nn.Sigmoid()  # use BCEWithLogitsLoss(), no need to do sigmoid.
                        # By default, Flatten() starts from dim=1. So output shape is [Batch, Classes].
                        nn.Flatten(),
                        nn.Linear(num_tail_feat, self.num_classes),
                   ]
            self.tail = nn.Sequential(*tail)
            self.tail.to(next(self.model.parameters()).device)
            
        scores = self.tail(scores)
        return scores
        