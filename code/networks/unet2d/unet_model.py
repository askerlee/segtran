""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
from networks.polyformer import Polyformer

class UNet(nn.Module):
    def __init__(self, n_channels, num_classes, bilinear=True, 
                 use_polyformer=False, num_polyformer_layers=1, 
                 num_attractors=256, num_modes=4, tie_qk_scheme='loose'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)
        self.use_polyformer = use_polyformer
        if self.use_polyformer:
            self.polyformer = Polyformer(feat_dim=64, num_layers=num_polyformer_layers, 
                                         num_attractors=num_attractors, num_modes=num_modes,
                                         tie_qk_scheme=tie_qk_scheme)
        self.num_vis_layers = 3 + (self.use_polyformer is not None)
            
    def forward(self, x):
        self.feature_maps = []
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.feature_maps.append(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        self.feature_maps.append(x)
        # x and x2 are concatenated along channel to 128+128=256 channels.
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        self.feature_maps.append(x)
        if self.use_polyformer:
            x = self.polyformer(x)
            self.feature_maps.append(x)
        logits = self.outc(x)
        return logits
