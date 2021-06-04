import torch.nn as nn
import torch.nn.functional as F
import torch
from .deform_conv_v2 import DeformConv2d

class double_deform_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_deform_conv, self).__init__()
        self.conv = nn.Sequential(
            DeformConv2d(in_ch, out_ch, kernel_size=3, padding=0),
            # DeformConv2d(in_ch, out_ch, kernel_size=3, padding=0,modulation=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DeformConv2d(out_ch, out_ch, kernel_size=3, padding=0),
            # DeformConv2d(out_ch, out_ch, kernel_size=3, padding=0, modulation=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            # old way
            # ConvOffset2D(in_ch),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # ConvOffset2D(out_ch),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class deform_inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deform_inconv, self).__init__()
        self.conv = double_deform_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class deform_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(deform_down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_deform_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class deform_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(deform_up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_deform_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
