import torch.nn as nn
import torch.nn.functional as F
from .deform_conv_v2 import DeformConv2d

class DeformConvNetV1V2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(DeformConvNetV1V2, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = DeformConv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = DeformConv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = DeformConv2d(128, 128, kernel_size=3, padding=1, stride=2)
        self.bn4 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        # convs
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        # deformable convolution
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        # deformable convolution
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        # deformable convolution
        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = F.avg_pool2d(x, kernel_size=8, stride=1).view(x.size(0), -1)
        x = self.classifier(x)
        return x

