import numpy as np

np.random.seed(0)

from .deform import *
from .deform_unet import *
from .unet import UNet

MODELS = {'unet': UNet,
          'deform_v1': DeformConvNetV1V2,
          'deform_unet_v1': DUNetV1V2,
          }
