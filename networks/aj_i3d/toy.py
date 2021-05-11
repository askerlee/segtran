cd D:\Dropbox\ihpc\segmentation\networks\aj_i3d
import aj_i3d
a=torch.load("aj_rgb_imagenet.pth")
from easydict import EasyDict
args = EasyDict()
args.nclass = 400
args.do_pool1 = False
model=aj_i3d.AJ_I3D.get(args)
model.load_state_dict(a)
aa=torch.randn(1, 3, 64, 224, 224)
feat=model.extract_features(aa)
feat.keys()
['Conv3d_1a_7x7', 'MaxPool3d_2a_3x3', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3', 
'MaxPool3d_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'MaxPool3d_4a_3x3', 
'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 
'MaxPool3d_5a_2x2', 'Mixed_5b', 'Mixed_5c', 'pooled'])

'MaxPool3d_2a_3x3' is only downsampling on H, W (out of [T, H, W]). 
Can be disabled by setting args.do_pool1 = False.

# Downsampling: 2, 2, 2
'Conv3d_1a_7x7':    [1, 64, 32, 112, 112]
    
If do_pool1:
# Downsampling: 2, 4, 4
0 'MaxPool3d_2a_3x3': [1, 64, 32, 56, 56]
# Downsampling: 2, 4, 4
1 'Conv3d_2c_3x3':    [1, 192, 32, 56, 56]
# Downsampling: 2, 8, 8
2 'Mixed_3c':         [1, 480, 32, 28, 28]
# Downsampling: 4, 16, 16
3 'Mixed_4f':         [1,  832, 16, 14, 14]
# Downsampling: 8, 32, 32
4 'Mixed_5c':         [1, 1024, 8,  7,  7]

If not do_pool1:
# Downsampling: 2, 2, 2
0 'MaxPool3d_2a_3x3': [1, 64,  32, 112, 112]
# Downsampling: 2, 2, 2
1 'Conv3d_2c_3x3':    [1, 192, 32, 112, 112]
# Downsampling: 2, 4, 4
2 'Mixed_3c':         [1, 480, 32, 56, 56]
# Downsampling: 4, 8, 8
3 'Mixed_4f':         [1,  832, 16, 28, 28]
# Downsampling: 8, 16, 16
4 'Mixed_5c':         [1, 1024, 8,  14,  14]
