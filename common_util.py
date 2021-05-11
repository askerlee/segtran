import torch
import numpy as np
import os
from matplotlib import cm
    
def get_default(args, var_name, default_settings, default_value, key_list):
    # var_name has been specified a custom value in command line. So not to use default value instead.
    if (var_name in args) and (args.__dict__[var_name] != default_value):
        return
    v = default_settings
    for k in key_list:
        v = v[k]
    args.__dict__[var_name] = v

def get_argument_list(arg_str, dtype):
    arg_list = [ dtype(elem) for elem in arg_str.split(",") ]
    return arg_list
    
def get_filename(file_path):
    filename = os.path.normpath(file_path).lstrip(os.path.sep).split(os.path.sep)[-1]
    return filename
    
class AverageMeters(object):
    """Computes and stores the average and current values of given keys"""
    def __init__(self):
        self.total_reset()

    def total_reset(self):
        self.val   = {'total': {}, 'disp': {}}
        self.avg   = {'total': {}, 'disp': {}}
        self.sum   = {'total': {}, 'disp': {}}
        self.count = {'total': {}, 'disp': {}}

    def disp_reset(self):
        self.val['disp']   = {}
        self.avg['disp']   = {}
        self.sum['disp']   = {}
        self.count['disp'] = {}

    def update(self, key, val, n=1, is_val_avg=True):
        if type(val) == torch.Tensor:
            val = val.item()
        if type(n) == torch.Tensor:
            n = n.item()
        
        if np.isnan(val):
            breakpoint()
            
        for sig in ('total', 'disp'):
            self.val[sig][key]    = val
            self.sum[sig].setdefault(key, 0)
            self.count[sig].setdefault(key, 0.0001)
            self.avg[sig].setdefault(key, 0)
            if is_val_avg:
                self.sum[sig][key] += val * n
            else:
                self.sum[sig][key] += val
                
            self.count[sig][key] += n
            self.avg[sig][key]    = self.sum[sig][key] / self.count[sig][key]


def get_seg_colormap(num_classes, return_torch):
    N = num_classes
    jet_colormap = cm.get_cmap('jet')
    seg_colormap = [ jet_colormap(i) for i in range(0, int(256/(N-1))*N, int(256/(N-1))) ]
    for i, color in enumerate(seg_colormap):
        seg_colormap[i] = [ int(256*frac) for frac in color[:3] ]

    if return_torch:
        seg_colormap = torch.tensor(seg_colormap, dtype=torch.uint8)
    else:
        seg_colormap = np.array(seg_colormap, dtype=np.uint8)
        
    return seg_colormap
    