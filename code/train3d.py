import os
import sys
import re
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import json
import numpy as np
import math
from functools import reduce

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from optimization import BertAdam

import segmentation_models_pytorch as smp
from networks.vnet import VNet
from networks.segtran3d import Segtran3d
from networks.segtran3d import CONFIG as config3d
from networks.segtran25d import Segtran25d
from networks.segtran25d import CONFIG as config25d
from networks.unet3d import Modified3DUNet as UNet3D
from utils.losses import dice_loss_indiv
import dataloaders.datasets3d
from dataloaders.datasets3d import brats_map_label, RandomCrop, CenterCrop, \
                                   RandomRotFlip, ToTensor, RandomNoise, RandomResizedCrop
from common_util import AverageMeters, get_default, get_filename
import imgaug.augmenters as iaa
import imgaug as ia

def print0(*print_args, **kwargs):
    if args.local_rank == 0:
        print(*print_args, **kwargs)

parser = argparse.ArgumentParser()
###### General arguments ######
parser.add_argument('--task', dest='task_name', type=str, default='brats', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='train_ds_names', type=str, default=None, help='Dataset folders. Can specify multiple datasets (separated by ",")')
parser.add_argument('--split', dest='ds_split', type=str, default='train', 
                    choices=['train', 'all'], help='Split of the dataset')
                    
parser.add_argument('--maxiter', type=int,  default=10000, help='maximum training iterations')
parser.add_argument('--saveiter', type=int,  default=500, help='save model snapshot every N iterations')
parser.add_argument('--cp', dest='checkpoint_path', type=str,  default=None, help='Load this checkpoint')

###### Optimization settings ######
parser.add_argument('--lrwarmup', dest='lr_warmup_steps', type=int,  default=500, help='Number of LR warmup steps')
parser.add_argument('--bs', dest='batch_size', type=int, default=4, help='Total batch_size on all GPUs')
parser.add_argument('--opt', type=str,  default=None, help='optimization algorithm')
parser.add_argument('--lr', type=float,  default=-1, help='learning rate')
parser.add_argument('--decay', type=float,  default=-1, help='weight decay')
parser.add_argument('--gradclip', dest='grad_clip', type=float,  default=-1, help='gradient clip')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--locprob", dest='localization_prob', default=0.5, 
                    type=float, help='Probability of doing localization during training')
parser.add_argument("--tunebn", dest='tune_bn_only', action='store_true', 
                    help='Only tune batchnorms for domain adaptation, and keep model weights unchanged.')
parser.add_argument('--diceweight', dest='MAX_DICE_W', type=float, default=0.5, 
                    help='Weight of the dice loss.')                    

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')

parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')

parser.add_argument('--bb', dest='backbone_type', type=str,  default=None, 
                    help='Backbone of Segtran / Encoder of other models')
parser.add_argument("--nopretrain", dest='use_pretrained', action='store_false', 
                    help='Do not use pretrained weights.')

###### Transformer architecture settings ######                    
parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false', 
                    help='Do not use attractor transformers (Default: use to increase scalability).')
parser.add_argument("--attractors", dest='num_attractors', default=1024,
                    type=int, help='Number of attractors in the squeezed transformer.')
parser.add_argument("--noqkbias", dest='qk_have_bias', action='store_false', 
                    help='Do not use biases in Q, K projections (Using biases leads to better performance on BraTS).')

parser.add_argument("--translayers", dest='num_translayers', default=1,
                    type=int, help='Number of Cross-Frame Fusion layers.')
parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=None, 
                    help='Compression ratio of channel numbers of each transformer layer to save RAM.')
parser.add_argument('--modes', type=int, dest='num_modes', default=-1, help='Number of transformer modes')
parser.add_argument('--multihead', dest='ablate_multihead', action='store_true', 
                    help='Ablation to multimode transformer (using multihead instead)')

parser.add_argument('--dropout', type=float, dest='dropout_prob', default=-1, help='Dropout probability')

parser.add_argument('--pos', dest='pos_code_type', type=str, default='lsinu', 
                    choices=['lsinu', 'none', 'rand', 'sinu', 'bias'],
                    help='Positional code scheme')
parser.add_argument('--posw', dest='pos_code_weight', type=float, default=1.0)
parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                    help='The radius of positional biases')                    
parser.add_argument("--squeezeuseffn", dest='has_FFN_in_squeeze', action='store_true', 
                    help='Use the full FFN in the first transformer of the squeezed attention '
                         '(Default: only use the first linear layer, i.e., the V projection)')

parser.add_argument("--attnconsist", dest='use_attn_consist_loss', action='store_true', 
                    help='This loss encourages the attention scores to be consistent with the segmentation mask')
parser.add_argument("--attnconsistweight", dest='ATTNCONSIST_W', type=float, default=0.01,
                    help='Weight of the attention consistency loss')

############## Mince transformer settings ##############                          
parser.add_argument("--mince", dest='use_mince_transformer', action='store_true',
                    help='Use Mince (Multi-scale) Transformer to save GPU RAM.')
parser.add_argument("--mincescales", dest='mince_scales', type=str, default=None, 
                    help='A list of numbers indicating the mince scales.')
parser.add_argument("--minceprops", dest='mince_channel_props', type=str, default=None, 
                    help='A list of numbers indicating the relative proportions of channels of each scale.')

parser.add_argument("--segtran", dest='segtran_type', 
                    default='3d',
                    choices=['25d', '3d'],
                    type=str, help='Use 3D or 2.5D of segtran.')
###### End of transformer architecture settings ######

###### Segtran (non-transformer part) settings ######
parser.add_argument("--into3", dest='inchan_to3_scheme', default=None,
                    choices=['avgto3', 'stemconv', 'dup3', 'bridgeconv'],
                    help='Scheme to convert input into pseudo-RGB format')
parser.add_argument("--upd", dest='out_fpn_upsampleD_scheme', default='interp',
                    choices=['conv', 'interp', 'none'],
                    help='Depth output upsampling scheme (if you have a bigger GPU, you can use conv instead of interp)')
                     
parser.add_argument("--infpn", dest='in_fpn_layers', default='34',
                    choices=['234', '34', '4'],
                    help='Specs of input FPN layers')
parser.add_argument("--outfpn", dest='out_fpn_layers', default='1234',
                    choices=['1234', '234', '34'],
                    help='Specs of output FPN layers')
parser.add_argument("--outdrop", dest='out_fpn_do_dropout', action='store_true', 
                    help='Do dropout on out fpn features.')
parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true', 
                    help='Use BatchNorm instead of GroupNorm in input FPN.')
                    
parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false', 
                    help='Do not upsize backbone feature maps by 2.')

parser.add_argument('--insize', dest='orig_input_size', type=str, default=None, 
                    help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='orig_patch_size', type=str, default=None, 
                    help='Crop input images to this size for training.')
parser.add_argument('--scale', dest='input_scale', type=str, default=None, 
                    help='Scale input images by this ratio for training.')
parser.add_argument('--dgroup', dest='D_groupsize', type=int, default=-1, 
                    help='For 2.5D segtran, group the depth dimension of the input images and merge into the batch dimension.')                    
parser.add_argument('--dpool', dest='D_pool_K', type=int, default=-1, 
                    help='Scale input images by this ratio for training.')

###### Augmentation settings ######
# Using random scaling as augmentation usually hurts performance. Not sure why.
parser.add_argument("--randscale", type=float, default=0, help='Probability of random scaling augmentation.')
parser.add_argument("--affine", dest='do_affine', action='store_true', help='Do random affine augmentation.')
parser.add_argument('--mod', dest='chosen_modality', type=int, default=-1, help='The modality to use if images are of multiple modalities')
parser.add_argument('--focus', dest='focus_class', type=int, default=-1, help='The class that is particularly predicted by the current modality (with higher loss weight)')
                    
args_dict = {   'trans_output_type': 'private',
                'mid_type': 'shared',
                'in_fpn_scheme':     'AN',
                'out_fpn_scheme':    'AN',                
            }
            
cond_args_dict = { '25d': {'backbone_type': 'eff-b3', 
                           'inchan_to3_scheme': 'stemconv', 
                           'D_groupsize': 1}, 
                   '3d':  {'backbone_type': 'i3d',    
                           'inchan_to3_scheme': 'bridgeconv',     
                           'D_groupsize': 1}
                 }
                    
args = parser.parse_args()
for arg, v in args_dict.items():
    args.__dict__[arg] = v
for arg, v in cond_args_dict[args.segtran_type].items():
    # if this arg is not set through command line (i.e., in its default value None or -1), 
    # then take value from cond_args_dict.
    if (arg not in args.__dict__) or (args.__dict__[arg] is None) or (args.__dict__[arg] is -1):
        args.__dict__[arg] = v

if args.mince_scales is not None:
    args.mince_scales = [ int(L) for L in args.mince_scales.split(",") ]
if args.mince_channel_props is not None:
    args.mince_channel_props = [ float(L) for L in args.mince_channel_props.split(",") ]

if args.ablate_multihead:
    args.use_squeezed_transformer = False
args.device = 'cuda'
    
unet_settings    = { 'opt': 'adam', 
                     'lr': { 'sgd': 0.01, 'adam': 0.001 },
                     'decay': 0.0001, 'grad_clip': -1,
                   }
segtran_settings = { 'opt': 'adamw',
                     'lr': { 'adamw': 0.0002 },
                     'decay': 0.0001,  'grad_clip': 0.1,
                     'dropout_prob': { '234': 0.3, '34': 0.2, '4': 0.2 },
                     'num_modes':    { '234': 2,   '34': 4,   '4': 4 }
                   }


default_settings = { 'unet':    unet_settings,
                     'vnet':    unet_settings,
                     'segtran': segtran_settings,
                     'brats': {
                                 'num_classes': 4,
                                 'bce_weight':  [0., 3, 1, 1.75],  # bg, ET, WT, TC
                                 'ds_class':    'BratsSet',
                                 'train_ds_names':  '2019train',
                                 'test_ds_name':    '2019valid',
                                 'chosen_modality': -1,
                                 'xyz_permute':     None, # (1, 2, 0),
                                 'orig_input_size': None,
                                 # each dim of the orig_patch_size should always be multiply of 8.
                                 'orig_patch_size': (112, 112, 96),
                                 'input_scale':     (1,   1,   1),
                                 'D_pool_K':         2,
                                 'has_mask':    { '2019train': True,    '2019valid': False, 
                                                  '2020train': True,    '2020valid': False },
                                 'weight':      { '2019train': 1,       '2019valid': 1, 
                                                  '2020train': 1,       '2020valid': 1 }
                               },
                     'atria': {
                                 'num_classes': 2,
                                 'bce_weight':  [0., 1],  
                                 'ds_class':    'AtriaSet',
                                 'train_ds_names':  'train',
                                 'test_ds_name':    'test',
                                 'chosen_modality': -1,
                                 'xyz_permute':     None,
                                 'orig_input_size': None,
                                 # each dim of the orig_patch_size should always be multiply of 8.
                                 'orig_patch_size': 112,
                                 'input_scale':     (1, 1, 1),
                                 'D_pool_K':         2,
                                 'has_mask':    { 'train': True },
                                 'weight':      { 'train': 1 }
                               },   
                   }

get_default(args, 'orig_input_size',    default_settings, None, [args.task_name, 'orig_input_size'])
get_default(args, 'orig_patch_size',    default_settings, None, [args.task_name, 'orig_patch_size'])
get_default(args, 'input_scale',        default_settings, None, [args.task_name, 'input_scale'])
get_default(args, 'D_pool_K',           default_settings, -1,   [args.task_name, 'D_pool_K'])
get_default(args, 'xyz_permute',        default_settings, None, [args.task_name, 'xyz_permute'])
get_default(args, 'chosen_modality',    default_settings, -1,   [args.task_name, 'chosen_modality'])
get_default(args, 'num_classes',        default_settings, None, [args.task_name, 'num_classes'])
args.binarize = (args.num_classes == 2)

if type(args.orig_patch_size) == str:
    args.orig_patch_size = [ int(L) for L in args.orig_patch_size.split(",") ]
if type(args.orig_patch_size) == int:
    args.orig_patch_size = (args.orig_patch_size, args.orig_patch_size, args.orig_patch_size)
if type(args.orig_input_size) == str:
    args.orig_input_size = [ int(L) for L in args.orig_input_size.split(",") ]
if type(args.orig_input_size) == int:
    args.orig_input_size = (args.orig_input_size, args.orig_input_size, args.orig_input_size)
if type(args.input_scale) == str:
    args.input_scale = [ float(L) for L in args.input_scale.split(",") ]
    
args.input_patch_size = [ int(args.input_scale[i] * L) for i, L in enumerate(args.orig_patch_size) ]
print("Orig patch: {}. Model input patch: {}".format(args.orig_patch_size, args.input_patch_size))

get_default(args, 'ds_class',           default_settings, None,   [args.task_name, 'ds_class'])  
get_default(args, 'train_ds_names',     default_settings, None, [args.task_name, 'train_ds_names'])
args.train_ds_names = args.train_ds_names.split(",")
train_data_paths = []

for ds_name in args.train_ds_names:
    train_data_path = os.path.join("../data/", args.task_name, ds_name)
    train_data_paths.append(train_data_path)

args.job_name = '{}-{}'.format(args.task_name, ','.join(args.train_ds_names))

timestamp = datetime.now().strftime("%m%d%H%M")
checkpoint_dir = "../model/%s-%s-%s" %(args.net, args.job_name, timestamp)
print0("Model checkpoints will be saved to '%s'" %checkpoint_dir)
if not os.path.isdir(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
        
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def init_optimizer(net, max_epoch, batch_per_epoch):
    # Prepare optimizer
    # Each param is a tuple ( param name, Parameter(tensor(...)) )
    optimized_params = list( param for param in net.named_parameters() if param[1].requires_grad )
    low_decay = ['backbone'] #['bias', 'LayerNorm.weight']
    no_decay  = []
    high_lr   = ['alphas']
    
    high_lr_params      = []
    high_lr_names       = []
    no_decay_params     = []
    no_decay_names      = []
    low_decay_params    = []
    low_decay_names     = []
    normal_params       = []
    normal_names        = []

    for n, p in optimized_params:
        if any(nd in n for nd in no_decay):
            no_decay_params.append(p)
            no_decay_names.append(n)
        elif any(nd in n for nd in low_decay):
            low_decay_params.append(p)
            low_decay_names.append(n)
        elif any(nd in n for nd in high_lr):
            high_lr_params.append(p)
            high_lr_names.append(n)
        else:
            normal_params.append(p)
            normal_names.append(n)

    optimizer_grouped_parameters = [
        { 'params': normal_params,       'weight_decay': args.decay,        'lr': args.lr },
        { 'params': low_decay_params,    'weight_decay': args.decay * 0.1,  'lr': args.lr },
        { 'params': no_decay_params,     'weight_decay': 0.0,               'lr': args.lr },
        { 'params': high_lr_params,      'weight_decay': 0.0,               'lr': args.lr * 100 },
    ]

    for group_name, param_group in zip( ('normal', 'low_decay', 'no_decay', 'high_lr'), 
                                        (normal_params, low_decay_params, no_decay_params, high_lr_params) ):
        print0("{}: {} weights".format(group_name, len(param_group)))

    args.t_total = int(batch_per_epoch * max_epoch)
    print0("Batch per epoch: %d" % batch_per_epoch)
    print0("Total Iters: %d" % args.t_total)
    print0("LR: %f" %args.lr)

    args.lr_warmup_ratio = args.lr_warmup_steps / args.t_total
    print0("LR Warm up: %.3f=%d iters" % (args.lr_warmup_ratio, args.lr_warmup_steps))

    optimizer = BertAdam(optimizer_grouped_parameters,
                         warmup=args.lr_warmup_ratio, t_total=args.t_total,
                         weight_decay=args.decay)

    return optimizer

def load_model(net, optimizer, args, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        if 'optim_state' in state_dict:
            optim_state_dict = state_dict['optim_state']
        else:
            optim_state_dict = None
        cp_args          = state_dict['args']
        cp_iter_num      = state_dict['iter_num']
    else:
        model_state_dict = state_dict
        optim_state_dict = None
        cp_args          = None
        cp_iter_num      = 0

    ignored_keys = [ 'maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                     'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size',
                     'saveiter', 'dice_warmup_steps', 'opt', 'lr', 'decay',
                     'initializer_range', 'base_initializer_range',
                     'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                     'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                     'bce_weight', 'translayer_compress_ratios',
                     'seed', 'debug', 'ds_name', 'batch_size', 'dropout_prob',
                     'patch_size', 'orig_input_size', 'output_upscale', 'use_pretrained',
                     'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                     'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'ds_split', 'train_ds_names',
                     'job_name', 'mean', 'std', 'mask_thres' ]
                    
    if args.net == 'segtran' and cp_args is not None:
        for k in cp_args:
            if (k not in ignored_keys) and (args.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args.__dict__[k], k, cp_args[k]))
                exit(0)

    params.update(model_state_dict)
    net.load_state_dict(params)
    if optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
                
    logging.info("Model loaded from '{}'".format(checkpoint_path))
    # warmup info is mainly in optim_state_dict. So after loading the checkpoint, 
    # the optimizer won't do warmup already.
    args.lr_warmup_steps = 0
    print0("LR Warm up reset to 0 iters.")
    
    return cp_iter_num

def save_model(net, optimizer, args, checkpoint_dir, iter_num):
    if args.local_rank == 0:
        save_model_path = os.path.join(checkpoint_dir, 'iter_'+str(iter_num)+'.pth')
        torch.save( { 'iter_num': iter_num, 'model': net.state_dict(),
                      # 'optim_state': optimizer.state_dict(),
                      'args': vars(args) },  
                    save_model_path)
                        
        logging.info("save model to '{}'".format(save_model_path))

def warmup_constant(x, warmup=500):
    if x < warmup:
        return x/warmup
    return 1


# layers_attn_scores: a list of [B0, 1, N, N]. 
# mask: [B0, C, H, W, D]. orig_feat_shape: [D2, H2, W2]. H2*W2*D2 = N.
def attn_consist_loss_fun(layers_attn_scores, orig_feat_shape, mask, only_first_layer=True):
    mask = mask.permute(0, 1, 4, 2, 3)
    # resized_mask: [B0, C, D2, H2, W2]. 
    resized_mask = F.interpolate(mask, size=orig_feat_shape, mode='trilinear', align_corners=False)
    # flat_mask: [B0, N, C]
    flat_mask = resized_mask.view(resized_mask.size(0), resized_mask.size(1), -1)
    # consistency_mat: [B0, N, N]. consistency_mat should contain binary values.
    consistency_mat = torch.matmul(flat_mask.transpose(-2, -1), flat_mask)
    consistency_mat = torch.clip(consistency_mat, 0, 1)

    attn_consist_loss = 0
    if only_first_layer:
        N = 1
    else:
        N = len(layers_attn_scores)

    for layer_attn_scores in layers_attn_scores[:N]:
        if type(layer_attn_scores) is list:
            in_ator_scores, ator_out_scores = layer_attn_scores
            #in_ator_scores: [6, 1, 256, 1600]. ator_out_scores: [6, 1, 1600, 256]
            layer_attn_scores = torch.matmul(ator_out_scores, in_ator_scores)
        attn_consist_loss += F.binary_cross_entropy_with_logits(layer_attn_scores.squeeze(1), consistency_mat)
    attn_consist_loss /= N
    return attn_consist_loss

def compose2(f, g):
    return lambda *a, **kw: g(f(*a, **kw))
def compose(*fs):
    return reduce(compose2, fs)
                     
if __name__ == "__main__":
    logFormatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    rootLogger = logging.getLogger()
    while rootLogger.handlers:
         rootLogger.handlers.pop()
    fileHandler = logging.FileHandler(checkpoint_dir+"/log.txt")
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.INFO)
    rootLogger.propagate = False

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        print0("Set this session to deterministic mode")

    if args.tune_bn_only:
        if args.checkpoint_path is None:
            print0("Tuning BN requires to specify a checkpoint to load")
            exit(0)
        args.lr_warmup_steps = 0
        
    is_master = (args.local_rank == 0)
    n_gpu = torch.cuda.device_count()
    args.distributed = False
    # 'WORLD_SIZE' is set by 'torch.distributed.launch'. Do not set manually
    # Its value is specified by "--nproc_per_node=k"
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1

    args.batch_size //= args.world_size
    print0("n_gpu: {}, world size: {}, rank: {}, batch size: {}, seed: {}".format(
                n_gpu, args.world_size, args.local_rank, args.batch_size,
                args.seed))

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://',
                                             world_size=args.world_size,
                                             rank=args.local_rank)

    get_default(args, 'opt',            default_settings, None,   [args.net, 'opt'])
    get_default(args, 'lr',             default_settings, -1,     [args.net, 'lr', args.opt])
    get_default(args, 'decay',          default_settings, -1,     [args.net, 'decay'])
    get_default(args, 'grad_clip',      default_settings, -1,     [args.net, 'grad_clip'])
    
    get_default(args, 'bce_weight',     default_settings, None,   [args.task_name, 'bce_weight'])

    if args.binarize:
        args.bce_weight = None
    else:
        args.bce_weight = torch.tensor(args.bce_weight).cuda()
        args.bce_weight = args.bce_weight * (args.num_classes - 1) / args.bce_weight.sum()
    
    if args.net == 'segtran':
        get_default(args, 'dropout_prob',   default_settings, -1, [args.net, 'dropout_prob', args.in_fpn_layers])
        get_default(args, 'num_modes',      default_settings, -1, [args.net, 'num_modes', args.in_fpn_layers])

    if args.randscale > 0:
        crop_percents = (-args.randscale, args.randscale)
    else:
        crop_percents = (0, 0)

    # Images after augmentation/transformation should keep their original size orig_input_size.  
    # Will be resized before fed into the model.  
    tgt_height, tgt_width, tgt_depth    = args.orig_patch_size

    if args.do_affine:
        affine_prob = 0.3
    else:
        affine_prob = 0
            
    common_aug_func =   iaa.Sequential(
                            [
                                # resize the image to the shape of orig_input_size
                                iaa.Resize({'height': tgt_height, 'width': tgt_width}),   
                                iaa.Sometimes(0.5, iaa.CropAndPad(
                                    percent=crop_percents,
                                    pad_mode='constant', # ia.ALL,
                                    pad_cval=0
                                )),
                                # apply the following augmenters to most images
                                iaa.Fliplr(0.2),  # Horizontally flip 20% of all images
                                iaa.Flipud(0.2),  # Vertically flip 20% of all images
                                iaa.Sometimes(0.3, iaa.Rot90((1,3))), # Randomly rotate 90, 180, 270 degrees 30% of the time
                                # Affine transformation reduces dice by ~1%. So disable it by setting affine_prob=0.
                                iaa.Sometimes(affine_prob, iaa.Affine(
                                        rotate=(-45, 45), # rotate by -45 to +45 degrees
                                        shear=(-16, 16), # shear by -16 to +16 degrees
                                        order=1,
                                        cval=(0,255),
                                        mode='reflect'
                                )),
                                # iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades.
                                # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
                                # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
                                iaa.PadToFixedSize(width=tgt_width,  height=tgt_height),    
                                iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
                                # iaa.Grayscale(alpha=args.gray_alpha)
                            ])
    

    DataSetClass = dataloaders.datasets3d.__dict__[args.ds_class]
    
    db_trains = []
    ds_settings = default_settings[args.task_name]
    # transform() is applied on both image and mask in DataSetClass.
    transform = compose(
                   # RandomNoise(),
                   RandomRotFlip(),
                   RandomCrop(args.orig_patch_size),
                   ToTensor(),
                )
                                            
    for i, train_data_path in enumerate(train_data_paths):
        ds_name         = args.train_ds_names[i]
        ds_weight       = ds_settings['weight'][ds_name]
        has_mask        = ds_settings['has_mask'][ds_name]
                                                                            
        db_train = DataSetClass(base_dir=train_data_path,
                                split=args.ds_split,
                                mode='train',
                                ds_weight=ds_weight,
                                xyz_permute=args.xyz_permute,
                                transform=transform,
                                chosen_modality=args.chosen_modality,
                                binarize=args.binarize,
                                train_loc_prob=args.localization_prob,
                                min_output_size=args.orig_patch_size
                               )

        db_trains.append(db_train)
        print0("{}: {} images, has_mask: {}".format(
                args.train_ds_names[i], len(db_train), has_mask))

    db_train_combo = ConcatDataset(db_trains)
    print0("Combined dataset: {} images".format(len(db_train_combo)))
                            
    if args.distributed:
        train_sampler = DistributedSampler(db_train_combo, shuffle=True, 
                                           num_replicas=args.world_size, 
                                           rank=args.local_rank)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    args.num_workers = 0 if args.debug else 2
    trainloader = DataLoader(db_train_combo, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.num_workers, pin_memory=False, shuffle=shuffle,
                             worker_init_fn=worker_init_fn)

    max_epoch = math.ceil(args.maxiter / len(trainloader))

    if args.chosen_modality == -1:
        args.orig_in_channels = db_trains[0].num_modalities
    else:
        # The dataset class only generates the chosen modality, so input channel number is 1.
        args.orig_in_channels = 1

    if args.translayer_compress_ratios is not None:
        args.translayer_compress_ratios = [ float(r) for r in args.translayer_compress_ratios.split(",") ]
    else:
        args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers + 1) ]
                                             
    logging.info(str(args))
    base_lr = args.lr

    if args.net == 'vnet':
        net = VNet(n_channels=1, num_classes=args.num_classes, normalization='batchnorm', has_dropout=True)
    elif args.net == 'unet':
        net = UNet3D(in_channels=1, num_classes=args.num_classes)
    elif args.net == 'segtran':
        if args.segtran_type == '3d':
            config3d.update_config(args)
            net = Segtran3d(config3d)
        else:
            config25d.update_config(args)
            net = Segtran25d(config25d)            
    else:
        breakpoint()
        
    net.cuda()
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.opt == 'adamw':
        optimizer = init_optimizer(net, max_epoch, len(trainloader))
    
    if args.checkpoint_path is not None:
        iter_num = load_model(net, optimizer, args, args.checkpoint_path)
        start_epoch = math.ceil(iter_num / len(trainloader))
        logging.info("Start epoch/iter: {}/{}".format(start_epoch, iter_num))
    else:
        iter_num = 0
        start_epoch = 0
                        
    net.train()

    real_net = net

    if args.distributed:
        sync_bn_net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
        sync_bn_net.cuda()
        net = DistributedDataParallel(sync_bn_net, device_ids=[args.local_rank],
                                      output_device=args.local_rank,
                                      find_unused_parameters=True)

    scheduler = None
    lr_ = base_lr

    if is_master:
        writer = SummaryWriter(checkpoint_dir + '/log')
    logging.info("{} epochs, {} itertations each.".format(max_epoch, len(trainloader)))

    dice_loss_func = dice_loss_indiv
    class_weights = torch.ones(args.num_classes).cuda()
    class_weights[0] = 0
    if args.focus_class != -1 and args.num_classes > 2:
        class_weights[args.focus_class] = 2
    class_weights /= class_weights.sum()
    
    # img_stats = AverageMeters()
    # The weight of first class, i.e., the background is set to 0. Because it's negative
    # of WT. Optimizing w.r.t. WT is enough.
    # Positive voxels (ET, WT, TC) receive higher weights as they are fewer than negative voxels.
    bce_loss_func = nn.BCEWithLogitsLoss( # weight=weights_batch.view(-1,1,1,1),
                                          pos_weight=args.bce_weight)
    
    for epoch_num in tqdm(range(start_epoch, max_epoch), ncols=70, disable=(args.local_rank != 0)):
        print0()
        time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch_num)

        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            if args.task_name == 'brats':
                # after mapping, mask_batch is already float.
                mask_batch = brats_map_label(mask_batch, args.binarize)
                
            if args.randscale:
                volume_batch, mask_batch = RandomResizedCrop(volume_batch, mask_batch, 
                                                             args.orig_patch_size, crop_percents)
            
            volume_batch = F.interpolate(volume_batch, size=args.input_patch_size,
                                         mode='trilinear', align_corners=False)
                
            iter_num = iter_num + 1
            # volume_batch: [4, 1, 112, 112, 80]
            # before brats_map_label: 
            # mask_batch:   [4,    112, 112, 80]
            # after brats_map_label: 
            # mask_batch:   [4, 4, 112, 112, 80]
            # outputs:      [4, 4, 112, 112, 80]
            outputs = net(volume_batch)
            if args.net == 'unet':
                outputs = outputs[1]
                
            outputs = F.interpolate(outputs, size=args.orig_patch_size, 
                                    mode='trilinear', align_corners=False)

            dice_losses = []
            DICE_W = args.MAX_DICE_W
            
            # Put the class dimension as the last dimension (required by BCEWithLogitsLoss).
            total_ce_loss   = bce_loss_func(outputs.permute([0,2,3,4,1]), 
                                            # after brats_map_label(), dim 1 of mask_batch is segmantation class.
                                            # It's permuted to the last dim to align with outputs for bce loss computation.
                                            mask_batch.permute([0,2,3,4,1]))
            total_dice_loss = 0
            outputs_soft    = torch.sigmoid(outputs)
            
            for cls in range(1, args.num_classes):
                # bce_loss_func is actually nn.BCEWithLogitsLoss(), so use raw scores as input.
                # dice loss always uses sigmoid/softmax transformed probs as input.
                dice_loss = dice_loss_func(outputs_soft[:, cls], mask_batch[:, cls])
                dice_losses.append(dice_loss)
                total_dice_loss = total_dice_loss + dice_loss * class_weights[cls]

            if args.net == 'segtran' and args.use_attn_consist_loss:
                attn_consist_loss = attn_consist_loss_fun(net.layers_attn_scores, 
                                                          net.orig_feat_shape, mask_batch)
            
            loss = (1 - DICE_W) * total_ce_loss + DICE_W * total_dice_loss + args.ATTNCONSIST_W * attn_consist_loss

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            if args.distributed:
                total_ce_loss   = reduce_tensor(total_ce_loss.data)
                total_dice_loss = reduce_tensor(total_dice_loss.data)
                dice_losses     = [ reduce_tensor(dice_loss.data) for dice_loss in dice_losses ]
                loss            = reduce_tensor(loss.data)
                
            if is_master:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/total_ce_loss', total_ce_loss.item(), iter_num)
                writer.add_scalar('loss/total_dice_loss', total_dice_loss.item(), iter_num)
                writer.add_scalar('loss/loss', loss.item(), iter_num)
                log_str = '%d loss: %.4f, ce: %.4f, dice: %.4f' % \
                                    (iter_num, loss.item(), total_ce_loss.item(),
                                     total_dice_loss.item())
                if len(dice_losses) > 1:
                    dice_loss_str = ",".join( [ "%.4f" %dice_loss for dice_loss in dice_losses ] )
                    log_str += " (%s)" %dice_loss_str
                if attn_consist_loss > 0:
                    log_str += ", attcon %.3f" %attn_consist_loss
                logging.info(log_str)
                                     
            if iter_num % 50 == 0  and is_master:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
                        
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = mask_batch[0, 1:2, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if args.opt == 'sgd' and iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            if iter_num % args.saveiter == 0:
                save_model(real_net, optimizer, args, checkpoint_dir, iter_num)
            if iter_num >= args.maxiter:
                break
            time1 = time.time()
        if iter_num >= args.maxiter:
            break

    if args.maxiter % args.saveiter != 0:
        save_model(real_net, optimizer, args, checkpoint_dir, iter_num)

    if is_master:
        writer.close()
