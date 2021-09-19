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
import copy
import itertools

from thop import clever_format, profile

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid
from optimization import BertAdam

import segmentation_models_pytorch as smp
from networks.segtran2d import Segtran2d, set_segtran2d_config
from networks.segtran2d import CONFIG as config
from networks.segtran_shared import SqueezedAttFeatTrans
from networks.polyformer import Polyformer, PolyformerLayer
import networks.deeplab as deeplab
from networks.nested_unet import UNet, NestedUNet
from networks.unet_3plus.unet_3plus import UNet_3Plus
from networks.unet2d.unet_model import UNet as VanillaUNet
from networks.pranet.PraNet_Res2Net import PraNet
from networks.att_unet import AttU_Net, R2AttU_Net
from networks.deformable_unet.deform_unet import DUNetV1V2 as DeformableUNet
from networks.transunet.vit_seg_modeling import VisionTransformer as TransUNet
from networks.transunet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
from networks.discriminator import Discriminator
from utils.losses import dice_loss_indiv, dice_loss_mix, calc_vcdr
from dataloaders.datasets2d import fundus_map_mask, polyp_map_mask, reshape_mask, index_to_onehot
from common_util import AverageMeters, get_default, get_argument_list, get_filename, get_seg_colormap
from train_util import init_augmentation, init_training_dataset, freeze_bn

def print0(*print_args, **kwargs):
    if args.local_rank == 0:
        print(*print_args, **kwargs)

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task_name', type=str, default='fundus', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='ds_names', type=str, default=None, help='Dataset folders. Can specify multiple datasets (separated by ",")')
parser.add_argument('--split', dest='ds_split', type=str, default='all', 
                    help='Split of the dataset (Can specify the split individually for each dataset)')
parser.add_argument("--profile", dest='do_profiling', action='store_true', help='Calculate amount of params and FLOPs. ')                    

parser.add_argument('--insize', dest='orig_input_size', type=str, default=None, 
                    help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='patch_size', type=str, default=None, 
                    help='Resize input images to this size for training.')

###### BEGIN of few-shot learning arguments ######                    
parser.add_argument('--samplenum', dest='sample_num', type=str,  default=None, 
                    help='Numbers of supervised training samples to use for each dataset (Default: None, use all images of each dataset. '
                         'Provide 0 for a dataset to use all images of it. Do not use -1 as it will cause errors of argparse).')
parser.add_argument("--bnopt", dest='bn_opt_scheme', type=str, default=None,
                    choices=[None, 'fixstats', 'affine'],
                    help='How to optimize BN stats/affine params during training.')                  
                    
###### BEGIN of Polyformer arguments ######                    
parser.add_argument("--polyformer", dest='polyformer_mode', type=str, default=None,
                    choices=[None, 'none', 'source', 'target'],
                    help='Do polyformer traning.')
parser.add_argument("--sourceopt", dest='poly_source_opt', type=str, default='allpoly',
                    help='What params to optimize on the source domain.')
parser.add_argument("--targetopt", dest='poly_target_opt', type=str, default='k',
                    help='What params to optimize on the target domain.')
###### END of Polyformer arguments ######

###### BEGIN of adversarial training arguments ######
parser.add_argument("--adv", dest='adversarial_mode', type=str, default=None,
                    choices=[None, 'none', 'feat', 'mask'],
                    help='Mode of adversarial training.')
parser.add_argument("--featdisinchan", dest='num_feat_dis_in_chan', type=int, default=64,
                    help='Number of input channels of the feature discriminator')

parser.add_argument("--sourceds", dest='source_ds_names', type=str, default=None,
                    help='Dataset name of the source domain.')
parser.add_argument("--sourcebs", dest='source_batch_size', type=int, default=-1,
                    help='Batch size of unsupervised adversarial learning on the source domain (access all source domain images).')
parser.add_argument("--targetbs", dest='target_unsup_batch_size', type=int, default=-1,
                    help='Batch size of unsupervised adversarial learning on the target domain (access all target domain images).')
parser.add_argument('--domweight', dest='DOMAIN_LOSS_W', type=float, default=0.002, 
                    help='Weight of the adversarial domain loss.')      
parser.add_argument('--supweight', dest='SUPERVISED_W', type=float, default=1, 
                    help='Weight of the supervised training loss. Set to 0 to do unsupervised DA.')      
parser.add_argument('--reconweight', dest='RECON_W', type=float, default=0, 
                    help='Weight of the reconstruction loss for DA. Default: 0, no reconstruction.')      
parser.add_argument("--adda", dest='adda', action='store_true', 
                    help='Use ADDA (instead of the default RevGrad objective).')

###### END of adversarial training arguments ######

###### END of few-shot learning arguments ######
                 
###### Begin of Robustness experiment settings ######
parser.add_argument("--optfilter", dest='opt_filters', type=str, default=None,
                    help='Only optimize params that match the specified keyword.')
parser.add_argument("--robustaug", dest='robust_aug_types', type=str, default=None,
                    # Examples: None, 'brightness,contrast',
                    help='Augmentation types used during robustness training.')
parser.add_argument("--robustaugdeg", dest='robust_aug_degrees', type=str, default='0.5,1.5',
                    help='Degrees of robustness augmentation (1 or 2 numbers).')
parser.add_argument("--gbias", dest='use_global_bias', action='store_true', 
                    help='Use the global bias instead of transformer layers.')

###### End of Robustness experiment settings ######
   
parser.add_argument('--maxiter', type=int,  default=10000, help='maximum training iterations')
parser.add_argument('--saveiter', type=int,  default=500, help='save model snapshot every N iterations')

###### Begin of optimization settings ######
parser.add_argument('--lrwarmup', dest='lr_warmup_steps', type=int,  default=500, help='Number of LR warmup steps')
parser.add_argument('--dicewarmup', dest='dice_warmup_steps', type=int,  default=0, help='Number of dice warmup steps (0: disabled)')
parser.add_argument('--bs', dest='batch_size', type=int, default=6, help='Total batch_size on all GPUs')
parser.add_argument('--opt', type=str,  default=None, help='optimization algorithm')
parser.add_argument('--lr', type=float,  default=-1, help='learning rate')
parser.add_argument('--decay', type=float,  default=-1, help='weight decay')
parser.add_argument('--gradclip', dest='grad_clip', type=float,  default=-1, help='gradient clip')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')
parser.add_argument('--cp', dest='checkpoint_path', type=str,  default=None, help='Load this checkpoint')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument("--locprob", dest='localization_prob', default=0, 
                    type=float, help='Probability of doing localization during training')
parser.add_argument("--tunebn", dest='tune_bn_only', action='store_true', 
                    help='Only tune batchnorms for domain adaptation, and keep model weights unchanged.')

parser.add_argument('--diceweight', dest='MAX_DICE_W', type=float, default=0.5, 
                    help='Weight of the dice loss.')
parser.add_argument('--focus', dest='focus_class', type=int, default=-1, 
                    help='The class that is particularly predicted (with higher loss weight)')
parser.add_argument('--exclusive', dest='use_exclusive_masks', action='store_true', 
                    help='Aim to predict exclulsive masks (instead of non-exclusive ones)')
                    
parser.add_argument("--vcdr", dest='vcdr_estim_scheme', type=str, default='none',
                    choices=['none', 'dual', 'single'],
                    help='The scheme of the learned vCDR loss for fundus images. none: not using vCDR loss. '
                         'dual:   estimate vCDR with an individual vC estimator and vD estimator. '
                         'single: estimate vCDR directly using a single CNN.')

parser.add_argument("--vcdrweight", dest='VCDR_W', type=float, default=0.01,
                    help='Weight of vCDR loss.')
parser.add_argument("--vcdrestimstart", dest='vcdr_estim_loss_start_iter', type=int, default=1000,
                    help='Start iteration of vCDR loss for the vCDR estimator.')
# vCDR estimator usually converges very fast. So 100 iterations are enough.                    
parser.add_argument("--vcdrnetstart",   dest='vcdr_net_loss_start_iter',   type=int, default=1100,
                    help='Start iteration of vCDR loss for the segmentation model.')

###### End of optimization settings ######

parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')

parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')
parser.add_argument('--bb', dest='backbone_type', type=str,  default='eff-b4', 
                    help='Backbone of Segtran / Encoder of other models')
parser.add_argument("--nopretrain", dest='use_pretrained', action='store_false', 
                    help='Do not use pretrained weights.')

###### Begin of transformer architecture settings ######
parser.add_argument("--translayers", dest='num_translayers', default=1,
                    type=int, help='Number of Cross-Frame Fusion layers.')
parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=None, 
                    help='Compression ratio of channel numbers of each transformer layer to save RAM.')
parser.add_argument("--baseinit", dest='base_initializer_range', default=0.02,
                    type=float, help='Base initializer range of transformer layers.')

parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false', 
                    help='Do not use attractor transformers (Default: use to increase scalability).')
parser.add_argument("--attractors", dest='num_attractors', default=256,
                    type=int, help='Number of attractors in the squeezed transformer.')
parser.add_argument("--noqkbias", dest='qk_have_bias', action='store_false', 
                    help='Do not use biases in Q, K projections (Using biases leads to better performance on BraTS).')
                    
parser.add_argument('--pos', dest='pos_code_type', type=str, default='lsinu', 
                    choices=['lsinu', 'zero', 'rand', 'sinu', 'bias'],
                    help='Positional code scheme')
parser.add_argument('--posw', dest='pos_code_weight', type=float, default=1.0)
parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                    help='The radius of positional biases')
parser.add_argument('--perturbposw', dest='perturb_posw_range', type=float, default=0.,
                    help='The range of added random noise to pos_code_weight during training')
parser.add_argument("--poslayer1", dest='pos_code_every_layer', action='store_false', 
                    help='Only add pos code to the first transformer layer input (Default: add to every layer).')
parser.add_argument("--posattonly", dest='pos_in_attn_only', action='store_true', 
                    help='Only use pos embeddings when computing attention scores (K, Q), '
                         'and not use them in the input for V or FFN.')
parser.add_argument("--squeezeuseffn", dest='has_FFN_in_squeeze', action='store_true', 
                    help='Use the full FFN in the first transformer of the squeezed attention '
                         '(Default: only use the first linear layer, i.e., the V projection)')

parser.add_argument('--dropout', type=float, dest='dropout_prob', default=-1, help='Dropout probability')
parser.add_argument('--modes', type=int, dest='num_modes', default=-1, help='Number of transformer modes')
parser.add_argument('--modedim', type=int, dest='attention_mode_dim', default=-1, 
                    help='Dimension of transformer modes')
parser.add_argument('--multihead', dest='ablate_multihead', action='store_true', 
                    help='Ablation to multimode transformer (using multihead instead)')

###### End of transformer architecture settings ######

###### Begin of Segtran (non-transformer part) settings ######
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
###### End of Segtran (non-transformer part) settings ######

###### Begin of augmentation settings ######
# Using random scaling as augmentation usually hurts performance. Not sure why.
parser.add_argument("--randscale", type=float, default=0.2, help='Do random scaling augmentation.')
parser.add_argument("--affine", dest='do_affine', action='store_true', help='Do random affine augmentation.')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5, 
                    help='Convert images to grayscale by so much degree.')
parser.add_argument("--reshape", dest='reshape_mask_type', type=str, default=None, 
                    choices=[None, 'rectangle', 'ellipse'],
                    help='Intentionally reshape the mask to test how well the model fits the mask bias.')
###### End of augmentation settings ######

args_dict = {  'trans_output_type': 'private',
               'mid_type': 'shared',
               'in_fpn_scheme':     'AN',
               'out_fpn_scheme':    'AN',
            }

args = parser.parse_args()
for arg, v in args_dict.items():
    args.__dict__[arg] = v

if args.ablate_multihead:
    args.use_squeezed_transformer = False
if args.polyformer_mode == 'none':
    args.polyformer_mode = None
if args.adversarial_mode == 'none':
    args.adversarial_mode = None    
        
unet_settings    = { 'opt': 'adamw', 
                     'lr': { 'sgd': 0.01, 'adam': 0.001, 'adamw': 0.001 },
                     'decay': 0.0001, 'grad_clip': -1,
                   }
segtran_settings = { 'opt': 'adamw',
                     'lr': { 'adamw': 0.0002 },
                     'decay': 0.0001,  'grad_clip': 0.1,
                     'dropout_prob': { '234': 0.3, '34': 0.2, '4': 0.2 },
                     'num_modes':    { '234': 2,   '34': 4,   '4': 4 }
                   }

default_settings = { 'unet':            unet_settings,
                     'unet-scratch':    unet_settings,
                     'nestedunet':      unet_settings,
                     'unet3plus':       unet_settings,
                     'deeplabv3plus':   unet_settings,
                     'deeplab-smp':     unet_settings,
                     'pranet':          unet_settings,
                     'attunet':         unet_settings,
                     'r2attunet':       unet_settings,
                     'dunet':           unet_settings,
                     'nnunet':          unet_settings,
                     'setr':            segtran_settings,
                     'transunet':       segtran_settings,
                     'segtran':         segtran_settings,
                     'fundus': {
                                 'num_classes': 3,
                                 'bce_weight':  [0., 1, 2],
                                 'ds_class':    'SegCrop',
                                 'ds_names':    'train,valid,test,drishiti,rim',
                                 'orig_input_size': 576,
                                 # Each dim of the patch_size should be multiply of 32.
                                 'patch_size':      288,
                                 'uncropped_size': { 'train':    (2056, 2124), 
                                                     'test':     (1634, 1634),
                                                     'valid':    (1634, 1634),
                                                     'valid2':   (1940, 1940),
                                                     'test2':    -1,    # varying sizes
                                                     'drishiti': (2050, 1750),
                                                     'rim':      (2144, 1424),
                                                     'train-cyclegan':    (2056, 2124), 
                                                     'rim-cyclegan':      (2144, 1424),
                                                     'gamma-train':       -1, # varying sizes
                                                     'gamma-valid':       -1, # varying sizes
                                                     'gamma-test':        -1, # varying sizes
                                                   },
                                 'has_mask':    { 'train': True,    'test': True, 
                                                  'valid': True,    'valid2': False,
                                                  'test2': False,
                                                  'drishiti': True, 'rim': True, 
                                                  'train-cyclegan': True,
                                                  'rim-cyclegan': True,
                                                  'gamma-train':  True,
                                                  'gamma-valid':  False,
                                                  'gamma-test':   False },
                                 'weight':      { 'train': 1,       'test': 1, 
                                                  'valid': 1,       'valid2': 1,
                                                  'test2': 1,
                                                  'drishiti': 1,    'rim': 1,
                                                  'train-cyclegan': 1,
                                                  'rim-cyclegan': 1,
                                                  'gamma-train':  1,
                                                  'gamma-valid':  1,
                                                  'gamma-test':   1   },
                                 # if the uncropped_size of a dataset == -1, then its orig_dir 
                                 # has to be specified here for the script to acquire 
                                 # the uncropped_size of each image. 
                                 'orig_dir':    { 'test2': 'test2_orig',
                                                  'gamma-train': 'gamma_train_orig/images',
                                                  'gamma-valid': 'gamma_valid_orig/images',
                                                  'gamma-test':  'gamma_test_orig/images' },
                                 'orig_ext':    { 'test2': '.jpg',
                                                  'gamma-train': '.png',
                                                  'gamma-valid': '.jpg',
                                                  'gamma-test':  '.jpg'  },
                               },
                     'polyp':  {
                                 'num_classes': 2,
                                 'bce_weight':  [0., 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names': 'CVC-ClinicDB-train,Kvasir-train',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # So set it to the same as patch_size to avoid output upscaling.
                                 # Setting orig_input_size to -1 also leads to output_upscale = 1.
                                 # All images of different sizes are resized to 320*320.
                                 'orig_input_size': 320,    
                                 'patch_size':      320,
                                 'has_mask':    { 'CVC-ClinicDB-train': True,   'Kvasir-train': True, 
                                                  'CVC-ClinicDB-test': True,    'Kvasir-test': True, 
                                                  'CVC-300': True,              
                                                  'CVC-ClinicDB-train-cyclegan': True,
                                                  'CVC-300-cyclegan': True,
                                                  'CVC-ColonDB': False,
                                                  'ETIS-LaribPolypDB': True },
                                 'weight':      { 'CVC-ClinicDB-train': 1,      'Kvasir-train': 1, 
                                                  'CVC-ClinicDB-test': 1,       'Kvasir-test': 1, 
                                                  'CVC-300': 1,                 
                                                  'CVC-ClinicDB-train-cyclegan': 1,
                                                  'CVC-300-cyclegan': 1,
                                                  'CVC-ColonDB': 1,
                                                  'ETIS-LaribPolypDB': 1  }
                               },
                     'oct':  {
                                 'num_classes': 10,
                                 'bce_weight':  [0., 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names':    'duke',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # If you want to avoid avoid output upscaling, set orig_input_size to the same as patch_size.
                                 # The actual resolution of duke is (296, 500~542). 
                                 # Set to (288, 512) will crop the central areas.
                                 # The actual resolution of pcv is (633, 720). Removing 9 pixels doesn't matter.
                                 'orig_input_size': { 'duke': (288, 512), 'seed': (1024, 512), 'pcv': (624, 720) } ,
                                 'patch_size':      { 'duke': (288, 512), 'seed': (512,  256), 'pcv': (312, 360) }, 
                                 'has_mask':        { 'duke': True,       'seed': False,       'pcv': False },
                                 'weight':          { 'duke': 1,          'seed': 1,           'pcv': 1 }
                               },
                   }

get_default(args, 'orig_input_size',    default_settings, None,   [args.task_name, 'orig_input_size'])
get_default(args, 'patch_size',         default_settings, None,   [args.task_name, 'patch_size'])
if type(args.patch_size) == str:
    args.patch_size = get_argument_list(args.patch_size, int)
    if len(args.patch_size) == 1:
        args.patch_size = (args.patch_size[0], args.patch_size[0])    
if type(args.patch_size) == int:
    args.patch_size = (args.patch_size, args.patch_size)
if type(args.orig_input_size) == str:
    args.orig_input_size = get_argument_list(args.orig_input_size, int)
if type(args.orig_input_size) == int:
    args.orig_input_size = (args.orig_input_size, args.orig_input_size)
if args.orig_input_size[0] > 0:
    args.output_upscale = args.orig_input_size[0] / args.patch_size[0]
else:
    args.output_upscale = 1

get_default(args, 'ds_names',           default_settings, None, [args.task_name, 'ds_names'])

ds_stats_map = { 'fundus': 'fundus-cropped-gray{:.1f}-stats.json',
                 'polyp':  'polyp-whole-gray{:.1f}-stats.json',
                 'oct':    'oct-whole-gray{:.1f}-stats.json' }

stats_file_tmpl = ds_stats_map[args.task_name]
stats_filename = stats_file_tmpl.format(args.gray_alpha)
ds_stats = json.load(open(stats_filename))
default_settings[args.task_name].update(ds_stats)
print0("'{}' mean/std loaded from '{}'".format(args.task_name, stats_filename))

if args.opt_filters:
    args.opt_filters = get_argument_list(args.opt_filters, str)
if args.robust_aug_types:
    args.robust_aug_types = get_argument_list(args.robust_aug_types, str)
    
args.ds_names = get_argument_list(args.ds_names, str)
train_data_paths = []

for ds_name in args.ds_names:
    train_data_path = os.path.join("../data/", args.task_name, ds_name)
    train_data_paths.append(train_data_path)

args.job_name = '{}-{}'.format(args.task_name, ','.join(args.ds_names))
args.robust_aug_degrees = get_argument_list(args.robust_aug_degrees, float)
if len(args.robust_aug_degrees) == 1:
    args.robust_aug_degrees = args.robust_aug_degrees * 2
    
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

def init_optimizer(net, max_epoch, batches_per_epoch, args):
    # Prepare optimizer
    # Each param is a tuple ( param name, Parameter(tensor(...)) )
    if not args.polyformer_mode:
        if not args.opt_filters:
            optimized_params = list( param for param in net.named_parameters() if param[1].requires_grad )
        else:
            optimized_params = []
            for n, p in net.named_parameters():
                if any(filter_key in n for filter_key in args.opt_filters if p.requires_grad):
                    optimized_params.append((n, p))
    else:
        args.decay = 0
        
        if args.net == 'segtran':
            translayers = net.voxel_fusion.translayers
        else:
            translayers = net.polyformer.polyformer_layers
            assert type(translayers[0]) == PolyformerLayer
            
        if args.polyformer_mode == 'source':
            poly_opt_mode = args.poly_source_opt
        else:
            poly_opt_mode = args.poly_target_opt
            
        if poly_opt_mode == 'allnet':
            optimized_params = list( param for param in net.named_parameters() if param[1].requires_grad )
        else:
            poly_opt_modes = poly_opt_mode.split(",")
            optimized_params = []

            for poly_opt_mode in poly_opt_modes:
                if poly_opt_mode == 'allpoly':
                    optimized_params += [ translayers.named_parameters() ]
                elif poly_opt_mode == 'inator':
                    optimized_params += [ translayer.in_ator_trans.named_parameters() for translayer in translayers ]
                elif poly_opt_mode == 'k':
                    optimized_params += [ translayer.in_ator_trans.key.named_parameters()   for translayer in translayers ]
                elif poly_opt_mode == 'v':
                    optimized_params += [ translayer.in_ator_trans.out_trans.first_linear.named_parameters() for translayer in translayers ]
                elif poly_opt_mode == 'q':
                    optimized_params += [ translayer.in_ator_trans.query.named_parameters() for translayer in translayers ]
                # optimize the segmentation head as well.
                elif poly_opt_mode == 'h':
                    # Only for VanillaUNet
                    assert args.net == 'unet-scratch'
                    optimized_params += [ net.outc.named_parameters() ]
            
            # Combine a list of lists of parameters into one list.
            optimized_params = list(itertools.chain.from_iterable(optimized_params))

            if net.discriminator and not args.adda:
                optimized_params += list(net.discriminator.named_parameters())
            if net.recon:
                optimized_params += list(net.recon.named_parameters())
                
    if args.bn_opt_scheme == 'affine':
        bn_params = []
        for layer in net.modules():
            # Also fine-tune LayerNorms when their elementwise_affine=True 
            if isinstance(layer, nn.BatchNorm2d): 
                bn_params.extend(list(layer.named_parameters()))
        optimized_params += bn_params
            
    low_decay = ['backbone'] #['bias', 'LayerNorm.weight']
    no_decay = []
    high_lr = ['alphas']
    
    high_lr_params = []
    high_lr_names = []
    no_decay_params = []
    no_decay_names = []
    low_decay_params = []
    low_decay_names = []
    normal_params = []
    normal_names = []

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
        
    args.t_total = int(batches_per_epoch * max_epoch)
    print0("Batches per epoch: %d" % batches_per_epoch)
    print0("Total Iters: %d" % args.t_total)
    print0("LR: %f" %args.lr)
    
    args.lr_warmup_steps = min(args.lr_warmup_steps, args.t_total // 2)
    args.lr_warmup_ratio = args.lr_warmup_steps / args.t_total
    print0("LR Warm up: %.3f=%d iters" % (args.lr_warmup_ratio, args.lr_warmup_steps))

    # pytorch adamw performs much worse. Not sure about the reason.
    optimizer = BertAdam(optimizer_grouped_parameters,
                         warmup=args.lr_warmup_ratio, t_total=args.t_total,
                         weight_decay=args.decay)
                         
    return optimizer

def load_model(net, optimizer, args, checkpoint_path, load_optim_state):
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        optim_state_dict = state_dict['optim_state']
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
                     'patch_size', 'orig_input_size', 'output_upscale',
                     'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                     'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'ds_split', 'ds_names',
                     'job_name', 'mean', 'std', 'mask_thres', ]
    warn_keys = [ 'num_recurrences' ]
                        
    if args.net == 'segtran' and cp_args is not None:
        for k in cp_args:
            if (k in warn_keys) and (args.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args.__dict__[k], k, cp_args[k]))
                continue

            if (k not in ignored_keys) and (args.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args.__dict__[k], k, cp_args[k]))
                exit(0)
                    
    params.update(model_state_dict)
    net.load_state_dict(params)
    if load_optim_state and optim_state_dict is not None:
        optimizer.load_state_dict(optim_state_dict)
        # warmup info is mainly in optim_state_dict. So after loading the checkpoint, 
        # the optimizer won't do warmup already.
        args.lr_warmup_steps = 0
        print0("LR Warm up reset to 0 iters.")
                            
    logging.info("Model loaded from '{}'".format(checkpoint_path))
    return cp_iter_num
    
def save_model(net, optimizer, args, checkpoint_dir, iter_num):
    if args.local_rank == 0:
        save_model_path = os.path.join(checkpoint_dir, 'iter_'+str(iter_num)+'.pth')
        torch.save( { 'iter_num': iter_num, 'model': net.state_dict(), 
                      'optim_state': optimizer.state_dict(),
                      'args': vars(args) },  
                    save_model_path)
                        
        logging.info("save model to '{}'".format(save_model_path))
                        
def warmup_constant(x, warmup=500):
    if x < warmup:
        return x/warmup
    return 1

def estimate_vcdr(args, net, x):
    if args.vcdr_estim_scheme == 'sep':
        vc_pred     = net.vc_estim(x)
        vd_pred     = net.vd_estim(x)
        vcdr_pred   = vc_pred / (vd_pred + 1e-6)
    else:
        vcdr_pred   = net.vcdr_estim(x)
    
    vcdr_pred = vcdr_pred.sigmoid().squeeze(1)
    return vcdr_pred
    
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
    
    if args.polyformer_mode:
        print0("Do polyformer training")
        if args.polyformer_mode == 'source':
            args.tie_qk_scheme = 'shared'
        else:
            # on target domain, decouple q and k.
            args.tie_qk_scheme = 'loose'
    else:
        args.tie_qk_scheme = 'shared'
        
    is_master = (args.local_rank == 0)
    n_gpu = torch.cuda.device_count()
    args.device = 'cuda'
    args.distributed = False
    # 'WORLD_SIZE' is set by 'torch.distributed.launch'. Do not set manually
    # Its value is specified by "--nproc_per_node=k"
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1

    # Avoid sampling the same image repetitively in the same batch.
    if args.sample_num:
        args.sample_num = get_argument_list(args.sample_num, int)
        args.sample_total_num = np.sum(args.sample_num)
        # A (supervised) batch shouldn't be larger than the total number of samples. 
        # To avoid sampling images repetitively when the training is purely supervised.
        args.batch_size = min(args.batch_size, args.sample_total_num)
        # If sample_total_num is set to 0, then dataloader initialization will raise an exception.
        args.batch_size = max(args.batch_size, 2)
    else:
        args.sample_num = [ -1 for ds_path in train_data_paths ]
    
    # Specify dataset split individually for each dataset.    
    if ',' in args.ds_split:
        args.ds_split = args.ds_split.split(",")
            
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
    get_default(args, 'num_classes',    default_settings, None,   [args.task_name, 'num_classes'])
    args.binarize = (args.num_classes == 2)

    get_default(args, 'bce_weight',     default_settings, None,   [args.task_name, 'bce_weight'])
    get_default(args, 'ds_class',       default_settings, None,   [args.task_name, 'ds_class'])
    
    args.bce_weight = torch.tensor(args.bce_weight).cuda()
    args.bce_weight = args.bce_weight * (args.num_classes - 1) / args.bce_weight.sum()
            
    if args.net == 'segtran':
        get_default(args, 'dropout_prob',   default_settings, -1, [args.net, 'dropout_prob', args.in_fpn_layers])
        get_default(args, 'num_modes',      default_settings, -1, [args.net, 'num_modes', args.in_fpn_layers])

    # common_aug_func: flip, shift, crop...
    # image_aug_func:  ColorJitter. robust_aug_funcs: ColorJitter.
    common_aug_func, image_aug_func, robust_aug_funcs = init_augmentation(args)
    db_trains = []
    db_target_unsup_trains = []
    ds_settings = default_settings[args.task_name]
    
    for i, train_data_path in enumerate(train_data_paths):
        ds_name  = args.ds_names[i]
        if isinstance(args.ds_split, list):
            ds_split = args.ds_split[i]
        else:
            ds_split = args.ds_split
            
        db_train = init_training_dataset(args, ds_settings, ds_name, ds_split, train_data_path, args.sample_num[i],
                                         common_aug_func, image_aug_func, robust_aug_funcs)
        db_trains.append(db_train)
        
        if args.adversarial_mode:
            # Use all data in the target domain for unsupervised adversarial training.
            db_target_unsup_train = init_training_dataset(args, ds_settings, ds_name, 'all', train_data_path, -1,
                                                   common_aug_func, image_aug_func, robust_aug_funcs)
            db_target_unsup_trains.append(db_target_unsup_train)
        
    db_train_combo = ConcatDataset(db_trains)
    print0("Combined supervised dataset: {} images".format(len(db_train_combo)))
    
    if args.adversarial_mode:
        db_target_unsup_combo = ConcatDataset(db_target_unsup_trains)
        print0("Combined unsupervised dataset: {} images".format(len(db_target_unsup_combo)))
        
    # num_modalities is used in segtran.
    # num_modalities = 0 means there's not the modality dimension 
    # (but still a single modality) in the images loaded from db_train.
    args.num_modalities = 0
    if args.translayer_compress_ratios is not None:
        args.translayer_compress_ratios = get_argument_list(args.translayer_compress_ratios, float)
    else:
        args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers + 1) ]
    
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

    args.num_workers = 0 if args.debug else 4
    train_loader = DataLoader(db_train_combo, batch_size=args.batch_size, sampler=train_sampler,
                             num_workers=args.num_workers, pin_memory=False, shuffle=shuffle,
                             worker_init_fn=worker_init_fn)
    if args.adversarial_mode:
        if args.adversarial_mode == 'mask':
            args.num_dis_in_chan = args.num_classes
        else:
            args.num_dis_in_chan = args.num_feat_dis_in_chan
            
        discriminator    = Discriminator(args.num_dis_in_chan, num_classes=1, do_revgrad=(not args.adda))
        
        if args.source_ds_names is None:
            get_default(args, 'source_ds_names', default_settings, None, [args.task_name, 'ds_names'])
        args.source_ds_names = args.source_ds_names.split(",")
        db_sources = []
        for source_ds_name in args.source_ds_names:
            source_data_path = os.path.join("../data/", args.task_name, source_ds_name)
            db_source        = init_training_dataset(args, ds_settings, source_ds_name, 'all', source_data_path, -1,
                                                     common_aug_func, image_aug_func, robust_aug_funcs)
            db_sources.append(db_source)
        
        db_source = ConcatDataset(db_sources)
                                                  
        if args.distributed:
            source_sampler = DistributedSampler(db_source, shuffle=True, 
                                                num_replicas=args.world_size, 
                                                rank=args.local_rank)
            target_unsup_sampler    = DistributedSampler(db_target_unsup_combo, shuffle=True, 
                                                         num_replicas=args.world_size, 
                                                         rank=args.local_rank)
            shuffle = False
        else:
            source_sampler = None
            target_unsup_sampler  = None
            shuffle = True
        
        if args.source_batch_size < 0:
            args.source_batch_size = args.batch_size
        if args.target_unsup_batch_size < 0:
            args.target_unsup_batch_size = args.batch_size
        source_loader  = DataLoader(db_source, batch_size=args.source_batch_size, sampler=source_sampler,
                                    num_workers=args.num_workers, pin_memory=False, shuffle=shuffle,
                                    worker_init_fn=worker_init_fn)
        target_unsup_loader = DataLoader(db_target_unsup_combo, batch_size=args.target_unsup_batch_size, sampler=target_unsup_sampler,
                                         num_workers=args.num_workers, pin_memory=False, shuffle=shuffle,
                                         worker_init_fn=worker_init_fn)
        
    else:
        discriminator = None
        
    if args.RECON_W > 0:
        recon = nn.Conv2d(args.num_feat_dis_in_chan, 3, kernel_size=1)
    else:
        recon = None
        
    max_epoch = math.ceil(args.maxiter / len(train_loader))
    
    logging.info(str(args))
    base_lr = args.lr

    if args.net == 'unet':
        # timm-efficientnet performs slightly worse.
        backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
        net = smp.Unet(backbone_type, classes=args.num_classes, encoder_weights='imagenet' if args.use_pretrained else None)
    elif args.net == 'unet-scratch':
        # net = UNet(num_classes=args.num_classes)
        net = VanillaUNet(n_channels=3, num_classes=args.num_classes, 
                          use_polyformer=args.polyformer_mode, 
                          num_polyformer_layers=args.num_translayers,
                          num_attractors=args.num_attractors,
                          num_modes=args.num_modes,
                          tie_qk_scheme=args.tie_qk_scheme)
    elif args.net == 'nestedunet':
        net = NestedUNet(num_classes=args.num_classes)
    elif args.net == 'unet3plus':
        net = UNet_3Plus(num_classes=args.num_classes)
    elif args.net == 'pranet':
        net = PraNet(num_classes=args.num_classes - 1)
    elif args.net == 'attunet':
        net = AttU_Net(output_ch=args.num_classes)
    elif args.net == 'r2attunet':
        net = R2AttU_Net(output_ch=args.num_classes)
    elif args.net == 'dunet':
        net = DeformableUNet(n_channels=3, n_classes=args.num_classes)
    elif args.net == 'setr':
        # Install mmcv first: 
        # pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu{Your CUDA Version}/torch{Your Pytorch Version}/index.html
        # E.g.: pip install mmcv-full==1.2.2 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.7.1/index.html
        from mmcv.utils import Config
        sys.path.append("networks/setr")
        from mmseg.models import build_segmentor
        
        task2config = { 'fundus': 'SETR_PUP_288x288_10k_fundus_context_bs_4.py', 
                        'polyp':  'SETR_PUP_320x320_10k_polyp_context_bs_4.py',
                        'profile': 'SETR_PUP_256x256_10k_profile_context_bs_4.py' }
        if args.do_profiling:
            task_name = 'profile'
        else:
            task_name = args.task_name
        setr_cfg = Config.fromfile("networks/setr/configs/SETR/{}".format(task2config[task_name]))
        net = build_segmentor(setr_cfg.model, train_cfg=setr_cfg.train_cfg, test_cfg=setr_cfg.test_cfg)
        # By default, net() calls forward_train(), which receives extra parameters, and returns losses.
        # net.forward_dummy() receives/returns the traditional input/output.
        # Relevant file: mmseg/models/segmentors/encoder_decoder.py
        net.forward = net.forward_dummy
    elif args.net == 'transunet':
        transunet_config = TransUNet_CONFIGS[args.backbone_type]
        transunet_config.n_classes = args.num_classes
        if args.backbone_type.find('R50') != -1:
            # The "patch" in TransUNet means grid-like patches of the input image.
            # The "patch" in our code means the whole input image after cropping/resizing (part of the augmentation)
            transunet_config.patches.grid = (int(args.patch_size[0] / transunet_config.patches.size[0]), 
                                             int(args.patch_size[1] / transunet_config.patches.size[1]))
        net = TransUNet(transunet_config, img_size=args.patch_size, num_classes=args.num_classes)
        if args.use_pretrained and args.checkpoint_path is None:
            net.load_from(weights=np.load(transunet_config.pretrained_path))
                
    elif args.net.startswith('deeplab'):
        use_smp_deeplab = args.net.endswith('smp')
        if use_smp_deeplab:
            backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
            net = smp.DeepLabV3Plus(backbone_type, classes=args.num_classes, encoder_weights='imagenet' if args.use_pretrained else None)
        else:
            model_name = args.net + "_" + args.backbone_type
            model_map = {
                'deeplabv3_resnet50':       deeplab.deeplabv3_resnet50,
                'deeplabv3plus_resnet50':   deeplab.deeplabv3plus_resnet50,
                'deeplabv3_resnet101':      deeplab.deeplabv3_resnet101,
                'deeplabv3plus_resnet101':  deeplab.deeplabv3plus_resnet101,
                'deeplabv3_mobilenet':      deeplab.deeplabv3_mobilenet,
                'deeplabv3plus_mobilenet':  deeplab.deeplabv3plus_mobilenet
            }
            net = model_map[model_name](num_classes=args.num_classes, output_stride=8)
    
    elif args.net == 'nnunet':
        from nnunet.network_architecture.generic_UNet import Generic_UNet
        from nnunet.network_architecture.initialization import InitWeights_He
        net = Generic_UNet(
                            input_channels=3,
                            base_num_features=32,
                            num_classes=args.num_classes,
                            num_pool=7,
                            num_conv_per_stage=2,
                            feat_map_mul_on_downscale=2,
                            norm_op=nn.InstanceNorm2d,
                            norm_op_kwargs={'eps': 1e-05, 'affine': True},
                            dropout_op_kwargs={'p': 0, 'inplace': True},
                            nonlin_kwargs={'negative_slope': 0.01, 'inplace': True},
                            final_nonlin=(lambda x: x),
                            weightInitializer=InitWeights_He(1e-2),
                            pool_op_kernel_sizes=[[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
                            conv_kernel_sizes=([[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]),
                            upscale_logits=False,
                            convolutional_pooling=True,
                            convolutional_upsampling=True,
                           )
        net.inference_apply_nonlin = (lambda x: F.softmax(x, 1))

    elif args.net == 'segtran':
        set_segtran2d_config(args)
        net = Segtran2d(config)
    else:
        breakpoint()

    if args.task_name == 'fundus' and args.vcdr_estim_scheme is not 'none':
        # "Abuse" domain discriminator CNN as the vCDR estimation CNN.
        if args.vcdr_estim_scheme == 'sep':
            net.vc_estim    = Discriminator(num_in_chan=3, num_classes=1, do_avgpool=True, do_revgrad=False)
            net.vd_estim    = Discriminator(num_in_chan=3, num_classes=1, do_avgpool=True, do_revgrad=False)
        else:
            net.vcdr_estim  = Discriminator(num_in_chan=3, num_classes=1, do_avgpool=True, do_revgrad=False)
    else:
        args.vcdr_estim_scheme = None
        
    net.discriminator = discriminator
    net.recon = recon    
    net.cuda()

    if args.do_profiling:
        S = args.patch_size[0]   # input image size
        input = torch.randn(1, 3, S, S).cuda()
        macs, params = profile(net, inputs=(input, ))
        macs, params = clever_format([macs, params], "%.3f")
        print("Params: {}, FLOPs: {}".format(params, macs))
        
        start = time.time()
        for i in range(20):
            with torch.no_grad():
                input = torch.randn(10, 3, S, S).cuda()
                result = net(input)
        end = time.time()
        print("FPS: {:.1f}".format(200 / (end - start) ))
        exit(0)
    
    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.opt == 'adamw':
        optimizer = init_optimizer(net, max_epoch, len(train_loader), args)
        if args.adda:
            discriminator_optim = BertAdam(net.discriminator.parameters(), lr=args.lr,
                                           warmup=args.lr_warmup_ratio, t_total=args.t_total,
                                           weight_decay=args.decay)
                        
    if args.checkpoint_path is not None:
        load_optim_state = (args.polyformer_mode is None) and (args.opt_filters is None) and (args.adversarial_mode is None)
        iter_num = load_model(net, optimizer, args, args.checkpoint_path, load_optim_state)
        continue_iter = False
        if args.polyformer_mode or not continue_iter:
            iter_num = 0
            start_epoch = 0
        else:
            start_epoch = math.ceil(iter_num / len(train_loader))
        logging.info("Start epoch/iter: {}/{}".format(start_epoch, iter_num))
    else:
        iter_num = 0
        start_epoch = 0

    if args.tune_bn_only:
        net.eval()
        if args.backbone_type.startswith('eff'):
            for idx, block in enumerate(net.backbone._blocks):
                # Stops at layer 3. Layers 0, 1, 2 are set to training mode (BN tunable).
                if idx == net.backbone.endpoint_blk_indices[3]:
                    print0("Tuning stops at block {} in backbone '{}'".format(idx, args.backbone_type))
                    break
                block.train()
        else:
            print0("Backbone '{}' not supported.".format(args.backbone_type))
            exit(0)
    elif args.bn_opt_scheme == 'fixstats':
        net.apply(freeze_bn)
    else:
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
    logging.info("{} epochs, {} itertations each.".format(max_epoch, len(train_loader)))

    dice_loss_func = dice_loss_indiv
    class_weights = torch.ones(args.num_classes).cuda()
    class_weights[0] = 0
    if args.focus_class != -1 and args.num_classes > 2:
        class_weights[args.focus_class] = 2
    class_weights /= class_weights.sum()
    bce_loss_func = nn.BCEWithLogitsLoss(pos_weight=args.bce_weight)
    unweighted_bce_loss_func = nn.BCEWithLogitsLoss()
    if args.adversarial_mode:
        source_loader_iter          = iter(source_loader)
        target_unsup_loader_iter    = iter(target_unsup_loader)
        
    for epoch_num in tqdm(range(start_epoch, max_epoch), ncols=70, disable=(args.local_rank != 0)):
        print0()
        time1 = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch_num)
        
        # https://discuss.pytorch.org/t/how-could-i-reset-dataloader-or-count-data-batch-with-iter-instead-of-epoch/22902/4
        # Internally every time you call enumerate(data_loader), it will call the object's 
        # iter method which will create and return a new _DataLoaderIter object, 
        # which internally calls iter() on your batch sampler here.
        for i_batch, sampled_batch in enumerate(train_loader):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            image_batch, mask_batch = sampled_batch['image'].cuda(), sampled_batch['mask'].cuda()
            weights_batch = sampled_batch['weight'].cuda()
            
            # SUP_B: supervised batch size.
            SUP_B = len(image_batch)
            if args.adversarial_mode:
                try:
                    source_batch        = next(source_loader_iter)
                except StopIteration:
                    source_loader_iter  = iter(source_loader)
                    source_batch        = next(source_loader_iter)
                source_image_batch = source_batch['image'].cuda()
                
                try:
                    target_unsup_batch         = next(target_unsup_loader_iter)
                except StopIteration:
                    target_unsup_loader_iter   = iter(target_unsup_loader)
                    target_unsup_batch         = next(target_unsup_loader_iter)
                target_unsup_image_batch  = target_unsup_batch['image'].cuda()
                if args.SUPERVISED_W > 0:
                    image_batch = torch.cat([ image_batch, target_unsup_image_batch ], dim=0)
                else:
                    # Only use unsupervised images to save RAM.
                    image_batch = target_unsup_image_batch
                    
            if args.task_name == 'fundus':
                # after mapping, mask_batch is already float.
                mask_batch              = fundus_map_mask(mask_batch, exclusive=args.use_exclusive_masks)
            elif args.task_name == 'polyp':
                mask_batch              = polyp_map_mask(mask_batch)
            elif args.task_name == 'oct':
                mask_batch              = index_to_onehot(mask_batch, args.num_classes)
                
            # args.patch_size is typically set as 1/2 of args.orig_input_size.
            # Scale down the input images to save RAM. 
            # The mask tensor is kept the original size. The output segmap will 
            # be scaled up by a factor of args.output_upscale.
            image_batch = F.interpolate(image_batch, size=args.patch_size,
                                        mode='bilinear', align_corners=False)
            if args.adversarial_mode:
                source_image_batch = F.interpolate(source_image_batch, size=args.patch_size,
                                                   mode='bilinear', align_corners=False)
                
            iter_num = iter_num + 1
            
            # image_batch: [4, 3, 288, 288]
            # mask_batch:  [4, 3, 576, 576]
            # outputs:     [4, 3, 288, 288]
            # If args.tune_bn_only, only tune backbone BNs. 
            # Transformer group norms do not have running stats.
            if args.tune_bn_only:
                with torch.no_grad():
                    outputs = net(image_batch)

                if iter_num % 50 == 0:
                    save_model(real_net, optimizer, args, checkpoint_dir, iter_num)                    
                continue

            outputs = net(image_batch)
            
            if args.net == 'pranet':
                # Use lateral_map_2 for single-loss training.
                # Outputs is missing one channel (background). 
                # As the background doesn't incur any loss, its value doesn't matter. 
                # So add an all-zero channel to it.
                outputs0 = outputs[3]
                background = torch.zeros_like(outputs0[:, [0]])
                outputs = torch.cat([background, outputs0], dim=1)

            if args.net == 'nnunet':
                outputs = outputs[0]
                
            outputs = F.interpolate(outputs, size=mask_batch.shape[2:], 
                                    mode='bilinear', align_corners=False)
            outputs_soft = torch.sigmoid(outputs)
                                                            
            dice_losses = []
            DICE_W = args.MAX_DICE_W # * warmup_constant(iter_num, args.dice_warmup_steps)
            
            # Only compute supervised losses on the SUP_B images, i.e., the images with supervision.
            if args.SUPERVISED_W > 0:
                # BCEWithLogitsLoss uses raw scores, so use outputs here instead of outputs_soft.
                # Permute the class dimension to the last dimension (required by BCEWithLogitsLoss).
                # In both outputs and mask_batch, the class dimension is 1. 
                # So we do the same permutation to mask_batch as well.
                total_ce_loss   = bce_loss_func(outputs.permute([0, 2, 3, 1])[:SUP_B], 
                                                mask_batch.permute([0, 2, 3, 1]))
                total_dice_loss = 0

                for cls in range(1, args.num_classes):
                    # bce_loss_func is actually nn.BCEWithLogitsLoss(), so use raw scores as input.
                    # dice loss always uses sigmoid/softmax transformed probs as input.
                    dice_loss = dice_loss_func(outputs_soft[:SUP_B, cls], mask_batch[:, cls])
                    dice_losses.append(dice_loss)
                    total_dice_loss = total_dice_loss + dice_loss * class_weights[cls]
            else:
                total_ce_loss   = torch.zeros(1, device='cuda')
                total_dice_loss = torch.zeros(1, device='cuda')

            # Recon has to be done before adversarial DA, 
            # because in adversarial DA, net receives a new input batch and updates net.feature_maps[-1]
            if args.RECON_W > 0:
                reconed_input   = net.recon(net.feature_maps[-1])
                recon_loss      = nn.functional.mse_loss(image_batch, reconed_input)
            else:
                recon_loss      = 0
                
            if args.adversarial_mode:
                target_feat     = net.feature_maps[-1]
                source_outputs  = net(source_image_batch)
                source_feat     = net.feature_maps[-1]
                mix_batch_size  = len(image_batch) + len(source_image_batch)
                domain_labels   = torch.ones((mix_batch_size, 1), device=args.device)
                domain_labels[:len(source_image_batch)] = 0

                if args.adversarial_mode == 'feat':
                    mix_dom_feat        = torch.cat([source_feat, target_feat], dim=0)
                    
                elif args.adversarial_mode == 'mask':
                    source_outputs      = F.interpolate(source_outputs, size=mask_batch.shape[2:], 
                                                        mode='bilinear', align_corners=False)   
                    source_outputs_soft = torch.sigmoid(source_outputs)
                    mix_dom_feat        = torch.cat([source_outputs_soft, outputs_soft], dim=0)
                
                domain_scores   = net.discriminator(mix_dom_feat)
                domain_loss     = unweighted_bce_loss_func(domain_scores, domain_labels)
                if args.adda:
                    discriminator_optim.zero_grad()
                    domain_loss.backward(retain_graph=True)
                    discriminator_optim.step()
                    # Invert labels, and recompute domain_loss for the optimization of the generator.
                    domain_loss = unweighted_bce_loss_func(domain_scores, 1 - domain_labels)

            else:
                domain_loss = 0

            if args.vcdr_estim_scheme:
                if iter_num >= args.vcdr_estim_loss_start_iter:
                    # vcdr_pred_hard, vcdr_gt:  [6]
                    vcdr_pred_hard          = calc_vcdr(outputs_soft)
                    # vcdr_pred_scores_nograd and vcdr_pred_nograd won't BP grads to net. Only optimize vcdr_estim.
                    # vcdr_pred_nograd, vcdr_pred: [6]
                    vcdr_pred_nograd        = estimate_vcdr(args, net, outputs_soft.data)
                    # mask_batch, outputs_soft: [6, 3, 576, 576]
                    # vcdr_estim_loss only optimizes vcdr_estim, making its estimation of 
                    # (vcdr_pred ~ vcdr_pred_hard) more accurate.
                    vcdr_estim_loss         = torch.abs(vcdr_pred_nograd - vcdr_pred_hard).mean()
                    # vcdr_net_loss optimizes both net and vcdr_estim, making their estimation of
                    # (vcdr_pred ~ vcdr_gt) more accurate.
                    if iter_num >= args.vcdr_net_loss_start_iter:
                        vcdr_gt         = calc_vcdr(mask_batch)
                        vcdr_pred       = estimate_vcdr(args, net, outputs_soft)
                        vcdr_net_loss   = torch.abs(vcdr_pred - vcdr_gt).mean()
                    else:
                        vcdr_net_loss   = 0
                        
                    vcdr_loss           = vcdr_estim_loss + vcdr_net_loss
                else:
                    vcdr_estim_loss = vcdr_net_loss = vcdr_loss = 0
            else:
                vcdr_estim_loss = vcdr_net_loss = vcdr_loss = 0
                
            supervised_loss = (1 - DICE_W) * total_ce_loss + DICE_W * total_dice_loss \
                              + args.VCDR_W * vcdr_loss
            unsup_loss      = args.DOMAIN_LOSS_W   * domain_loss + args.RECON_W * recon_loss
                              
            loss = args.SUPERVISED_W * supervised_loss + unsup_loss
            
            optimizer.zero_grad()
            loss.backward()
            
            if args.polyformer_mode and args.net == 'segtran':
                weight_tensors = []
                for i, translayer in enumerate(net.voxel_fusion.translayers):
                    weight_tensors.append(translayer.in_ator_trans.key.weight.data.clone())
                    weight_grad = translayer.in_ator_trans.key.weight.grad.abs().sum()
                    print("%d weight grad: %.6f" %(i, weight_grad))
                
            #breakpoint()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            optimizer.step()

            if args.polyformer_mode and args.net == 'segtran':
                for i, translayer in enumerate(net.voxel_fusion.translayers):
                    weight_diff = (translayer.in_ator_trans.key.weight - weight_tensors[i]).abs().sum()
                    print("%d weight diff: %.6f" %(i, weight_diff))
                    
            if args.distributed:
                total_ce_loss   = reduce_tensor(total_ce_loss.data)
                total_dice_loss = reduce_tensor(total_dice_loss.data)
                dice_losses     = [ reduce_tensor(dice_loss.data) for dice_loss in dice_losses ]
                loss            = reduce_tensor(loss.data)
                
                if isinstance(vcdr_estim_loss, torch.Tensor):
                    vcdr_estim_loss = reduce_tensor(vcdr_estim_loss.data)
                if isinstance(vcdr_net_loss, torch.Tensor):
                    vcdr_net_loss   = reduce_tensor(vcdr_net_loss.data)
                    
            if is_master:
                writer.add_scalar('lr', lr_, iter_num)
                writer.add_scalar('loss/total_ce_loss', total_ce_loss.item(), iter_num)
                writer.add_scalar('loss/total_dice_loss', total_dice_loss.item(), iter_num)
                writer.add_scalar('loss/loss', loss.item(), iter_num)

                log_str = '%d loss %.3f, ce %.3f, dice %.3f' % \
                                (iter_num, loss.item(), total_ce_loss.item(),
                                 total_dice_loss.item())
                if len(dice_losses) > 1:
                    dice_loss_str = ",".join( [ "%.3f" %dice_loss for dice_loss in dice_losses ] )
                    log_str += " (%s)" %dice_loss_str
                if domain_loss > 0:
                    log_str += ", domain %.3f" %(domain_loss)
                if recon_loss > 0:
                    log_str += ", recon %.3f" %(recon_loss)
                if vcdr_loss > 0:
                    log_str += ", vcdr %.3f/%.3f" %(vcdr_estim_loss, vcdr_net_loss)
                    
                logging.info(log_str)
                                     
            if iter_num % 50 == 0 and is_master:
                grid_image = make_grid(image_batch, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)
                
                # make_grid can only handle <= 3 classes
                grid_image = make_grid(outputs_soft[:, :3], 5, normalize=False)
                writer.add_image('train/Predicted_mask', grid_image, iter_num)

                grid_image = make_grid(mask_batch[:, :3], 5, normalize=False)
                writer.add_image('train/Groundtruth_mask', grid_image, iter_num)

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
