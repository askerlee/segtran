import os
import time
import re
import sys
from datetime import datetime
import copy
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from networks.segtran2d import Segtran2d, set_segtran2d_config
import networks.segtran_shared as segtran_shared
from networks.segtran_shared import SqueezedAttFeatTrans
from networks.polyformer import PolyformerLayer
from networks.segtran2d import CONFIG as config
import networks.deeplab as deeplab
from networks.nested_unet import UNet, NestedUNet
from networks.unet_3plus.unet_3plus import UNet_3Plus
from networks.unet2d.unet_model import UNet as VanillaUNet
from networks.pranet.PraNet_Res2Net import PraNet
from networks.att_unet import AttU_Net, R2AttU_Net
from networks.deformable_unet.deform_unet import DUNetV1V2 as DeformableUNet
from networks.transunet.vit_seg_modeling import VisionTransformer as TransUNet
from networks.transunet.vit_seg_modeling import CONFIGS as TransUNet_CONFIGS
from test_util2d import test_all_cases, remove_fragmentary_segs
import dataloaders.datasets2d
from dataloaders.datasets2d import fundus_map_mask, fundus_inv_map_mask, polyp_map_mask, \
                                   polyp_inv_map_mask, reshape_mask, index_to_onehot, onehot_inv_map
import imgaug.augmenters as iaa
from common_util import get_default, get_argument_list, get_filename, get_seg_colormap
from internal_util import visualize_model, eval_robustness
from functools import partial
import subprocess
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task_name', type=str, default='fundus', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='ds_name', type=str, default='valid2', help='Dataset name for test')
parser.add_argument('--split', dest='ds_split', type=str, default='all',
                    choices=['train', 'test', 'all'], help='Split of the dataset')
parser.add_argument('--samplenum', dest='sample_num', type=int,  default=-1, 
                    help='Numbers of samples in the dataset split used to train the model (Default: -1, no images are used in the dataset).')
parser.add_argument('--cpdir', dest='checkpoint_dir', type=str, default=None,
                    help='Load checkpoint(s) from this directory')
parser.add_argument('--iters', type=str,  default='8000,7000', help='checkpoint iteration(s)')
parser.add_argument('--bs', dest='batch_size', type=int, default=8, help='batch size')
parser.add_argument('--exclusive', dest='use_exclusive_masks', action='store_true', 
                    help='Aim to predict exclulsive masks (instead of non-exclusive ones)')

parser.add_argument("--gbias", dest='use_global_bias', action='store_true', 
                    help='Use the global bias instead of transformer layers.')
parser.add_argument("--polyformer", dest='polyformer_mode', type=str, default=None,
                    choices=[None, 'none', 'source', 'target'],
                    help='Do polyformer traning.')
parser.add_argument("--nosave", dest='save_results', action='store_false',
                    help='Do not save prediction results.')
                                                                                
parser.add_argument('--insize', dest='orig_input_size', type=str, default=None,
                    help='Use images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='patch_size', type=str, default=None,
                    help='Do test on such input image patches.')

parser.add_argument('--outorigsize', dest='out_origsize', action='store_true',
                    help='Output seg maps in the same size of original uncropped images')
parser.add_argument("--ext", dest='save_ext', type=str, default='png',
                    help='Extension of saved predicted masks.')

parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')
parser.add_argument("--verbose", dest='verbose_output', action='store_true', 
                    help='Output individual scores of each image.')

parser.add_argument('--gpu', type=str,  default='0', help='ID of GPU to use')
parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')
parser.add_argument('--bb', dest='backbone_type', type=str,  default='eff-b4', help='Segtran backbone')

parser.add_argument("--nosqueeze", dest='use_squeezed_transformer', action='store_false',
                    help='Do not use attractor transformers (Default: use to increase scalability).')
parser.add_argument("--attractors", dest='num_attractors', default=256,
                    type=int, help='Number of attractors in the squeezed transformer.')
parser.add_argument("--noqkbias", dest='qk_have_bias', action='store_false', 
                    help='Do not use biases in Q, K projections (Using biases leads to better performance on BraTS).')

parser.add_argument("--translayers", dest='num_translayers', default=1,
                    type=int, help='Number of Cross-Frame Fusion layers.')
parser.add_argument('--layercompress', dest='translayer_compress_ratios', type=str, default=None, 
                    help='Compression ratio of channel numbers of each transformer layer to save RAM.')
parser.add_argument("--baseinit", dest='base_initializer_range', default=0.02,
                    type=float, help='Initializer range of transformer layers.')

parser.add_argument('--pos', dest='pos_code_type', type=str, default='lsinu', 
                    choices=['lsinu', 'zero', 'rand', 'sinu', 'bias'],
                    help='Positional code scheme')
parser.add_argument('--posw', dest='pos_code_weight', type=float, default=1.0)  
parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                    help='The radius of positional biases')                  
parser.add_argument("--poslayer1", dest='pos_code_every_layer', action='store_false',
                    help='Only add pos code to the first transformer layer input (Default: add to every layer).')
parser.add_argument("--posattonly", dest='pos_in_attn_only', action='store_true', 
                    help='Only use pos embeddings when computing attention scores (K, Q), and not use them in the input for V or FFN.')

parser.add_argument("--squeezeuseffn", dest='has_FFN_in_squeeze', action='store_true', 
                    help='Use the full FFN in the first transformer of the squeezed attention '
                         '(Default: only use the first linear layer, i.e., the V projection)')

parser.add_argument("--infpn", dest='in_fpn_layers', default='34',
                    choices=['234', '34', '4'],
                    help='Specs of input FPN layers')
parser.add_argument("--outfpn", dest='out_fpn_layers', default='1234',
                    choices=['1234', '234', '34'],
                    help='Specs of output FPN layers')

parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true',
                    help='Use BatchNorm instead of GroupNorm in input FPN.')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')

parser.add_argument('--modes', type=int, dest='num_modes', default=-1, help='Number of transformer modes')
parser.add_argument('--modedim', type=int, dest='attention_mode_dim', default=-1, help='Dimension of transformer modes')
parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false', 
                    help='Do not upsize backbone feature maps by 2.')
parser.add_argument("--testinterp", dest='test_interp', type=str, default=None,
                    help='Test how much error simple interpolation would cause. (Specify scaling ratio here)')
parser.add_argument("--gray", dest='gray_alpha', type=float, default=0.5,
                    help='Convert images to grayscale by so much degree.')
parser.add_argument("--removefrag", dest='do_remove_frag', action='store_true',
                    help='As a postprocessing step, remove fragmentary segments; only keep the biggest segment.')
parser.add_argument("--reloadmask", dest='reload_mask', action='store_true',
                    help='Reload mask directly from the gt file (ignoring the dataloader input). Used when input images are in varying sizes.')
parser.add_argument("--reshape", dest='reshape_mask_type', type=str, default=None,
                    choices=[None, 'rectangle', 'ellipse'],
                    help='Intentionally reshape the mask to test how well the model fits the mask bias.')
parser.add_argument("--t", dest='mask_thres', type=float, default=0.5,
                    help='The threshold of converting soft mask scores to 0/1.')
parser.add_argument('--multihead', dest='ablate_multihead', action='store_true',
                    help='Ablation to multimode transformer (using multihead instead)')
parser.add_argument('--vis', dest='vis_mode', type=str, default=None,
                    choices=[None, 'rf'],
                    help='Do visualization')
parser.add_argument('--vislayers', dest='vis_layers', type=str, default=None,
                    help='Indices of feature map layers used for visualization')
                    
parser.add_argument('--robust', dest='eval_robustness', action='store_true',
                    help='Evaluate feature map robustness against augmentation.')
parser.add_argument('--robustsamplenum', dest='robust_sample_num', type=int,  default=120, 
                    help='Number of test samples to use for robustness evaluation')
parser.add_argument("--robustaug", dest='robust_aug_types', type=str, default=None,
                    # Examples: None, 'brightness,contrast',
                    help='Augmentation types used during robustness training.')
parser.add_argument("--robustaugdeg", dest='robust_aug_degrees', type=str, default='0.5,1.5',
                    help='Degrees of robustness augmentation (1 or 2 numbers).')
parser.add_argument('--robustcp', dest='robust_ref_cp_path', type=str, default=None, 
                    help='Load this checkpoint for reference')
parser.add_argument('--savefeat', dest='save_features_img_count', type=int,  default=0, 
                    help='Save features of n images for t-SNE visualization (default: 0, no saving).')
                                                            
args_dict = {   'trans_output_type': 'private',
                'mid_type': 'shared',
                'in_fpn_scheme': 'AN',
                'out_fpn_scheme': 'AN',
                'tie_qk_scheme': 'none',
                'use_pretrained': True, # Doesn't matter if we load a trained checkpoint.    
                'ablate_pos_embed_type': None,                
            }

args = parser.parse_args()
for arg, v in args_dict.items():
    args.__dict__[arg] = v
    
if args.ablate_multihead:
    args.use_squeezed_transformer = False
if args.polyformer_mode == 'none':
    args.polyformer_mode = None
            
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
test_data_parent = os.path.join("../data/", args.task_name)
test_data_path = os.path.join("../data/", args.task_name, args.ds_name)
if args.checkpoint_dir is not None:
    timestamp = args.checkpoint_dir.split("-")[-1]
    timestamp = timestamp.replace("/", "")
else:
    timestamp = ""
args.job_name = '{}-{}'.format(args.task_name, args.ds_name)
if args.robust_aug_types:
    args.robust_aug_types = get_argument_list(args.robust_aug_types, str)
args.robust_aug_degrees = get_argument_list(args.robust_aug_degrees, float)
if len(args.robust_aug_degrees) == 1:
    args.robust_aug_degrees = args.robust_aug_degrees * 2
    
segtran_settings = {
                     'num_modes':  { '234': 2,   '34': 4,   '4': 4 }
                   }

default_settings = { 'unet':            {},
                     'unet-scratch':    {},
                     'nestedunet':      {},
                     'unet3plus':       {},
                     'deeplabv3plus':   {},
                     'deeplab-smp':     {},
                     'pranet':          {},
                     'attunet':         {},
                     'attr2unet':       {},
                     'dunet':           {},
                     'nnunet':          {},
                     'setr':            {},
                     'transunet':       {},
                     'segtran':         segtran_settings,
                     'fundus': {
                                 'num_classes': 3,
                                 'ds_class':    'SegCrop',
                                 # 'ds_names': 'train,valid,test',
                                 'orig_input_size': 576,
                                 # Each dim of the patch_size should always be multiply of 8.
                                 'patch_size':      288,
                                 'uncropped_size': { 'train':    (2056, 2124),
                                                     'test':     (1634, 1634),
                                                     'valid':    (1634, 1634),
                                                     'valid2':   (1940, 1940),
                                                     'test2':    -1,    # varying sizes
                                                     'drishiti': (2050, 1750),
                                                     'rim':      (2144, 1424),
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
                                 # 'bce_weight':  [0., 1],
                                 'ds_class':    'SegWhole',
                                 # 'ds_names': 'CVC-ClinicDB-train,Kvasir-train',
                                 'orig_input_size': 320,    # actual images are at various sizes. All resize to 320*320.
                                 'patch_size':      320,
                                 'has_mask':    { 'CVC-ClinicDB-train': True,   'Kvasir-train': True,
                                                  'CVC-ClinicDB-test': True,    'Kvasir-test': True,
                                                  'CVC-300': True,              'CVC-300-cyclegan': True,
                                                  'CVC-ColonDB': False,
                                                  'ETIS-LaribPolypDB': True },
                                 'weight':      { 'CVC-ClinicDB-train': 1,      'Kvasir-train': 1,
                                                  'CVC-ClinicDB-test': 1,       'Kvasir-test': 1,
                                                  'CVC-300': 1,                 'CVC-300-cyclegan': 1,
                                                  'CVC-ColonDB': 1,
                                                  'ETIS-LaribPolypDB': 1  }
                               },
                     'oct':  {
                                 'num_classes': 10,
                                 # 'bce_weight':  [0., 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                 'ds_class':    'SegWhole',
                                 'ds_names':    'duke',
                                 # Actual images are at various sizes. As the dataset is SegWhole, orig_input_size is ignored.
                                 # But output_upscale is computed as the ratio between orig_input_size and patch_size.
                                 # If you want to avoid output upscaling, set orig_input_size to the same as patch_size.
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

ds_stats_map = { 'fundus': 'fundus-cropped-gray{:.1f}-stats.json',
                 'polyp':  'polyp-whole-gray{:.1f}-stats.json',
                 'oct':    'oct-whole-gray{:.1f}-stats.json' }

stats_file_tmpl = ds_stats_map[args.task_name]
stats_filename = stats_file_tmpl.format(args.gray_alpha)
ds_stats = json.load(open(stats_filename))
default_settings[args.task_name].update(ds_stats)
print("'{}' mean/std loaded from '{}'".format(args.task_name, stats_filename))

get_default(args, 'mean',           default_settings, None,   [args.task_name, 'mean', args.ds_name])
get_default(args, 'std',            default_settings, None,   [args.task_name, 'std',  args.ds_name])
get_default(args, 'num_classes',    default_settings, None,   [args.task_name, 'num_classes'])
args.binarize = (args.num_classes == 2)
get_default(args, 'ds_class',       default_settings, None,   [args.task_name, 'ds_class'])

DataSetClass = dataloaders.datasets2d.__dict__[args.ds_class]

# Images after augmentation/transformation should keep their original size model_input_size.
# Will be resized before fed into the model.
tgt_width, tgt_height = args.orig_input_size

common_aug_func     = iaa.Sequential([
                            iaa.Resize({'height': tgt_height, 'width': tgt_width}),
                            iaa.Grayscale(alpha=args.gray_alpha)
                      ])
image_trans_func    = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize(args.mean, args.std)
                      ])

robust_aug_funcs = []
args.robustness_augs = None
if args.robust_aug_types:
    for aug_type in args.robust_aug_types:
        if aug_type == 'brightness':
            robust_aug_func = transforms.ColorJitter(brightness=args.robust_aug_degrees)
        elif aug_type == 'contrast':
            robust_aug_func = transforms.ColorJitter(contrast=args.robust_aug_degrees)
        else:
            breakpoint()
        robust_aug_funcs.append(robust_aug_func)
    args.robustness_augs = transforms.Compose(robust_aug_funcs)
    print("Robustness augmentation: {}".format(args.robustness_augs))
                       
if args.vis_mode is not None:
    image_trans_func = None             
# image is torch.tensor.
# mask is still np.array. Because ToTensor() divides mask values by 255, which should be avoided.                      
segmap_trans_func   = None
if args.vis_layers is not None:
    args.vis_layers = get_argument_list(args.vis_layers, int)

'''
 transforms.Compose([
                          transforms.Lambda(lambda mask: reshape_mask(mask, 0, 255, shape=args.reshape_mask_type)),
                          transforms.ToTensor()
                      ])
'''

ds_settings     = default_settings[args.task_name]
if 'uncropped_size' in ds_settings:
    uncropped_size  = ds_settings['uncropped_size'][args.ds_name]
else:
    uncropped_size  = -1

if uncropped_size == -1 and 'orig_dir' in ds_settings:
    orig_dir  = ds_settings['orig_dir'][args.ds_name]
    # orig_dir  = os.path.join(test_data_parent, orig_dir)
    orig_ext  = ds_settings['orig_ext'][args.ds_name]
else:
    orig_dir = orig_ext = None

has_mask = ds_settings['has_mask'][args.ds_name]

if args.sample_num:
    args.sample_num = int(args.sample_num)
else:
    args.sample_num = -1
        
db_test = DataSetClass(base_dir=test_data_path,
                       split=args.ds_split,
                       mode='test',
                       sample_num=args.sample_num,
                       mask_num_classes=args.num_classes,
                       has_mask=has_mask,
                       common_aug_func=common_aug_func,
                       image_trans_func=image_trans_func,
                       segmap_trans_func=segmap_trans_func,
                       binarize=args.binarize,
                       train_loc_prob=0,
                       chosen_size=args.orig_input_size[0],
                       uncropped_size=uncropped_size,
                       orig_dir=orig_dir,
                       orig_ext=orig_ext)

args.num_workers = 0 if args.debug else 4
testloader = DataLoader(db_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)
# num_modalities is used in segtran.
# num_modalities = 0 means there's not the modality dimension
# (but still a single modality) in the images loaded from db_train.
args.num_modalities = 0
if args.translayer_compress_ratios is not None:
    args.translayer_compress_ratios = get_argument_list(args.translayer_compress_ratios, float)
else:
    args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers+1) ]

def load_model(net, args, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=torch.device(args.device))
    params = net.state_dict()
    if 'model' in state_dict:
        model_state_dict = state_dict['model']
        cp_args          = state_dict['args']
        cp_iter_num      = state_dict['iter_num']
    else:
        model_state_dict = state_dict
        cp_args          = None
        cp_iter_num      = 0

    ignored_args_keys = [ 'maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                          'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size',
                          'saveiter', 'dice_warmup_steps', 'opt', 'lr', 'decay', 'tie_qk_scheme',
                          'initializer_range', 'base_initializer_range',
                          'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                          'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                          'bce_weight', 'seed', 'debug', 'ds_name', 'batch_size', 'dropout_prob',
                          'patch_size', 'orig_input_size', 'output_upscale',
                          'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output',
                          'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'ds_split', 'ds_names',
                          'job_name', 'mean', 'std', 'mask_thres', 'sample_num', 'sample_nums',
                          'poly_source_opt', 'poly_target_opt', 'ref_feat_cp_path', 'num_contrast_features',
                          'num_ref_features', 'selected_ref_classes', 'CONTRAST_LOSS_W', 'do_neg_contrast',
                          'adversarial_mode', 'num_feat_dis_in_chan', 'source_ds_name', 'source_batch_size',
                          'unsup_batch_size', 'DOMAIN_LOSS_W', 'SUPERVISED_W', 'RECON_W', 'ATTRACTOR_CONTRAST_W', 
                          'adda', 'bn_opt_scheme', 'opt_filters', 'use_pretrained', 'do_profiling', 
                          'only_first_linear_in_squeeze', 'source_ds_names', 'target_unsup_batch_size',
                          'use_vcdr_loss', 'VCDR_W', 'vcdr_estim_loss_start_iter', 'apply_attn_stage',
                          'vcdr_net_loss_start_iter', 'vcdr_estim_scheme', 'perturb_pew_range',
                          'perturb_posw_range', 'pos_embed_every_layer' ]

    warn_args_keys = [ 'num_recurrences', 'translayer_squeeze_ratios', 'use_exclusive_masks',
                       'use_attractor_transformer', 'squeeze_outfpn_dim_ratio', 'eff_feat_upsize' ]

    # Some old models don't have these keys in args. But they use the values specified here.
    old_default_keys = { 'num_recurrences': 1, 'qk_have_bias': True }
    args2 = copy.copy(args)

    if args.net == 'segtran' and cp_args is not None:
        if cp_args['task_name'] == 'refuge':
            cp_args['task_name'] = 'fundus'
            
        for k in old_default_keys:
            if k not in args:
                args2.__dict__[k] = old_default_keys[k]

        for k in cp_args:
            if (k in warn_args_keys) and (k not in args):
                print("args[{}] doesn't exist, checkpoint args[{}]={}, inconsistent!".format(k, k, cp_args[k]))
                continue
            elif (k in warn_args_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args2.__dict__[k], k, cp_args[k]))
                continue

            if (k not in ignored_args_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args2.__dict__[k], k, cp_args[k]))
                exit(0)

    model_state_dict2 = {}
    discarded_param_names = ['discriminator', 'recon']
    for k, v in model_state_dict.items():
        discarded = False
        for dk in discarded_param_names:
            if k.startswith(dk):
                discarded = True
                break
        if discarded:
            continue
                    
        if 'out_bridgeconv' in k:
            k2 = k.replace('out_bridgeconv', 'out_fpn_bridgeconv')
        else:
            k2 = k
        model_state_dict2[k2] = v
        
    params.update(model_state_dict2)
    net.load_state_dict(params, strict=False)

    randomize_qk = False
    if randomize_qk:
        if args.net == 'segtran':
            translayer = net.voxel_fusion.translayers[0]
            assert type(translayer) == SqueezedAttFeatTrans
        else:
            translayer = net.polyformer
            assert type(translayer) == PolyformerLayer
            
        translayer.in_ator_trans.query.weight.data.zero_() #normal_(mean=0.0, std=args.base_initializer_range)
        translayer.in_ator_trans.query.bias.data.zero_()
        translayer.in_ator_trans.key.weight.data.zero_() #normal_(mean=0.0, std=args.base_initializer_range)
        translayer.in_ator_trans.key.bias.data.zero_()
        translayer.ator_out_trans.query.weight.data.zero_() #normal_(mean=0.0, std=args.base_initializer_range)
        translayer.ator_out_trans.query.bias.data.zero_()
        translayer.ator_out_trans.key.weight.data.zero_() #normal_(mean=0.0, std=args.base_initializer_range)
        translayer.ator_out_trans.key.bias.data.zero_()            
            
        print("Query, key have been randomized")
              
    print("Model loaded from '{}'".format(checkpoint_path))

def test_calculate_metric(iter_nums):
    if args.net == 'unet':
        # timm-efficientnet performs slightly worse.
        if not args.vis_mode:
            backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
            net = smp.Unet(backbone_type, classes=args.num_classes, encoder_weights='imagenet')
        else:
            net = VanillaUNet(n_channels=3, num_classes=args.num_classes)
    elif args.net == 'unet-scratch':
        # net = UNet(num_classes=args.num_classes)
        net = VanillaUNet(n_channels=3, num_classes=args.num_classes, 
                          use_polyformer=args.polyformer_mode,
                          num_polyformer_layers=args.num_translayers,
                          num_attractors=args.num_attractors,
                          num_modes=args.num_modes)
    elif args.net == 'nestedunet':
        net = NestedUNet(num_classes=args.num_classes)
    elif args.net == 'unet3plus':
        net = UNet_3Plus(num_classes=args.num_classes)
    elif args.net == 'pranet':
        net = PraNet(num_classes=args.num_classes - 1)
    elif args.net == 'attunet':
        net = AttU_Net(output_ch=args.num_classes)
    elif args.net == 'attr2unet':
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
                        'polyp':  'SETR_PUP_320x320_10k_polyp_context_bs_4.py' }
        setr_cfg = Config.fromfile("networks/setr/configs/SETR/{}".format(task2config[args.task_name]))
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
        
    elif args.net.startswith('deeplab'):
        use_smp_deeplab = args.net.endswith('smp')
        if use_smp_deeplab:
            backbone_type = re.sub("^eff", "efficientnet", args.backbone_type)
            net = smp.DeepLabV3Plus(backbone_type, classes=args.num_classes, encoder_weights='imagenet')
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
        from nnunet.network_architecture.initialization import InitWeights_He
        from nnunet.network_architecture.generic_UNet import Generic_UNet
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
        get_default(args, 'num_modes',  default_settings, -1,   [args.net, 'num_modes', args.in_fpn_layers])
        set_segtran2d_config(args)
        print(args)
        net = Segtran2d(config)
    else:
        breakpoint()

    net.cuda()
    net.eval()

    if args.robust_ref_cp_path:
        refnet = copy.deepcopy(net)
        print("Reference network created")
        load_model(refnet, args, args.robust_ref_cp_path)
    else:
        refnet = None
        
    # Currently colormap is used only for OCT task.
    colormap = get_seg_colormap(args.num_classes, return_torch=True).cuda()

    # prepred: pre-prediction. postpred: post-prediction.
    task2mask_prepred   = { 'fundus': partial(fundus_map_mask, exclusive=args.use_exclusive_masks),
                            'polyp':  polyp_map_mask,
                            'oct':    partial(index_to_onehot, num_classes=args.num_classes) }
    task2mask_postpred  = { 'fundus': fundus_inv_map_mask,  
                            'polyp':  polyp_inv_map_mask,
                            'oct':    partial(onehot_inv_map, colormap=colormap) }
                                
    mask_prepred_mapping_func   =   task2mask_prepred[args.task_name]
    mask_postpred_mapping_funcs = [ task2mask_postpred[args.task_name] ]
    
    if args.do_remove_frag:
        remove_frag = lambda segmap: remove_fragmentary_segs(segmap, 255)
        mask_postpred_mapping_funcs.append(remove_frag)

    if not args.checkpoint_dir:
        if args.vis_mode is not None:
            visualize_model(net, args.vis_mode, args.vis_layers, args.patch_size, db_test)
            return

        if args.eval_robustness:
            eval_robustness(args, net, refnet, testloader, mask_prepred_mapping_func)
            return

    args.do_calc_vcdr_error = (args.task_name == 'fundus')
    
    all_results = np.zeros((args.num_classes, len(iter_nums)))
    
    for iter_idx, iter_num in enumerate(iter_nums):
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, 'iter_' + str(iter_num) + '.pth')
            load_model(net, args, checkpoint_path)

            if args.vis_mode is not None:
                visualize_model(net, args.vis_mode, args.vis_layers, args.patch_size, db_test)
                continue

            if args.eval_robustness:
                eval_robustness(args, net, refnet, testloader, mask_prepred_mapping_func)
                continue

        save_results = args.save_results and (not args.test_interp)
        if save_results:
            test_save_paths = []
            test_save_dirs  = []
            test_save_dir_tmpl  = "%s-%s-%s-%d" %(args.net, args.job_name, timestamp, iter_num)
            for suffix in ("-soft", "-%.1f" %args.mask_thres):
                test_save_dir = test_save_dir_tmpl + suffix
                test_save_path = "../prediction/%s" %(test_save_dir)
                if not os.path.exists(test_save_path):
                    os.makedirs(test_save_path)
                test_save_dirs.append(test_save_dir)
                test_save_paths.append(test_save_path)
        else:
            test_save_paths = None
            test_save_dirs  = None

        if args.save_features_img_count > 0:
            args.save_features_file_path = "%s-%s-feat-%s.pth" %(args.net, args.job_name, timestamp)
        else:
            args.save_features_file_path = None
            
        allcls_avg_metric, allcls_metric_count = \
                test_all_cases(net, testloader, 
                               task_name    = args.task_name,
                               num_classes  = args.num_classes,
                               mask_thres   = args.mask_thres,
                               model_type   = args.net,
                               orig_input_size = args.orig_input_size,
                               patch_size   = args.patch_size,
                               stride = (args.orig_input_size[0] // 2, args.orig_input_size[1] // 2),
                               test_save_paths = test_save_paths,
                               out_origsize = args.out_origsize,
                               mask_prepred_mapping_func    = mask_prepred_mapping_func,
                               mask_postpred_mapping_funcs  = mask_postpred_mapping_funcs,
                               reload_mask  = args.reload_mask,
                               test_interp  = args.test_interp,
                               do_calc_vcdr_error      = args.do_calc_vcdr_error,
                               save_features_img_count = args.save_features_img_count,
                               save_features_file_path = args.save_features_file_path,
                               save_ext     = args.save_ext,
                               verbose      = args.verbose_output)

        print("Iter-%d scores on %d images:" %(iter_num, allcls_metric_count[0]))
        dice_sum = 0
        for cls in range(1, args.num_classes):
            dice = allcls_avg_metric[cls-1]
            print('class %d: dice = %.3f' %(cls, dice))
            dice_sum += dice
            all_results[cls, iter_idx] = dice
        avg_dice = dice_sum / (args.num_classes - 1)
        print("Average dice: %.3f" %avg_dice)
        if args.do_calc_vcdr_error:
            print("vCDR error: %.3f" %allcls_avg_metric[args.num_classes - 1])
            
        if save_results:
            FNULL = open(os.devnull, 'w')
            for pred_type, test_save_dir, test_save_path in zip(('soft', 'hard'), test_save_dirs, test_save_paths):
                try:
                    do_zip = subprocess.run(["zip", "-FSr", "%s.zip" %test_save_dir, test_save_dir], cwd="../prediction",
                                            stdout=FNULL, stderr=subprocess.STDOUT)
                    print("{} archive:\n{}.zip".format(pred_type, os.path.abspath(test_save_path)))
                    
                except Exception as e:
                    print("Error when running zip:\n{}".format(e))        # To print out the exception message

    np.set_printoptions(precision=3, suppress=True)
    print(all_results[1:])
    return allcls_avg_metric


if __name__ == '__main__':
    if re.match(r"\d+-\d+,\d+", args.iters):
        match = re.match(r"(\d+)-(\d+),(\d+)", args.iters)
        start, end, step = int(match.group(1)), int(match.group(2)), int(match.group(3))
        iter_nums = list(np.arange(start, end+step, step))
    else:
        iter_nums = get_argument_list(args.iters, int)
    if args.test_interp is not None:
        args.test_interp = [ int(i) for i in args.test_interp.split(",") ]

    args.device = 'cuda'
    if args.vis_mode is not None:
        # Gradients are required for visualization.
        allcls_avg_metric = test_calculate_metric(iter_nums)
    else:
        with torch.no_grad():
            allcls_avg_metric = test_calculate_metric(iter_nums)
