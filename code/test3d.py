import os
import time
from datetime import datetime
import argparse
import torch
import torch.nn.functional as F
from networks.vnet import VNet
from networks.segtran3d import Segtran3d
from networks.segtran3d import CONFIG as config3d
from networks.segtran25d import Segtran25d
from networks.segtran25d import CONFIG as config25d

from networks.unet3d import Modified3DUNet as UNet3D
from test_util3d import test_all_cases
import dataloaders.datasets3d
from dataloaders.datasets3d import ToTensor
from common_util import AverageMeters, get_default, get_filename
import subprocess
import copy
from fvcore.nn import FlopCountAnalysis
from internal_util import visualize_model, eval_robustness

parser = argparse.ArgumentParser()
parser.add_argument('--task', dest='task_name', type=str, default='brats', help='Name of the segmentation task.')
parser.add_argument('--ds', dest='test_ds_name', type=str, default=None, help='Dataset name for test')

parser.add_argument('--split', dest='ds_split', type=str, default='all',
                    choices=['train', 'test', 'all'], help='Split of the dataset')
parser.add_argument('--cpdir', dest='checkpoint_dir', type=str, default=None,
                    help='Load checkpoint(s) from this directory')
parser.add_argument('--iters', type=str,  default='11000,10000', help='checkpoint iteration(s)')
parser.add_argument('--bs', dest='batch_size', type=int, default=20, help='batch_size')

parser.add_argument('--insize', dest='orig_input_size', type=str, default=None, 
                    help='Select images of this size (among all cropping sizes) for training. Set to 0 to use all sizes.')
parser.add_argument('--patch', dest='orig_patch_size', type=str, default=None, 
                    help='Do test on such input image patches.')
parser.add_argument('--scale', dest='input_scale', type=str, default=None, 
                    help='Scale input images by this ratio for training.')
parser.add_argument('--dpool', dest='D_pool_K', type=int, default=-1, 
                    help='Scale input images by this ratio for training.')
                    
parser.add_argument("--debug", dest='debug', action='store_true', help='Debug program.')
parser.add_argument("--verbose", dest='verbose_output', action='store_true', 
                    help='Output individual scores of each image.')

parser.add_argument('--gpu', type=str,  default='0', help='ID of GPU to use')
parser.add_argument('--net', type=str,  default='segtran', help='Network architecture')
parser.add_argument('--bb', dest='backbone_type', type=str, default=None, 
                    help='Backbone of Segtran / Encoder of other models')

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

parser.add_argument('--pos', dest='pos_code_type', type=str, default='lsinu', 
                    choices=['lsinu', 'none', 'rand', 'sinu', 'bias'],
                    help='Positional code scheme')
parser.add_argument('--posw', dest='pos_code_weight', type=float, default=1.0)     
parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                    help='The radius of positional biases')               
parser.add_argument("--poslayer1", dest='pos_code_every_layer', action='store_false', 
                    help='Only add pos code to the first transformer layer input (Default: add to every layer).')
parser.add_argument("--squeezeuseffn", dest='has_FFN_in_squeeze', action='store_true', 
                    help='Use the full FFN in the first transformer of the squeezed attention '
                         '(Default: only use the first linear layer, i.e., the V projection)')

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
parser.add_argument("--inbn", dest='in_fpn_use_bn', action='store_true', 
                    help='Use BatchNorm instead of GroupNorm in input FPN.')
parser.add_argument('--attnclip', dest='attn_clip', type=int,  default=500, help='Segtran attention clip')
                    
parser.add_argument('--mod', dest='chosen_modality', type=int, default=-1, help='The modality to use if images are of multiple modalities')
parser.add_argument("--nofeatup", dest='bb_feat_upsize', action='store_false', 
                    help='Do not upsize backbone feature maps by 2.')

parser.add_argument("--testinterp", dest='test_interp', type=str, default=None, 
                    help='Test how much error simple interpolation would cause. (Specify scaling ratio here)')
parser.add_argument('--vis', dest='vis_mode', type=str, default=None,
                    choices=[None, 'rf'],
                    help='Do visualization')
parser.add_argument('--robust', dest='eval_robustness', action='store_true',
                    help='Evaluate feature map robustness against augmentation.')
parser.add_argument('--augdeg', dest='aug_degree', type=float, default=0.5,
                    help='Augmentation degree when doing robustness evaluation.')
parser.add_argument('--flop', dest='calc_flop', action='store_true', help="Compute model FLOPs")

args_dict = {   'trans_output_type': 'private',
                'mid_type': 'shared',
                'in_fpn_scheme':     'AN',
                'out_fpn_scheme':    'AN',    
                'use_pretrained': True, # Doesn't matter if we load a trained checkpoint.
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
    # if this arg is not set through command line (i.e., in its default value None), then take value from cond_args_dict.
    if (arg not in args.__dict__) or (args.__dict__[arg] is None):
        args.__dict__[arg] = v

if args.mince_scales is not None:
    args.mince_scales = [ int(L) for L in args.mince_scales.split(",") ]
if args.mince_channel_props is not None:
    args.mince_channel_props = [ float(L) for L in args.mince_channel_props.split(",") ]

if args.ablate_multihead:
    args.use_squeezed_transformer = False
        
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.device = 'cuda'

segtran_settings = { 
                     'num_modes':  { '234': 2,   '34': 4,   '4': 4 }
                   }

default_settings = { 'unet':            {},
                     'vnet':            {},
                     'segtran':         segtran_settings,
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
get_default(args, 'test_ds_name',       default_settings, None, [args.task_name, 'test_ds_name'])
get_default(args, 'input_scale',        default_settings, None, [args.task_name, 'input_scale'])
get_default(args, 'D_pool_K',           default_settings, -1, [args.task_name, 'D_pool_K'])
get_default(args, 'xyz_permute',        default_settings, None, [args.task_name, 'xyz_permute'])
get_default(args, 'chosen_modality',    default_settings, -1,   [args.task_name, 'chosen_modality'])
get_default(args, 'num_classes',        default_settings, -1,   [args.task_name, 'num_classes'])
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

get_default(args, 'ds_class',       default_settings, None,   [args.task_name, 'ds_class'])

test_data_path = os.path.join("../data/", args.task_name, args.test_ds_name)
if args.checkpoint_dir is not None:
    timestamp = args.checkpoint_dir.split("-")[-1]
    timestamp = timestamp.replace("/", "")
else:
    timestamp = ""
args.job_name = '{}-{}'.format(args.task_name, args.test_ds_name)
        
DataSetClass = dataloaders.datasets3d.__dict__[args.ds_class]
ds_settings  = default_settings[args.task_name]
has_mask     = ds_settings['has_mask'][args.test_ds_name]

db_test = DataSetClass(base_dir=test_data_path,
                       split=args.ds_split,
                       mode='test',
                       ds_weight=1.,
                       #common_aug_func=common_aug_func,
                       #image_trans_func=image_trans_func,
                       #segmap_trans_func=segmap_trans_func,
                       transform=ToTensor(),
                       chosen_modality=args.chosen_modality,
                       binarize=args.binarize,
                       train_loc_prob=0,
                       #chosen_size=args.orig_input_size,
                       min_output_size=args.orig_patch_size,
                       xyz_permute=args.xyz_permute)
                                         
if args.chosen_modality == -1:
    args.orig_in_channels = db_test.num_modalities
else:
    args.orig_in_channels = 1

if args.translayer_compress_ratios is not None:
    args.translayer_compress_ratios = [ float(r) for r in args.translayer_compress_ratios.split(",") ]
else:
    args.translayer_compress_ratios = [ 1 for layer in range(args.num_translayers + 1) ]

print(args)
            
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

    deleted_keys_prefix = ['voxel_fusion.attn_scaler']
    deleted_keys = []
    for key in model_state_dict:
        for key_prefix in deleted_keys_prefix:
            if key.startswith(key_prefix):
                deleted_keys.append(key)
    for key in deleted_keys:
        del model_state_dict[key]
        
    ignored_keys = [ 'maxiter', 'checkpoint_path', 'model_input_size', 't_total', 'num_workers',
                     'lr_warmup_ratio', 'lr_warmup_steps', 'local_rank', 'distributed', 'world_size', 
                     'seed', 'debug', 'test_ds_name', 'test_ds_name', 'batch_size', 'dropout_prob', 
                     'orig_patch_size', 'input_patch_size', 'D_pool_K', 'binarize',
                     'checkpoint_dir', 'iters', 'out_origsize', 'out_softscores', 'verbose_output', 
                     'gpu', 'test_interp', 'do_remove_frag', 'reload_mask', 'saveiter',
                     'job_name', 'ds_names', 'train_ds_names', 'dice_warmup_steps', 'opt', 'lr', 'decay',
                     'grad_clip', 'localization_prob', 'tune_bn_only', 'MAX_DICE_W', 'deterministic',
                     'lr_schedule', 'out_fpn_do_dropout', 'randscale', 'do_affine', 'focus_class',
                     'bce_weight', 'D_scale', 'orig_input_size', 'input_scale',
                     'mean', 'std', 'mask_thres', 'use_pretrained', 'only_first_linear_in_squeeze',
                     'perturb_posw_range', 'pos_in_attn_only', 'attention_mode_dim',
                     'use_attn_consist_loss', 'ATTNCONSIST_W'
                   ]
                     
    # Some old models don't have these keys in args. But they use the values specified here.
    old_default_keys = { 'out_fpn_upsampleD_scheme': 'interpolate', 
                         'num_recurrences': 1, 'qk_have_bias': True }
    args2 = copy.copy(args)
    
    if args.net == 'segtran' and cp_args is not None:
        for k in old_default_keys:
            if k not in args:
                args2.__dict__[k] = old_default_keys[k]

        for k in cp_args:
            if (k not in ignored_keys) and (args2.__dict__[k] != cp_args[k]):
                print("args[{}]={}, checkpoint args[{}]={}, inconsistent!".format(k, args2.__dict__[k], k, cp_args[k]))
                exit(0)
    
    params.update(model_state_dict)
    net.load_state_dict(params)
    
    print("Model loaded from '{}'".format(checkpoint_path))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def test_calculate_metric(iter_nums):
    if args.net == 'vnet':
        net = VNet(n_channels=1, num_classes=args.num_classes, normalization='batchnorm', has_dropout=False)
    elif args.net == 'unet':
        net = UNet3D(in_channels=1, num_classes=args.num_classes)
    elif args.net == 'segtran':
        get_default(args, 'num_modes',  default_settings, -1,   [args.net, 'num_modes', args.in_fpn_layers])      
        if args.segtran_type == '3d':
            config3d.update_config(args)
            net = Segtran3d(config3d)
        else:
            config25d.update_config(args)
            net = Segtran25d(config25d)    


    net.cuda()
    net.eval()
    preproc_fn = None

    print(f"Parameter Count: {count_parameters(net)}")
    if args.calc_flop:
        test_blob = db_test[0]
        # test_img: [4, 240, 240, 155]
        test_img = test_blob['image'].unsqueeze(0).cuda()
        # test_img: [1, 4, 112, 112, 96]
        test_img = F.interpolate(test_img, args.input_patch_size)
        flops = FlopCountAnalysis(net, test_img)
        print(flops.by_module())
        exit()

    if not args.checkpoint_dir:
        if args.vis_mode is not None:
            visualize_model(net, args.vis_mode)
            return

        if args.eval_robustness:
            eval_robustness(net, db_test, args.aug_degree)
            return
                
    for iter_num in iter_nums:
        if args.checkpoint_dir:
            checkpoint_path = os.path.join(args.checkpoint_dir, 'iter_' + str(iter_num) + '.pth')
            load_model(net, args, checkpoint_path)

            if args.vis_mode is not None:
                visualize_model(net, args.vis_mode)
                continue

            if args.eval_robustness:
                eval_robustness(net, db_test, args.aug_degree)
                continue
                                
        save_result = not args.test_interp
     
        if save_result:
            test_save_paths = []
            test_save_dirs  = []      
            test_save_dir  = "%s-%s-%s-%d" %(args.net, args.job_name, timestamp, iter_num)  
            test_save_path = "../prediction/%s" %(test_save_dir)
            if not os.path.exists(test_save_path):
                os.makedirs(test_save_path)
            test_save_dirs.append(test_save_dir)
            test_save_paths.append(test_save_path)
        else:
            test_save_paths = [ None ]
            test_save_dirs  = [ None ]
            
        # No need to use dataloader to pass data, 
        # as one 3D image is split into many patches to do segmentation.
        allcls_avg_metric = test_all_cases(net, db_test, task_name=args.task_name, 
                                           net_type=args.net,
                                           num_classes=args.num_classes,
                                           batch_size=args.batch_size,
                                           orig_patch_size=args.orig_patch_size, 
                                           input_patch_size=args.input_patch_size,
                                           stride_xy=args.orig_patch_size[0] // 2, 
                                           stride_z=args.orig_patch_size[2]  // 2,
                                           save_result=save_result, 
                                           test_save_path=test_save_paths[0],
                                           preproc_fn=preproc_fn, 
                                           test_interp=args.test_interp,
                                           has_mask=has_mask)

        print("%d scores:" %iter_num)
        for cls in range(1, args.num_classes):
            dice, jc, hd, asd = allcls_avg_metric[cls-1]
            print('%d: dice: %.3f, jc: %.3f, hd: %.3f, asd: %.3f' %(cls, dice, jc, hd, asd))

        if save_result:
            FNULL = open(os.devnull, 'w')
            # Currently only save hard predictions.
            for pred_type, test_save_dir, test_save_path in zip(('hard',), test_save_dirs, test_save_paths):
                try:
                    do_tar = subprocess.run(["tar", "cvf", "%s.tar" %test_save_dir, test_save_dir], cwd="../prediction", 
                                            stdout=FNULL, stderr=subprocess.STDOUT)
                    print("{} tarball:\n{}.tar".format(pred_type, os.path.abspath(test_save_path)))
                    
                except Exception as e:
                    print("Error when running tar:\n{}".format(e))        # To print out the exception message
                    
    return allcls_avg_metric


if __name__ == '__main__':
    iter_nums = [ int(i) for i in args.iters.split(",") ]
    if args.test_interp is not None:
        args.test_interp = [ float(i) for i in args.test_interp.split(",") ]
        
    with torch.no_grad():
        allcls_avg_metric = test_calculate_metric(iter_nums)
