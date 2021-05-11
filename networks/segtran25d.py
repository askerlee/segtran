import math
import numpy as np
import re

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import resnet
import resnet_ibn
from efficientnet.model import EfficientNet
from networks.segtran_shared import bb2feat_dims, SegtranFusionEncoder, CrossAttFeatTrans, MultiModeFeatTrans, \
                                    SegtranInitWeights, get_all_indices
from train_util import batch_norm
    
class Segtran25dConfig:
    def __init__(self):
        self.backbone_type = 'eff-b3'         # resnet34, resnet50, efficientnet-b0~b4
        self.use_pretrained = True
        self.bb_feat_dims = bb2feat_dims[self.backbone_type]
        self.num_translayers = 1    
        # vanilla transformer: 1, use_attractor_transformer: 2.
        self.cross_attn_score_scales = [1., 2.]
            
        # Set in_fpn_scheme and out_fpn_scheme to 'NA' and 'NA', respectively.
        # NA: normalize first, then add. AN: add first, then normalize. 
        self.set_fpn_layers('default', '34', '1234', 'AN', 'AN',
                            translayer_compress_ratios=[1,1], do_print=False)
        self.bb_feat_upsize = True      # Configure backbone to generate x2 feature maps.
        self.in_fpn_use_bn   = False    # If in FPN uses BN, it performs slightly worse than using GN.
        self.out_fpn_use_bn  = False    # If out FPN uses BN, it performs worse than using GN.
        self.resnet_bn_to_gn = False    # Converting resnet BN to GN reduces performance.
        
        self.G = 8                      # number of groups in all group norms.
        self.pos_dim  = 3
        self.input_scale = (1., 1., 1.)
        
        self.num_classes = 2
        
        # Architecture settings
        self.num_modes = 4
        # Use AttractorAttFeatTrans instead of the vanilla CrossAttFeatTrans.
        self.use_attractor_transformer = True
        self.num_attractors = 1024
        self.tie_qk_scheme = 'shared'           # shared, loose, or none.
        self.mid_type      = 'shared'           # shared, private, or none.
        self.trans_output_type  = 'private'     # shared or private.
        self.apply_attn_early = True
        self.act_fun = F.gelu
        self.pos_embed_every_layer = True

        self.attn_clip = 500
        self.base_initializer_range = 0.02
        # Add an identity matrix (*0.02*query_idbias_scale) to query/key weights
        # to make a bias towards identity mapping.
        # Set to 0 to disable the identity bias.
        self.query_idbias_scale = 10
        self.feattrans_lin1_idbias_scale = 10

        # Pooling settings
        # Aggregate box attentions of different seg modes according to their seg losses.
        self.pool_modes_attn  = 'softmax'     # softmax, max, mean or none.
        # Do not aggregate seg modes in CrossAttFeatTrans.
        # Instead, aggregate them in mobert_pretrain.py according to their seg losses.
        self.pool_modes_feat  = 'softmax'   # softmax, max, mean, or none. With [] means keepdim=True.
        self.pool_modes_basis = 'feat'      # attn or feat

        # Randomness settings
        self.hidden_dropout_prob = 0.2
        self.attention_probs_dropout_prob = 0.2
        self.out_fpn_do_dropout = False
        self.eval_robustness = False

        self.orig_in_channels = 1
        # 2.5D specific settings.
        # inchan_to3_scheme: 
        # avgto3 (average central two slices, yield 3 slices), only for effective slices == 2 or 4.
        # or stemconv (modifying stem conv to take 2-channel input). Only implemented for efficientnet.
        # or dup3 (duplicate 1 channel 3 times to fake RGB channels).
        # or bridgeconv (a conv to convert C>1 channels to 3),
        # or None (when effective channels == 3), do nothing.
        self.inchan_to3_scheme = 'stemconv'
        self.D_groupsize    = 1                         # Depth grouping: 1, 2, 4.
        self.D_pool_K       = 2                         # Depth pooling after infpn
        self.out_fpn_upsampleD_scheme = 'conv'          # conv, interpolate, none
        
    def set_fpn_layers(self, config_name, in_fpn_layers, out_fpn_layers,
                       in_fpn_scheme, out_fpn_scheme,
                       translayer_compress_ratios, do_print=True):
        self.in_fpn_layers  = [ int(layer) for layer in in_fpn_layers ]
        self.out_fpn_layers = [ int(layer) for layer in out_fpn_layers ]
        # out_fpn_layers cannot be a subset of in_fpn_layers, like: in=234, out=34.
        # in_fpn_layers  could be a subset of out_fpn_layers, like: in=34,  out=234.
        if self.out_fpn_layers[-1] > self.in_fpn_layers[-1]:
            print("in_fpn_layers=%s is not compatible with out_fpn_layers=%s" %(self.in_fpn_layers, self.out_fpn_layers))
            exit(0)
            
        self.orig_in_feat_dim    = self.bb_feat_dims[self.in_fpn_layers[-1]]
        self.translayer_compress_ratios = translayer_compress_ratios
        assert len(translayer_compress_ratios) == self.num_translayers + 1, \
               "Length of {} != 1 + num_translayers {}".format(translayer_compress_ratios, self.num_translayers)
        
        # Convert adjacent ratios to absolute ratios: 
        # 1., 2., 2., 2. => 1, 2., 4., 8.
        translayer_compress_ratios = np.cumprod(translayer_compress_ratios)
        # Input/output dimensions of each transformer layer.
        # Could be different from self.orig_in_feat_dim, 
        # which is the backbone feature dimension from in_fpn.
        self.translayer_dims = [ int(self.orig_in_feat_dim / ratio) for ratio in translayer_compress_ratios ]
        self.trans_in_dim   = self.translayer_dims[0]
        self.min_feat_dim   = np.min(self.translayer_dims)
        self.trans_out_dim  = self.translayer_dims[-1]
                
        self.in_fpn_scheme  = in_fpn_scheme
        self.out_fpn_scheme = out_fpn_scheme    
        
        if do_print:
            print("'%s' orig in-feat: %d, in-feat: %d, out-feat: %d, in-scheme: %s, out-scheme: %s, "
                  "translayer_dims: %s" % \
                    (config_name, self.orig_in_feat_dim, self.trans_in_dim, self.trans_out_dim,
                     self.in_fpn_scheme, self.out_fpn_scheme,
                     self.translayer_dims))
            
CONFIG = Segtran25dConfig()

def set_segtran25d_config(args):
    CONFIG.num_classes                  = args.num_classes
    CONFIG.backbone_type                = args.backbone_type
    CONFIG.use_pretrained               = args.use_pretrained
    CONFIG.bb_feat_upsize               = args.bb_feat_upsize
    CONFIG.bb_feat_dims                 = bb2feat_dims[CONFIG.backbone_type]
    CONFIG.in_fpn_use_bn                = args.in_fpn_use_bn
    CONFIG.use_attractor_transformer    = args.use_attractor_transformer
    CONFIG.cross_attn_score_scale       = CONFIG.cross_attn_score_scales[CONFIG.use_attractor_transformer]
    CONFIG.num_attractors               = args.num_attractors
    CONFIG.num_translayers              = args.num_translayers
    CONFIG.apply_attn_early             = (args.apply_attn_stage == 'early')
    CONFIG.num_modes                    = args.num_modes
    CONFIG.trans_output_type            = args.trans_output_type
    CONFIG.mid_type                     = args.mid_type
    CONFIG.pos_embed_every_layer        = args.pos_embed_every_layer
    CONFIG.base_initializer_range       = args.base_initializer_range
    CONFIG.ablate_pos_embed_type        = args.ablate_pos_embed_type
    CONFIG.ablate_multihead             = args.ablate_multihead
    if 'dropout_prob' in args:
        CONFIG.hidden_dropout_prob          = args.dropout_prob
        CONFIG.attention_probs_dropout_prob = args.dropout_prob
    if 'out_fpn_do_dropout' in args:
        CONFIG.out_fpn_do_dropout           = args.out_fpn_do_dropout
    CONFIG.attn_clip                        = args.attn_clip
    CONFIG.set_fpn_layers('args', args.in_fpn_layers, args.out_fpn_layers,
                          args.in_fpn_scheme, args.out_fpn_scheme,
                          translayer_compress_ratios=args.translayer_compress_ratios)
        
    CONFIG.orig_in_channels             = args.orig_in_channels
    CONFIG.inchan_to3_scheme            = args.inchan_to3_scheme
    CONFIG.D_groupsize                  = args.D_groupsize
    CONFIG.D_pool_K                     = args.D_pool_K
    CONFIG.out_fpn_upsampleD_scheme     = args.out_fpn_upsampleD_scheme
    CONFIG.input_scale                  = args.input_scale
    
    CONFIG.device                       = args.device
    if 'eval_robustness' in args:
        CONFIG.eval_robustness          = args.eval_robustness
            
    return CONFIG

            
class Segtran25d(SegtranInitWeights):
    def __init__(self, config):
        super(Segtran25d, self).__init__(config)
        self.config         = config
        self.device         = config.device
        self.orig_in_channels   = config.orig_in_channels
        self.trans_in_dim   = config.trans_in_dim
        self.trans_out_dim  = config.trans_out_dim
        self.num_translayers = config.num_translayers
        self.bb_feat_upsize = config.bb_feat_upsize
        self.G              = config.G
        self.voxel_fusion   = SegtranFusionEncoder(config, 'Fusion')
        self.backbone_type  = config.backbone_type
        self.use_pretrained = config.use_pretrained
        self.pos_embed_every_layer = config.pos_embed_every_layer
        if self.backbone_type.startswith('resnet'):
            self.backbone   = resnet.__dict__[self.backbone_type](pretrained=self.use_pretrained, 
                                                                  do_pool1=not self.bb_feat_upsize)
            print("%s created" %self.backbone_type)
        elif self.backbone_type.startswith('resibn'):
            mat = re.search(r"resibn(\d+)", self.backbone_type)
            backbone_type   = 'resnet{}_ibn_a'.format(mat.group(1))
            self.backbone   = resnet_ibn.__dict__[backbone_type](pretrained=self.use_pretrained, 
                                                                 do_pool1=not self.bb_feat_upsize)
            print("%s created" %backbone_type)
        elif self.backbone_type.startswith('eff'):
            backbone_type   = self.backbone_type.replace("eff", "efficientnet")
            stem_stride     = 1 if self.bb_feat_upsize else 2
            advprop         = True
            
            if self.use_pretrained:
                self.backbone   = EfficientNet.from_pretrained(backbone_type, advprop=advprop,
                                                               ignore_missing_keys=True,
                                                               stem_stride=stem_stride)
            else:
                self.backbone   = EfficientNet.from_name(backbone_type,
                                                         stem_stride=stem_stride)
                                                                     
            print("{} created (stem_stride={}, advprop={})".format(backbone_type, stem_stride, advprop))
            
        self.inchan_to3_scheme  = config.inchan_to3_scheme
        self.D_groupsize        = config.D_groupsize
        self.eff_in_channels    = self.orig_in_channels * self.D_groupsize
        
        self.D_pool_K           = config.D_pool_K
        self.out_fpn_upsampleD_scheme = config.out_fpn_upsampleD_scheme
        self.input_scale        = config.input_scale

        # For brats, eff_in_channels = 4 (4 modalities, D_groupsize = 1).
        if self.eff_in_channels != 3:
            if self.inchan_to3_scheme == 'avgto3':
                if self.eff_in_channels == 2:
                    self.in_bridge_to3  = nn.Linear(2, 3, bias=False)
                    in_avg_2to3_weight  = torch.tensor([[1, 0], [0.5, 0.5], [0, 1]])
                    self.in_bridge_to3.weight.data.copy_(in_avg_2to3_weight)
                elif self.eff_in_channels == 4:
                    self.in_bridge_to3  = nn.Linear(4, 3, bias=False)
                    in_avg_4to3_weight  = torch.tensor([[1, 0, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0, 1]])
                    self.in_bridge_to3.weight.data.copy_(in_avg_4to3_weight)
                else:
                    raise NotImplementedError("'avgto3' is only for effective channels == 2 or 4, not {}".format(self.eff_in_channels))
                self.in_bridge_to3.weight.requires_grad = False
            elif self.eff_in_channels == 1 and self.inchan_to3_scheme == 'dup3':
                self.in_bridge_to3  = lambda x: x.expand(-1, 3, -1, -1, -1)
            elif self.inchan_to3_scheme == 'bridgeconv':
                self.in_bridge_to3  = nn.Conv3d(self.eff_in_channels, 3, 1)
            # stemconv is only applicable for efficientnet.
            elif self.eff_in_channels > 3 and self.inchan_to3_scheme == 'stemconv':
                if self.backbone_type.startswith('eff'):
                    self.backbone._change_in_channels(4, keep_RGB_weight=True)
                    self.in_bridge_to3  = nn.Identity()
                else:
                    raise NotImplementedError("Changing stemconv channel number is not supported for {}".format(self.backbone_type))
            else:
                raise NotImplementedError("Effective input channel size={}*{} is not supported for scheme '{}'".format(
                                          self.orig_in_channels, self.D_groupsize, self.inchan_to3_scheme))

        self.in_fpn_use_bn  = config.in_fpn_use_bn
        self.in_fpn_layers  = config.in_fpn_layers
        self.in_fpn_scheme  = config.in_fpn_scheme
        
        # FPN output resolution is determined by the smallest number (lowest layer).
        pool_stride = 2**np.min(self.in_fpn_layers)
        if not self.bb_feat_upsize:
            pool_stride *= 2
        self.mask_pool = nn.AvgPool2d((pool_stride, pool_stride))
                
        self.bb_feat_dims = config.bb_feat_dims
        self.in_fpn23_conv  = nn.Conv2d(self.bb_feat_dims[2], self.bb_feat_dims[3], 1)
        self.in_fpn34_conv  = nn.Conv2d(self.bb_feat_dims[3], self.bb_feat_dims[4], 1)
        # Default in_fpn_layers: 34. last_in_fpn_layer_idx: 4.
        last_in_fpn_layer_idx = self.in_fpn_layers[-1]
        if self.bb_feat_dims[last_in_fpn_layer_idx] != self.trans_in_dim:
            self.in_fpn_bridgeconv = nn.Conv2d(self.bb_feat_dims[last_in_fpn_layer_idx], self.trans_in_dim, 1)
        else:
            self.in_fpn_bridgeconv = nn.Identity()
                
        # in_bn4b/in_gn4b normalizes in_fpn43_conv(layer 4 features), 
        # so the feature dim = dim of layer 3.
        # in_bn3b/in_gn3b normalizes in_fpn32_conv(layer 3 features), 
        # so the feature dim = dim of layer 2.
        if self.in_fpn_use_bn:
            self.in_bn3b = nn.BatchNorm2d(self.bb_feat_dims[3])
            self.in_bn4b = nn.BatchNorm2d(self.bb_feat_dims[4])
            self.in_fpn_norms = [ None, None, None, self.in_bn3b, self.in_bn4b ]
        else:            
            self.in_gn3b = nn.GroupNorm(self.G, self.bb_feat_dims[3])
            self.in_gn4b = nn.GroupNorm(self.G, self.bb_feat_dims[4])
            self.in_fpn_norms = [ None, None, None, self.in_gn3b, self.in_gn4b ]
            
        self.in_fpn_convs   = [ None, None, self.in_fpn23_conv, self.in_fpn34_conv ]
        
        self.num_classes    = config.num_classes
        
        self.out_fpn_use_bn = config.out_fpn_use_bn
        self.out_fpn_layers = config.out_fpn_layers
        self.out_fpn_scheme = config.out_fpn_scheme
        self.out_fpn_do_dropout = config.out_fpn_do_dropout
        
        if self.out_fpn_layers != self.in_fpn_layers:
            self.do_out_fpn = True
                                              
            self.out_fpn12_conv3d = nn.Conv3d(self.bb_feat_dims[1], 
                                              self.bb_feat_dims[2], 1)
            self.out_fpn23_conv3d = nn.Conv3d(self.bb_feat_dims[2], 
                                              self.bb_feat_dims[3], 1)
            self.out_fpn34_conv3d = nn.Conv3d(self.bb_feat_dims[3], 
                                              self.bb_feat_dims[4], 1)
            last_out_fpn_layer = self.out_fpn_layers[-len(self.in_fpn_layers)]
            self.out_fpn_bridgeconv3d = nn.Conv3d(self.bb_feat_dims[last_out_fpn_layer], 
                                                  self.trans_out_dim, 1)
            if self.out_fpn_upsampleD_scheme == 'conv':
                self.out_feat_dim       = self.trans_out_dim // self.D_pool_K
                self.out_fpn_upsampleD  = nn.Conv3d(self.trans_out_dim, self.out_feat_dim * self.D_pool_K, 1)
            else:
                self.out_feat_dim = self.trans_out_dim
                
            # out_bn3b/out_gn3b normalizes out_fpn23_conv3d(layer 3 features), 
            # so the feature dim = dim of layer 2.
            # out_bn2b/out_gn2b normalizes out_fpn12_conv3d(layer 2 features), 
            # so the feature dim = dim of layer 1.
            if self.out_fpn_use_bn:
                self.out_bn2b       = nn.BatchNorm3d(self.bb_feat_dims[2])
                self.out_bn3b       = nn.BatchNorm3d(self.bb_feat_dims[3])
                self.out_bn4b       = nn.BatchNorm3d(self.bb_feat_dims[4])
                self.out_fpn_norms  = [ None, None, self.out_bn2b, self.out_bn3b, self.out_bn4b ]
            else:
                self.out_gn2b       = nn.GroupNorm(self.G, self.bb_feat_dims[2])
                self.out_gn3b       = nn.GroupNorm(self.G, self.bb_feat_dims[3])
                self.out_gn4b       = nn.GroupNorm(self.G, self.bb_feat_dims[4])
                self.out_fpn_norms  = [ None, None, self.out_gn2b, self.out_gn3b, self.out_gn4b ]
                
            self.out_fpn_convs   = [ None, self.out_fpn12_conv3d, self.out_fpn23_conv3d, self.out_fpn34_conv3d ]
            self.out_conv3d      = nn.Conv3d(self.out_feat_dim, self.num_classes, 1)
            self.out_fpn_dropout = nn.Dropout(config.hidden_dropout_prob)
        # out_fpn_layers = in_fpn_layers, no need to do fpn at the output end. 
        # Output class scores directly.
        else:
            self.do_out_fpn = False
            if '2' in self.in_fpn_layers:
                # Output resolution is 1/4 of input already. No need to do upsampling here.
                self.out_conv3d = nn.Conv3d(config.trans_out_dim, self.num_classes, 1)
            else:
                # Output resolution is 1/8 of input. Do upsampling to make resolution x 2
                self.out_conv3d = nn.ConvTranspose3d(config.trans_out_dim, self.num_classes,
                                                     (2,2,1), (2,2,1))
            
        self.apply(self.init_weights)
        # tie_qk() has to be executed after weight initialization.
        self.apply(self.tie_qk)
        self.apply(self.add_identity_bias)

        self.scales_printed = False
        self.translayer_dims = config.translayer_dims
        self.num_vis_layers = 1 + 2 * self.num_translayers
        
    def tie_qk(self, module):
        if isinstance(module, CrossAttFeatTrans) and module.tie_qk_scheme != 'none':
            module.tie_qk()

    def add_identity_bias(self, module):
        if isinstance(module, CrossAttFeatTrans) or isinstance(module, MultiModeFeatTrans):
            module.add_identity_bias()

    # fake2D_batch: [48, 3, 112, 112]
    # nonzero_mask: [48, 14, 14]
    def get_mask(self, fake2D_batch):
        with torch.no_grad():
            avg_pooled_batch = self.mask_pool(fake2D_batch.abs())
            nonzero_mask = ( avg_pooled_batch.sum(dim=1) > 0 ).long()
        return nonzero_mask
    
    def in_fpn_forward(self, batch_base_feats, nonzero_mask, B, D2):
        # batch_base_feat3: [48, 256, 14, 14], batch_base_feat4: [48, 512, 7, 7]
        # batch_base_feat2: [48, 128, 28, 28]
        # nonzero_mask: if '3': [48, 14, 14]; if '2': [48, 28, 28].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.in_fpn_layers[0]]
        
        # curr_feat: [48, 128, 28, 28] -> [48, 256, 28, 28] -> [48, 512, 28, 28]
        #                   2                   3                    4
        for layer in self.in_fpn_layers[:-1]:
            upconv_feat = self.in_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer+1]
            if self.in_fpn_scheme == 'AN':
                # Using 'nearest' mode causes significant degradation.
                curr_feat           = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:], 
                                                                  mode='bilinear', 
                                                                  align_corners=False)
                curr_feat           = self.in_fpn_norms[layer+1](curr_feat)
            else:
                upconv_feat_normed  = self.in_fpn_norms[layer+1](upconv_feat)
                curr_feat           = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:], 
                                                                         mode='bilinear', 
                                                                         align_corners=False)
        
        batch_base_feat_fpn = self.in_fpn_bridgeconv(curr_feat)
            
        H2, W2 = batch_base_feat_fpn.shape[2:]
        # batch_base_feat_fpn:        [B, 20, 256, 14, 14]
        batch_base_feat_fpn      = batch_base_feat_fpn.view((B, D2) + batch_base_feat_fpn.shape[1:])
        # nonzero_mask:               [B, 20, 14, 14]
        batch_mask_fpn           = nonzero_mask.view((B, D2) + (H2, W2))
        # batch_base_feat_fpn_chwd:   [B, 256, 14, 14,  28]
        batch_base_feat_fpn_chwd = batch_base_feat_fpn.permute([0, 2, 3, 4, 1])
        batch_mask_fpn_hwd       = batch_mask_fpn.permute([0, 2, 3, 1])
        dpooled_shape            = list(batch_base_feat_fpn_chwd.shape[2:])
        dpooled_shape[2]         = dpooled_shape[2] // self.D_pool_K

        # Pooling along the depth to half size.
        batch_base_feat_fpn_chwd2 = F.interpolate(batch_base_feat_fpn_chwd, size=dpooled_shape, 
                                                  mode='trilinear', align_corners=False)
        batch_mask_fpn_hwd2       = F.interpolate(batch_mask_fpn_hwd.float(), size=dpooled_shape[1:], 
                                                  mode='bilinear',  align_corners=False)
        batch_mask_fpn_hwd2       = (batch_mask_fpn_hwd2 >= 0.5).long()
        
        # batch_base_feat_fpn_hwdc: [B, 14, 14,  14, 256]
        batch_base_feat_fpn_hwdc = batch_base_feat_fpn_chwd2.permute([0, 2, 3, 4, 1])
        # vfeat_fpn:                [B, 2744, 256]
        vfeat_fpn                = batch_base_feat_fpn_hwdc.reshape((B, -1, self.trans_in_dim))
        # vmask_fpn:                [B, 2744]
        vmask_fpn                = batch_mask_fpn_hwd2.reshape((B, -1))
        
        return vfeat_fpn, vmask_fpn, H2, W2

    def out_fpn_forward(self, batch_base_feats, vfeat_fused, B, D2):
        # batch_base_feat3: [48, 256, 14, 14], batch_base_feat4: [48, 512, 7, 7]
        # batch_base_feat2: [48, 128, 28, 28]
        # nonzero_mask: if '3': [48, 14, 14]; if '2': [48, 28, 28].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.out_fpn_layers[0]]
        curr_feat = curr_feat.view((B, D2) + curr_feat.shape[1:4])
        curr_feat = curr_feat.permute([0, 2, 3, 4, 1])
        # Only consider the extra layers in output fpn compared with input fpn, 
        # plus the last layer in input fpn.
        # If in: [3,4], out: [1,2,3,4], then out_fpn_layers=[1,2,3].
        out_fpn_layers = self.out_fpn_layers[:-len(self.in_fpn_layers)]

        # curr_feat: [2, 64, 56, 56, 24] -> [2, 128, 56, 56, 24] -> [2, 256, 56, 56, 24]
        #                     1                       2                        3
        for layer in out_fpn_layers:
            upconv_feat = self.out_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer+1]
            # higher_feat: [48, 128, 28, 28] => [2, 20, 128, 28, 28] => [2, 128, 28, 28, 24]
            higher_feat     = higher_feat.view((B, D2) + higher_feat.shape[1:4])
            higher_feat     = higher_feat.permute([0, 2, 3, 4, 1])
            if self.out_fpn_scheme == 'AN':
                curr_feat           = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:], 
                                                                  mode='trilinear', 
                                                                  align_corners=False)
                curr_feat           = self.out_fpn_norms[layer+1](curr_feat)
            else:
                upconv_feat_normed  = self.out_fpn_norms[layer+1](upconv_feat)
                curr_feat           = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:], 
                                                                         mode='trilinear', 
                                                                         align_corners=False)

        # curr_feat:   [2, 512, 56, 56, 24]
        # vfeat_fused: [2, 512, 14, 14, 24]
        out_feat_fpn = self.out_fpn_bridgeconv3d(curr_feat) + F.interpolate(vfeat_fused, size=curr_feat.shape[2:], 
                                                                        mode='trilinear', 
                                                                        align_corners=False)

        if self.D_pool_K > 1:
            if self.out_fpn_upsampleD_scheme == 'conv':
                out_feat_fpn_ups = self.out_fpn_upsampleD(out_feat_fpn)
                out_feat_fpn_ups = out_feat_fpn_ups.view((out_feat_fpn.shape[0], self.out_feat_dim, self.D_pool_K) + out_feat_fpn.shape[2:])
                out_feat_fpn_ups = out_feat_fpn_ups.permute([0, 1, 3, 4, 5, 2])
                feat_upsampleD_shape = list(out_feat_fpn_ups.shape[:5])
                feat_upsampleD_shape[4] = -1
                out_feat_fpn = out_feat_fpn_ups.reshape(feat_upsampleD_shape)
            elif self.out_fpn_upsampleD_scheme == 'interpolate':
                dunpooled_shape             = list(out_feat_fpn.shape[2:])
                dunpooled_shape[2]          = dunpooled_shape[2] * self.D_pool_K
                # out_feat_fpn: [2, 512, 56, 56, 48]
                out_feat_fpn = F.interpolate(out_feat_fpn, size=dunpooled_shape, 
                                             mode='trilinear', 
                                             align_corners=False)
            elif self.out_fpn_upsampleD_scheme == 'none':
                pass
                
        if self.out_fpn_do_dropout:
            out_feat_drop = self.out_fpn_dropout(out_feat_fpn)
            return out_feat_drop
        else:
            return out_feat_fpn
        
    def forward(self, batch):
        #                    B,  C, H,   W,   D
        # batch:            [B, 4, 112, 112, 80]
        B, C, H, W, D = batch.shape
        assert C == self.orig_in_channels
        
        if self.D_groupsize > 1:
            # D_split_shape1 = [B, C, 112, 112, -1, 2]
            D_split_shape   = (B, C, H, W, -1, self.D_groupsize)
            # batch_D_split: [B, 112, 112, 20, 2]
            batch_D_split   = batch.view(D_split_shape)
            batch_DChan     = batch_D_split.permute([0, 1, 5, 2, 3, 4])
            DChan_shape     = list(batch_DChan.shape)
            # Merge the depth groups into the channel dimension. 
            # This is why self.eff_in_channels = self.orig_in_channels * self.D_groupsize.
            DChan_shape[1:3] = [ DChan_shape[1] * DChan_shape[2] ]
            batch_DChan     = batch_DChan.view(DChan_shape)
        else:
            batch_DChan     = batch
        
        D2 = batch_DChan.shape[-1]
        # If inchan_to3_scheme != 'stemconv': batch_DChan: [B, 4, 112, 112, 96] => fakeRGB_batch: [B, 3, 112, 112, 96]
        # If inchan_to3_scheme == 'stemconv': fakeRGB_batch = batch_DChan = [B, 4, 112, 112, 96]
        fakeRGB_batch = self.in_bridge_to3(batch_DChan)
        # Merge depth to batch to make a 3D-image tensor (5D) to a 2D-image tensor (4D).
        # fake2D_batch: [B, 96, 3/4, 112, 112]
        fake2D_batch   = fakeRGB_batch.permute([0, 4, 1, 2, 3])
        # fake2D_shape = [-1, 3/4, 112, 112]
        fake2D_shape = (-1,) + fake2D_batch.shape[2:]
        # fake2D_batch: [B*96, 3/4, 112, 112]
        fake2D_batch = fake2D_batch.reshape(fake2D_shape)
        # nonzero_mask: if '3': [B*96, 14, 14]; if '2': [B*96, 28, 28].
        nonzero_mask = self.get_mask(fake2D_batch)
        
        if self.backbone_type.startswith('res'):
            batch_base_feats = self.backbone.ext_features(fake2D_batch)
        elif self.backbone_type.startswith('eff'):
            feats_dict = self.backbone.extract_endpoints(fake2D_batch)
            #                       [48, 24, 112, 112],        [48, 32,  56, 56]
            batch_base_feats = ( feats_dict['reduction_1'], feats_dict['reduction_2'], \
            #                       [48, 48, 28,  28],         [48, 136, 14, 14],       [48, 1536, 7, 7]
                                 feats_dict['reduction_3'], feats_dict['reduction_4'], feats_dict['reduction_5'] )
                                
        # vfeat_fpn: [B, 3920, 256]
        vfeat_fpn, vmask, H2, W2 = self.in_fpn_forward(batch_base_feats, nonzero_mask, B, D2)
        
        D3 = D2 // self.D_pool_K
        # if self.in_fpn_layers == '234', xyz_shape = (28, 28, 20)
        # if self.in_fpn_layers == '34',  xyz_shape = (14, 14, 20)
        xyz_shape = torch.Size((H2, W2, D3))
        # xyz_indices: [14, 14, 20, 3]
        xyz_indices =  get_all_indices(xyz_shape, device=self.device)
        model_scale_H = H // H2
        model_scale_W = W // W2
        model_scale_D = D // D3

        # Has to be exactly divided.
        if (model_scale_H * H2 != H) or (model_scale_W * W2 != W) or (model_scale_D * D3 != D):
            breakpoint()

        total_pos_scale = [ model_scale_H / self.input_scale[0], 
                            model_scale_W / self.input_scale[1], 
                            model_scale_D / self.input_scale[2] ]

        if not self.scales_printed:
            print("\nVoxels: %s. Model HWD scales: %dx%dx%d. Total scales: %s" % \
                  (list(vfeat_fpn.shape), model_scale_H, model_scale_W, model_scale_D, total_pos_scale))
            self.scales_printed = True
                      
        scale = torch.tensor([total_pos_scale], device='cuda')
        # xyz_indices: [3920, 3]
        # Rectify the scales on H, W, D.
        xyz_indices = xyz_indices.view([-1, 3]).float() * scale

        # voxels_pos: [B, 3920, 3], "3" is coordinates.
        voxels_pos = xyz_indices.unsqueeze(0).repeat((B, 1, 1))

        # vfeat_fused: [2, 3920, 256]
        vfeat_fused = self.voxel_fusion(vfeat_fpn, voxels_pos, vmask.unsqueeze(2))
        
        # vfeat_fused: [4, 14, 14, 12, 1024]
        vfeat_fused = vfeat_fused.view([B, H2, W2, D3, self.trans_out_dim])
        # vfeat_fused: [4, 1024, 14, 14, 12]
        vfeat_fused = vfeat_fused.permute([0, 4, 1, 2, 3])
        
        if self.do_out_fpn:
            vfeat_fused_fpn     = self.out_fpn_forward(batch_base_feats, vfeat_fused, B, D2)
        else:
            vfeat_fused_fpn = vfeat_fused

        # trans_scores_small: [B, 4, 56, 56, 48].
        trans_scores_small  = self.out_conv3d(vfeat_fused_fpn)
        out_size        = (H, W, D)     # [112, 112, 96]
        # Upsize trans_scores_small by 2. 
        # trans_scores_up: [B, 4, 112, 112, 96]. 
        trans_scores_up = F.interpolate(trans_scores_small, size=out_size, 
                                        mode='trilinear', align_corners=False)

        return trans_scores_up
        