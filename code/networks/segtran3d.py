import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import resnet
from efficientnet.model import EfficientNet
import networks.aj_i3d.aj_i3d as aj_i3d
from networks.aj_i3d.aj_i3d import InceptionI3d
from networks.segtran_shared import SegtranConfig, bb2feat_dims, SegtranFusionEncoder, CrossAttFeatTrans, ExpandedFeatTrans, \
                                    SegtranInitWeights, gen_all_indices
from train_util import batch_norm

# Only application-specific settings and overrode settings.
class Segtran3dConfig(SegtranConfig):
    def __init__(self):
        super(Segtran3dConfig, self).__init__()
        self.backbone_type = 'i3d'         # only i3d is supported.
        self.use_pretrained = True
        self.bb_feat_dims = bb2feat_dims[self.backbone_type]
        self.num_translayers = 1
        # vanilla transformer: 1, use_squeezed_transformer: 2.
        self.cross_attn_score_scales = [1., 2.]
        # Set in_fpn_scheme and out_fpn_scheme to 'NA' and 'NA', respectively.
        # NA: normalize first, then add. AN: add first, then normalize.
        self.set_fpn_layers('default', '34', '1234', 'AN', 'AN',
                            translayer_compress_ratios=[1,1], do_print=False)
        self.bb_feat_upsize  = True     # Configure backbone to generate x2 feature maps.
        self.in_fpn_use_bn   = False    # If in FPN uses BN, it performs slightly worse than using GN.
        self.out_fpn_use_bn  = False    # If out FPN uses BN, it performs worse than using GN.
        self.resnet_bn_to_gn = False    # Converting resnet BN to GN reduces performance.

        self.G = 8                      # number of groups in all group norms.
        self.pos_dim  = 3
        self.input_scale = (1., 1., 1.)
        self.num_classes = 2

        # Architecture settings
        self.num_attractors = 1024

        self.orig_in_channels = 1
        # 3D specific settings.
        # inchan_to3_scheme: 
        # avgto3 (average central two slices, yield 3 slices), only for effective slices == 2 or 4.
        # or stemconv (modifying stem conv to take 2-channel input). Not implemented for i3d;
        # or dup3 (duplicate 1 channel 3 times to fake RGB channels).
        # or bridgeconv (a conv to convert C > 1 channels to 3),
        # or None (only if effective channels == 3), do nothing.
        self.inchan_to3_scheme = 'bridgeconv'
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

CONFIG = Segtran3dConfig()

def set_segtran3d_config(args):
    CONFIG.num_classes                  = args.num_classes
    CONFIG.backbone_type                = args.backbone_type
    CONFIG.use_pretrained               = args.use_pretrained
    CONFIG.bb_feat_upsize               = args.bb_feat_upsize
    CONFIG.bb_feat_dims                 = bb2feat_dims[CONFIG.backbone_type]
    CONFIG.in_fpn_use_bn                = args.in_fpn_use_bn
    
    CONFIG.use_squeezed_transformer     = args.use_squeezed_transformer
    CONFIG.cross_attn_score_scale       = CONFIG.cross_attn_score_scales[CONFIG.use_squeezed_transformer]
    CONFIG.num_attractors               = args.num_attractors
    CONFIG.num_translayers              = args.num_translayers
    CONFIG.num_modes                    = args.num_modes
    CONFIG.trans_output_type            = args.trans_output_type
    CONFIG.mid_type                     = args.mid_type
    CONFIG.pos_code_every_layer         = args.pos_code_every_layer
    CONFIG.pos_in_attn_only             = args.pos_in_attn_only
    CONFIG.base_initializer_range       = args.base_initializer_range
    CONFIG.pos_code_type                = args.pos_code_type
    CONFIG.pos_code_weight              = args.pos_code_weight
    CONFIG.pos_bias_radius              = args.pos_bias_radius
    CONFIG.ablate_multihead             = args.ablate_multihead
    if 'dropout_prob' in args:
        CONFIG.hidden_dropout_prob          = args.dropout_prob
        CONFIG.attention_probs_dropout_prob = args.dropout_prob
    if 'out_fpn_do_dropout' in args:
        CONFIG.out_fpn_do_dropout           = args.out_fpn_do_dropout
    if 'perturb_posw_range' in args:
        CONFIG.perturb_posw_range           = args.perturb_posw_range
    if 'qk_have_bias' in args:
        CONFIG.qk_have_bias                 = args.qk_have_bias
                        
    CONFIG.has_FFN_in_squeeze           = args.has_FFN_in_squeeze
    CONFIG.attn_clip                    = args.attn_clip
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


class Segtran3d(SegtranInitWeights):
    def __init__(self, config):
        super(Segtran3d, self).__init__(config)
        self.config             = config
        self.device             = config.device
        self.orig_in_channels   = config.orig_in_channels
        self.trans_in_dim       = config.trans_in_dim
        self.trans_out_dim      = config.trans_out_dim
        self.num_translayers    = config.num_translayers
        self.bb_feat_upsize     = config.bb_feat_upsize
        self.G                  = config.G
        self.voxel_fusion       = SegtranFusionEncoder(config, 'Fusion')
        self.backbone_type      = config.backbone_type
        self.use_pretrained     = config.use_pretrained
        self.pos_code_every_layer = config.pos_code_every_layer
        if self.backbone_type.startswith('i3d'):
            self.backbone   = InceptionI3d(do_pool1=not self.bb_feat_upsize)
            print("%s created" %self.backbone_type)
            # if backbone_type == 'i3d-scratch', then do not load pretrained weights.
            if self.use_pretrained:
                i3d_folder = os.path.dirname(aj_i3d.__file__)
                pretrained_i3d_path = os.path.join(i3d_folder, "aj_rgb_imagenet.pth")
                state_dict = torch.load(pretrained_i3d_path, map_location=torch.device('cpu'))
                self.backbone.load_state_dict(state_dict)
                print("Loaded pretrained i3d model '{}'".format(pretrained_i3d_path))
        else:
            raise NotImplementedError("Only support i3d as the 3D backbone")

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
                raise NotImplementedError("Changing stemconv channel number is not supported for i3d")
            else:
                raise NotImplementedError("Effective input channel size={}*{} is not supported for scheme '{}'".format(
                                          self.orig_in_channels, self.D_groupsize, self.inchan_to3_scheme))
            
        self.in_fpn_use_bn  = config.in_fpn_use_bn
        self.in_fpn_layers  = config.in_fpn_layers
        self.in_fpn_scheme  = config.in_fpn_scheme

        # in_fpn_layers: default [3,4].
        # FPN output resolution is determined by the smallest number (lowest layer).
        if self.bb_feat_upsize:
            if 2 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool3d((2,4,4))
            elif 3 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool3d((4,8,8))
            else:
                self.mask_pool = nn.AvgPool3d((8,16,16))
        else:
            if 2 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool3d((2,8,8))
            elif 3 in self.in_fpn_layers:
                self.mask_pool = nn.AvgPool3d((4,16,16))
            else:
                # This resolution is too low. Put here for completeness.
                self.mask_pool = nn.AvgPool3d((8,32,32))
            
        self.bb_feat_dims = config.bb_feat_dims
        self.in_fpn23_conv  = nn.Conv3d(self.bb_feat_dims[2], self.bb_feat_dims[3], 1)
        self.in_fpn34_conv  = nn.Conv3d(self.bb_feat_dims[3], self.bb_feat_dims[4], 1)
        # Default in_fpn_layers: 34. last_in_fpn_layer_idx: 4.
        last_in_fpn_layer_idx = self.in_fpn_layers[-1]
        if self.bb_feat_dims[last_in_fpn_layer_idx] != self.trans_in_dim:
            self.in_fpn_bridgeconv = nn.Conv3d(self.bb_feat_dims[last_in_fpn_layer_idx], self.trans_in_dim, 1)
        else:
            self.in_fpn_bridgeconv = nn.Identity()

        # in_bn4b/in_gn4b normalizes in_fpn43_conv(layer 4 features),
        # so the feature dim = dim of layer 3.
        # in_bn3b/in_gn3b normalizes in_fpn32_conv(layer 3 features),
        # so the feature dim = dim of layer 2.
        if self.in_fpn_use_bn:
            self.in_bn3b = nn.BatchNorm3d(self.bb_feat_dims[3])
            self.in_bn4b = nn.BatchNorm3d(self.bb_feat_dims[4])
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
            # For i3d, even if D_pool_K == 2, out_fpn_upsampleD is not used. So the input feature dim is still trans_out_dim.
            self.out_conv3d      = nn.Conv3d(self.trans_out_dim, self.num_classes, 1)
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
        if isinstance(module, CrossAttFeatTrans) or isinstance(module, ExpandedFeatTrans):
            module.add_identity_bias()

    # batch: [4, 3, 96, 112, 112]
    # mask_pool default kernel size: (4, 8, 8)
    # nonzero_mask: [4, 24, 14, 14]
    def get_mask(self, batch):
        with torch.no_grad():
            avg_pooled_batch = self.mask_pool(batch.abs())
            nonzero_mask = ( avg_pooled_batch.sum(dim=1) > 0 ).long()
        return nonzero_mask

    def select_voxels_by_mask(self, vfeat, vmask):
        vfeat_list = []
        insts_num_voxels = []

        # B should be quite small (usually 2). Iteration would be still efficient.
        for b in range(vfeat.shape[0]):
            nonzero_vfeat = vfeat[b, vmask[b]]
            vfeat_list.append(nonzero_vfeat)
            insts_num_voxels.append(len(nonzero_vfeat))

        vfeat2 = nn.utils.rnn.pad_sequence(vfeat_list, batch_first=True)
        return vfeat2, insts_num_voxels

    def in_fpn_forward(self, batch_base_feats, nonzero_mask):
        # batch_base_feat1: [4, 192, 24, 56, 56], batch_base_feat2: [4, 480, 24, 28, 28]
        # batch_base_feat3: [4, 832, 12, 14, 14], batch_base_feat4: [4, 1024, 6, 7, 7]
        # nonzero_mask: [4, 12, 14, 14].
        feat0_pool, batch_base_feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.in_fpn_layers[0]]

        # if in_fpn_layers == [2, 3, 4]:
        # curr_feat: [4, 480, 24, 28, 28] -> [4, 832, 24, 28, 28] -> [4, 1024, 24, 28, 28]
        #                   2                         3                       4
        # if in_fpn_layers == [3, 4]:
        # curr_feat: [4, 832, 12, 14, 14] -> [4, 1024, 12, 14, 14]
        #                   3                       4
        
        for layer in self.in_fpn_layers[:-1]:
            upconv_feat = self.in_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer+1]
            if self.in_fpn_scheme == 'AN':
                # Using 'nearest' mode causes significant degradation.
                curr_feat           = upconv_feat + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                                  mode='trilinear',
                                                                  align_corners=False)
                curr_feat           = self.in_fpn_norms[layer+1](curr_feat)
            else:
                upconv_feat_normed  = self.in_fpn_norms[layer+1](upconv_feat)
                curr_feat           = upconv_feat_normed + F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                                         mode='trilinear',
                                                                         align_corners=False)

        batch_base_feat_fpn = self.in_fpn_bridgeconv(curr_feat)
        dpooled_shape       = list(batch_base_feat_fpn.shape[2:])
        dpooled_shape[0]    = dpooled_shape[0] // self.D_pool_K

        # Pooling along the depth to half size.
        batch_base_feat_fpn = F.interpolate(batch_base_feat_fpn, size=dpooled_shape, 
                                            mode='trilinear', align_corners=False)
        nonzero_mask        = F.interpolate(nonzero_mask.float().unsqueeze(1), size=dpooled_shape, 
                                            mode='trilinear',  align_corners=False)
        nonzero_mask        = (nonzero_mask.squeeze(1) >= 0.5).long()
                                                    
        # FeatDim: self.trans_in_dim, 1024.
        B, FeatDim, D2, H2, W2 = batch_base_feat_fpn.shape
        # batch_base_feat_fpn_hwdc: [B, 12, 14, 14, 1024]
        batch_base_feat_fpn_hwdc = batch_base_feat_fpn.permute([0, 2, 3, 4, 1])
        # vfeat_fpn:                [B, 2352, 1024]
        vfeat_fpn                = batch_base_feat_fpn_hwdc.reshape((B, -1, FeatDim))
        # vmask_fpn:                [B, 2352]
        vmask_fpn                = nonzero_mask.reshape((B, -1))

        return vfeat_fpn, vmask_fpn, D2, H2, W2

    def out_fpn_forward(self, batch_base_feats, vfeat_fused):
        # batch_base_feat1: [4, 192, 24, 56, 56], batch_base_feat2: [4, 480, 24, 28, 28]
        # batch_base_feat3: [4, 832, 12, 14, 14], batch_base_feat4: [4, 1024, 6, 7, 7]
        feat0_pool, batch_base_feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.out_fpn_layers[0]]
        # Only consider the extra layers in output fpn compared with input fpn.
        # If in: [3,4], out: [1,2,3,4], then out_fpn_layers=[1,2].
        # By default, out_fpn_layers=[1,2].
        out_fpn_layers = self.out_fpn_layers[:-len(self.in_fpn_layers)]

        # curr_feat: [4, 192, 24, 56, 56] -> [4, 480, 24, 56, 56] -> [4, 832, 24, 56, 56]
        #                     1                        2                      3
        for layer in out_fpn_layers:
            upconv_feat = self.out_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer+1]
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

        # curr_feat:    [4, 832,  24, 56, 56]
        # vfeat_fused:  [4, 1024, 12, 14, 14]
        # out_feat_fpn: [4, 1024, 24, 56, 56]
        out_feat_fpn = self.out_fpn_bridgeconv3d(curr_feat) + \
                         F.interpolate(vfeat_fused, size=curr_feat.shape[2:],
                                       mode='trilinear',
                                       align_corners=False)

        if self.backbone_type.startswith('i3d'):
            # For i3d, if bb_feat_upsize==True, then feature maps from layers 1 and 2 
            # have the same downsampling rates (2,2,2) for D, H, W. No need to upsample D.
            if self.D_pool_K > 1 and not self.bb_feat_upsize:
                if self.out_fpn_upsampleD_scheme == 'conv':
                    out_feat_fpn_ups = self.out_fpn_upsampleD(out_feat_fpn)
                    # Divide along feature dim to two chunks. One for elem 0 in depth groups, and the other for elem 1 in depth groups.
                    out_feat_fpn_ups = out_feat_fpn_ups.view((out_feat_fpn.shape[0], self.out_feat_dim, self.D_pool_K) + out_feat_fpn.shape[2:])
                    feat_upsampleD_shape = list(out_feat_fpn_ups.shape)
                    feat_upsampleD_shape[2:4] = [feat_upsampleD_shape[2] * feat_upsampleD_shape[3]]
                    out_feat_fpn = out_feat_fpn_ups.reshape(feat_upsampleD_shape)
                elif self.out_fpn_upsampleD_scheme == 'interpolate':
                    dunpooled_shape             = list(out_feat_fpn.shape[2:])
                    dunpooled_shape[0]          = dunpooled_shape[0] * self.D_pool_K
                    # out_feat_fpn: [4, 1024, 48, 56, 56]
                    out_feat_fpn = F.interpolate(out_feat_fpn, size=dunpooled_shape, 
                                                 mode='trilinear', 
                                                 align_corners=False)
                elif self.out_fpn_upsampleD_scheme == 'none':
                    pass
        else:
            breakpoint()
               
        if self.out_fpn_do_dropout:
            out_feat_drop = self.out_fpn_dropout(out_feat_fpn)
            return out_feat_drop
        else:
            return out_feat_fpn

    def forward(self, batch):
        #                    B, C,  H,   W,   D
        # batch:            [B, 4, 112, 112, 96]

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

        # batch_DChan: [B, 4, 112, 112, 96] => fakeRGB_batch: [B, 3, 112, 112, 96]            
        fakeRGB_batch = self.in_bridge_to3(batch_DChan)
        # Depth -> Frame dimension. B, C, H, W, D => B, C, D, H, W
        fakeRGB_batch = fakeRGB_batch.permute([0, 1, 4, 2, 3])
        # nonzero_mask: if '3': [24, 14, 14]; if '2': [48, 28, 28].
        nonzero_mask = self.get_mask(fakeRGB_batch)
        
        if self.backbone_type.startswith('i3d'):
            feats_dict = self.backbone.extract_features(fakeRGB_batch)
            #                       [4, 64, 24, 56, 56],          [B, 192, 24, 56, 56],        
            batch_base_feats = ( feats_dict['MaxPool3d_2a_3x3'], feats_dict['Conv3d_2c_3x3'], \
            #                       [B, 480, 24, 28, 28],   [B, 832, 12, 14, 14], [B, 1024, 6, 7, 7]
                                 feats_dict['Mixed_3c'], feats_dict['Mixed_4f'], feats_dict['Mixed_5c'] )
                                 
        # vfeat_fpn: [B, 3920, 256]
        # D2, H2, W2: feature map size before flattening
        vfeat_fpn, vmask, D2, H2, W2 = self.in_fpn_forward(batch_base_feats, nonzero_mask)

        # xyz_* are actually zxy_* (depth, height, width). 
        # Keep the names to make them compatible with the 2.5D code.
        # if self.in_fpn_layers == '234', xyz_shape = (28, 28, 20)
        # if self.in_fpn_layers == '34',  xyz_shape = (14, 14, 20)
        xyz_shape = torch.Size((D2, H2, W2))
        # xyz_indices: [14, 14, 20, 3]
        xyz_indices =  gen_all_indices(xyz_shape, device=self.device)
        model_scale_H = H // H2
        model_scale_W = W // W2
        model_scale_D = D // D2

        # Has to be exactly divided.
        if (model_scale_H * H2 != H) or (model_scale_W * W2 != W) or (model_scale_D * D2 != D):
            breakpoint()

        # input_scale is usually 1 for 3D images.
        total_pos_scale = [ model_scale_D / self.input_scale[2], 
                            model_scale_H / self.input_scale[0],
                            model_scale_W / self.input_scale[1]
                          ]

        if not self.scales_printed:
            print("\nFeat: %s, Voxels: %s. Model DHW scales: %dx%dx%d. Total scales: %s" % \
                  (list(xyz_shape), list(vfeat_fpn.shape), model_scale_D, model_scale_H, model_scale_W, total_pos_scale))
            self.scales_printed = True

        scale = torch.tensor([total_pos_scale], device='cuda')
        # xyz_indices: [3920, 3]. 
        # Rectify the scales on H, W, D. The indices now are pixel coordinates in the original input image.
        xyz_indices = xyz_indices.view([-1, 3]).float() * scale

        # voxels_pos: [B, 3920, 3], "3" is coordinates.
        voxels_pos = xyz_indices.unsqueeze(0).repeat((B, 1, 1))

        # vfeat_fused: [4, 2352, 1024]
        vfeat_fused = self.voxel_fusion(vfeat_fpn, voxels_pos, vmask.unsqueeze(2), xyz_shape)

        # vfeat_fused: [4, 12, 14, 14, 1024]
        vfeat_fused = vfeat_fused.view([B, D2, H2, W2, self.trans_out_dim])
        # vfeat_fused: [4, 1024, 12, 14, 14]
        vfeat_fused = vfeat_fused.permute([0, 4, 1, 2, 3])

        if self.do_out_fpn:
            vfeat_fused_fpn = self.out_fpn_forward(batch_base_feats, vfeat_fused)
        else:
            vfeat_fused_fpn = vfeat_fused
        
        # features swapped back to (height, width, depth): [4, 1024, 56, 56, 24 or 48]
        vfeat_fused_fpn = vfeat_fused_fpn.permute([0, 1, 3, 4, 2])
        # trans_scores_small: [B, 4, 56, 56, 48].
        trans_scores_small  = self.out_conv3d(vfeat_fused_fpn)
        
        out_size        = (H, W, D)     # [112, 112, 96]
        # Upsize trans_scores_small by 2. 
        # trans_scores_up: [B, 4, 112, 112, 96]. 
        trans_scores_up = F.interpolate(trans_scores_small, size=out_size,
                                        mode='trilinear', align_corners=False)

        return trans_scores_up
