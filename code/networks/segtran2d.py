import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import resnet
from efficientnet.model import EfficientNet
from networks.segtran_shared import SegtranConfig, bb2feat_dims, SegtranFusionEncoder, CrossAttFeatTrans, ExpandedFeatTrans, \
                                    SegtranInitWeights, gen_all_indices
from train_util import batch_norm
from timm.models import tf_efficientnetv2_s_in21k, tf_efficientnetv2_m_in21k, tf_efficientnetv2_l_in21k

class Segtran2dConfig(SegtranConfig):
    def __init__(self):
        super(Segtran2dConfig, self).__init__()
        self.backbone_type = 'eff-b4'         # resnet50, resnet101, resibn101, eff-b1~b4
        self.use_pretrained = True        
        self.bb_feat_dims = bb2feat_dims[self.backbone_type]
        self.num_translayers = 1
        # Set in_fpn_scheme and out_fpn_scheme to 'NA' and 'NA', respectively.
        # NA: normalize first, then add. AN: add first, then normalize.
        self.set_fpn_layers('default', '34', '1234', 'AN', 'AN',
                            translayer_compress_ratios=[1,1], do_print=False)
        self.bb_feat_upsize   = True     # Configure efficient net to generate x2 feature maps.
        self.in_fpn_use_bn    = False    # If in FPN uses BN, it performs slightly worse than using GN.
        self.out_fpn_use_bn   = False    # If out FPN uses BN, it performs worse than using GN.
        self.resnet_bn_to_gn  = False    # Converting resnet BN to GN reduces performance.
        self.posttrans_use_bn = False    # Output features from transformers are processed with BN.
        self.G = 8                       # number of groups in all group norms.
        self.pos_dim  = 2

        self.num_classes = 2
        # num_modalities = 0 means there's not the modality dimension
        # (i.e. images are in one modality only). Not to use num_modalities = 1,
        # so as to to distingush from the case where there's a modality dimension
        # in which only one modality presents.
        self.num_modalities = 0
        self.out_features = False
        
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

CONFIG = Segtran2dConfig()

def set_segtran2d_config(args):
    CONFIG.num_classes                  = args.num_classes
    CONFIG.backbone_type                = args.backbone_type
    CONFIG.use_pretrained               = args.use_pretrained
    CONFIG.bb_feat_upsize               = args.bb_feat_upsize
    CONFIG.bb_feat_dims                 = bb2feat_dims[CONFIG.backbone_type]
    CONFIG.in_fpn_use_bn                = args.in_fpn_use_bn
    CONFIG.num_modalities               = args.num_modalities
    CONFIG.use_squeezed_transformer     = args.use_squeezed_transformer
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
    if 'out_features' in args:
        CONFIG.out_features                 = args.out_features
    if 'eval_robustness' in args:
        CONFIG.eval_robustness              = args.eval_robustness
    if 'qk_have_bias' in args:
        CONFIG.qk_have_bias                 = args.qk_have_bias
        
    CONFIG.has_FFN_in_squeeze           = args.has_FFN_in_squeeze
    CONFIG.attn_clip                    = args.attn_clip
    CONFIG.set_fpn_layers('args', args.in_fpn_layers, args.out_fpn_layers,
                          args.in_fpn_scheme, args.out_fpn_scheme,
                          translayer_compress_ratios=args.translayer_compress_ratios)

    CONFIG.tie_qk_scheme                = args.tie_qk_scheme
    CONFIG.device                       = args.device
    CONFIG.use_global_bias              = args.use_global_bias
        
    return CONFIG

class Segtran2d(SegtranInitWeights):
    def __init__(self, config):
        super(Segtran2d, self).__init__(config)
        self.config         = config
        self.device         = config.device
        self.trans_in_dim   = config.trans_in_dim
        self.trans_out_dim  = config.trans_out_dim
        self.num_translayers = config.num_translayers
        self.bb_feat_upsize = config.bb_feat_upsize
        self.G              = config.G
        self.use_global_bias = config.use_global_bias
        self.out_features   = config.out_features
        
        if not self.use_global_bias:
            self.voxel_fusion   = SegtranFusionEncoder(config, 'Fusion')
            self.vfeat_bias     = None
            self.vfeat_bias_norm_layer = nn.Identity()
        else:
            self.vfeat_bias     = Parameter(torch.randn(1, 1, self.trans_out_dim))
            self.vfeat_bias_norm_layer = nn.LayerNorm(self.trans_out_dim, elementwise_affine=True)
            
        self.backbone_type  = config.backbone_type
        self.use_pretrained = config.use_pretrained
        self.pos_code_every_layer = config.pos_code_every_layer
        if self.backbone_type.startswith('resnet'):
            self.backbone   = resnet.__dict__[self.backbone_type](pretrained=self.use_pretrained, 
                                                                  do_pool1=not self.bb_feat_upsize)
            print("%s created" %self.backbone_type)
        # EfficientNet-V1
        elif self.backbone_type.startswith('eff-'):
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
        elif self.backbone_type.startswith('effv2'):
            backbone_type   = self.backbone_type.replace("effv2", "tf_efficientnetv2_")
            stem_stride     = 1 if self.bb_feat_upsize else 2
            if self.backbone_type[-1] == 's':
                self.backbone = tf_efficientnetv2_s_in21k(pretrained=self.use_pretrained, features_only=True)
            if self.backbone_type[-1] == 'm':
                self.backbone = tf_efficientnetv2_m_in21k(pretrained=self.use_pretrained, features_only=True)
            if self.backbone_type[-1] == 'l':
                self.backbone = tf_efficientnetv2_l_in21k(pretrained=self.use_pretrained, features_only=True)
            
            self.backbone.conv_stem.stride = (stem_stride, stem_stride)
            print("{} created (stem_stride={})".format(backbone_type, stem_stride))

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
        self.num_modalities = config.num_modalities
        if self.num_modalities > 0:
            self.mod_fuse_conv = nn.Conv2d(self.num_modalities, 1, 1)

        self.out_fpn_use_bn = config.out_fpn_use_bn
        self.out_fpn_layers = config.out_fpn_layers
        self.out_fpn_scheme = config.out_fpn_scheme
        self.out_fpn_do_dropout = config.out_fpn_do_dropout
        self.posttrans_use_bn   = config.posttrans_use_bn
        
        if self.out_fpn_layers != self.in_fpn_layers:
            self.do_out_fpn = True

            self.out_fpn12_conv = nn.Conv2d(self.bb_feat_dims[1],
                                            self.bb_feat_dims[2], 1)
            self.out_fpn23_conv = nn.Conv2d(self.bb_feat_dims[2],
                                            self.bb_feat_dims[3], 1)
            self.out_fpn34_conv = nn.Conv2d(self.bb_feat_dims[3],
                                            self.bb_feat_dims[4], 1)
            # Default in_fpn_layers: 34, out_fpn_layers: 1234. last_out_fpn_layer_idx: 3.
            last_out_fpn_layer_idx = self.out_fpn_layers[-len(self.in_fpn_layers)]
            if self.bb_feat_dims[last_out_fpn_layer_idx] != self.trans_out_dim:
                self.out_fpn_bridgeconv = nn.Conv2d(self.bb_feat_dims[last_out_fpn_layer_idx], self.trans_out_dim, 1)
            else:
                self.out_fpn_bridgeconv = nn.Identity()
                
            # out_bn3b/out_gn3b normalizes out_fpn23_conv(layer 3 features),
            # so the feature dim = dim of layer 2.
            # out_bn2b/out_gn2b normalizes out_fpn12_conv(layer 2 features),
            # so the feature dim = dim of layer 1.
            if self.out_fpn_use_bn:
                self.out_bn2b       = nn.BatchNorm2d(self.bb_feat_dims[2])
                self.out_bn3b       = nn.BatchNorm2d(self.bb_feat_dims[3])
                self.out_bn4b       = nn.BatchNorm2d(self.bb_feat_dims[4])
                self.out_fpn_norms  = [ None, None, self.out_bn2b, self.out_bn3b, self.out_bn4b ]
            else:
                self.out_gn2b       = nn.GroupNorm(self.G, self.bb_feat_dims[2])
                self.out_gn3b       = nn.GroupNorm(self.G, self.bb_feat_dims[3])
                self.out_gn4b       = nn.GroupNorm(self.G, self.bb_feat_dims[4])
                self.out_fpn_norms  = [ None, None, self.out_gn2b, self.out_gn3b, self.out_gn4b ]

            self.out_fpn_convs   = [ None, self.out_fpn12_conv, self.out_fpn23_conv, self.out_fpn34_conv ]
            self.out_conv        = nn.Conv2d(self.trans_out_dim, self.num_classes, 1)
            self.out_fpn_dropout = nn.Dropout(config.hidden_dropout_prob)
        # out_fpn_layers = in_fpn_layers, no need to do fpn at the output end.
        # Output class scores directly.
        else:
            self.do_out_fpn = False
            if '2' in self.in_fpn_layers:
                # Output resolution is 1/4 of input already. No need to do upsampling here.
                self.out_conv = nn.Conv2d(config.trans_out_dim, self.num_classes, 1)
            else:
                # Output resolution is 1/8 of input. Do upsampling to make resolution x 2
                self.out_conv = nn.ConvTranspose2d(config.trans_out_dim, self.num_classes,
                                                   2, 2)

        self.apply(self.init_weights)
        # tie_qk() has to be executed after weight initialization.
        self.apply(self.tie_qk)
        self.apply(self.add_identity_bias)
        # Initialize mod_fuse_conv weights and bias.
        # Set all modalities to have equal weights.
        if self.num_modalities > 0:
            self.mod_fuse_conv.weight.data.fill_(1/self.num_modalities)
            self.mod_fuse_conv.bias.data.zero_()

        self.scales_printed = False
        self.translayer_dims = config.translayer_dims
        if not self.use_global_bias:
            self.num_vis_layers = 1 + 2 * self.num_translayers
        else:
            self.num_vis_layers = 1
        
    # batch:        [B, 3, 112, 112]
    # nonzero_mask: [B, 36, 36]
    def get_mask(self, batch):
        with torch.no_grad():
            avg_pooled_batch = self.mask_pool(batch.abs())
            nonzero_mask = avg_pooled_batch.sum(dim=1) > 0
        return nonzero_mask

    def in_fpn_forward(self, batch_base_feats, nonzero_mask, B):
        # batch_base_feat3: [B, 256, 18, 18], batch_base_feat4: [B, 512, 9, 9]
        # batch_base_feat2: [B, 128, 36, 36]
        # nonzero_mask: if '3': [B, 18, 18]; if '2': [B, 36, 36].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.in_fpn_layers[0]]

        # curr_feat: [B, 128, 36, 36] -> [B, 256, 36, 36] -> [B, 512, 36, 36]
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
        # batch_base_feat_fpn:        [B, 512, 36, 36]
        # batch_base_feat_fpn_hwc:    [B, 28,  28, 512]
        batch_base_feat_fpn_hwc = batch_base_feat_fpn.permute([0, 2, 3, 1])
        # vfeat_fpn:            [B, 784, 512]
        vfeat_fpn               = batch_base_feat_fpn_hwc.reshape((B, -1, self.trans_in_dim))
        # nonzero_mask:         [B, 36, 36]
        # vmask_fpn:            [B, 784]
        vmask_fpn               = nonzero_mask.reshape((B, -1))

        return vfeat_fpn, vmask_fpn, H2, W2

    def out_fpn_forward(self, batch_base_feats, vfeat_fused, B0):
        # batch_base_feat3: [B, 256, 14, 14], batch_base_feat4: [B, 512, 7, 7]
        # batch_base_feat2: [B, 128, 36, 36]
        # nonzero_mask: if '3': [B, 14, 14]; if '2': [B, 36, 36].
        feat0_pool, feat1, batch_base_feat2, batch_base_feat3, batch_base_feat4 = batch_base_feats
        curr_feat = batch_base_feats[self.out_fpn_layers[0]]
        # Only consider the extra layers in output fpn compared with input fpn,
        # plus the last layer in input fpn.
        # If in: [3,4], out: [1,2,3,4], then out_fpn_layers=[1,2,3].
        out_fpn_layers = self.out_fpn_layers[:-len(self.in_fpn_layers)]

        # curr_feat: [2, 64, 56, 56] -> [2, 128, 56, 56] -> [2, 256, 56, 56]
        #                 1                  2                   3
        for layer in out_fpn_layers:
            upconv_feat = self.out_fpn_convs[layer](curr_feat)
            higher_feat = batch_base_feats[layer+1]
            if self.out_fpn_scheme == 'AN':
                curr_feat           = upconv_feat + \
                                       F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                     mode='bilinear',
                                                     align_corners=False)
                curr_feat           = self.out_fpn_norms[layer+1](curr_feat)
            else:
                upconv_feat_normed  = self.out_fpn_norms[layer+1](upconv_feat)
                curr_feat           = upconv_feat_normed + \
                                       F.interpolate(higher_feat, size=upconv_feat.shape[2:],
                                                     mode='bilinear',
                                                     align_corners=False)

        # curr_feat:   [2, 512, 56, 56]
        # vfeat_fused: [2, 512, 36, 36]
        out_feat_fpn = self.out_fpn_bridgeconv(curr_feat) + \
                        F.interpolate(vfeat_fused, size=curr_feat.shape[2:],
                                      mode='bilinear', align_corners=False)

        if self.out_fpn_do_dropout:
            out_feat_drop = self.out_fpn_dropout(out_feat_fpn)
            return out_feat_drop
        else:
            return out_feat_fpn

    def forward(self, batch):
        # feature_maps is for visualization with https://github.com/fornaxai/receptivefield
        self.feature_maps = []

        #                    B,  C, H,   W
        # batch:            [B0, 1, 112, 112]
        # => padded_batch   [B0, 1, 112, 112]
        if self.num_modalities > 0:
            B0, C, H, W, MOD = batch.shape
            batch_bm = batch.permute([0, 4, 1, 2, 3])
            batch_bm = batch.view([B0*MOD, C, H, W])
            # Merge the modality dim into the batch dim.
            # Then go through ordinary preprocessing (splitting of depth, merging depth into batch).
            # After extracting features, merge MOD sets of feature maps into one set.
            batch = batch_bm
        else:
            # MOD = 0 means there's not the modality dimension
            # (i.e. images are in one modality only). Didn't use MOD = 1 to distingush from
            # the case that there's a modality dimension containing only one modality.
            MOD = 0
            B0 = batch.shape[0]

        B, C, H, W = batch.shape

        # nonzero_mask: if '3': [B, 14, 14]; if '2': [B, 36, 36].
        nonzero_mask = self.get_mask(batch)

        if self.backbone_type.startswith('res'):
            batch_base_feats = self.backbone.ext_features(batch)
        elif self.backbone_type.startswith('eff-'):
            feats_dict = self.backbone.extract_endpoints(batch)
            #                       [10, 16, 288, 288],        [10, 24, 144, 144]
            batch_base_feats = ( feats_dict['reduction_1'], feats_dict['reduction_2'], \
            #                       [10, 40, 72, 72],          [10, 112, 36, 36],       [10, 1280, 18, 18]
                                 feats_dict['reduction_3'], feats_dict['reduction_4'], feats_dict['reduction_5'] )
            # Corresponding stages in efficient-net paper, Table 1: 2, 3, 4, 6, 9
        elif self.backbone_type.startswith('effv2'):
            batch_base_feats = self.backbone(batch)
            #                       [10, 24, 288, 288],        [10, 48, 144, 144]
            #                       [10, 80, 72, 72],          [10, 176, 36,  36],       [10, 512, 18, 18]
            # Corresponding stages: 1, 2, 3, 5, 7

        # vfeat_fpn: [B (B0*MOD), 1296, 1792]
        vfeat_fpn, vmask, H2, W2 = self.in_fpn_forward(batch_base_feats, nonzero_mask, B)
        vfeat_origshape = vfeat_fpn.transpose(1, 2).view(B, -1, H2, W2)
        self.feature_maps.append(vfeat_origshape)

        if self.num_modalities > 0:
            # vfeat_fpn_MOD: [B0, MOD, 1296, 1792]
            vfeat_fpn_MOD = vfeat_fpn.view(B0, MOD, -1, self.trans_in_dim)
            # vfeat_fpn: [B0, 1296, 1792]
            # vfeat_fpn = self.mod_fuse_conv(vfeat_fpn_MOD).squeeze(1)
            vfeat_fpn = vfeat_fpn_MOD.max(dim=1)[0]
            # No need to normalize features here. Each feature vector in vfeat_fpn
            # will be layer-normed in SegtranInputFeatEncoder.

        # if self.in_fpn_layers == '234', xy_shape = (36, 36)
        # if self.in_fpn_layers == '34',  xy_shape = (14, 14)
        xy_shape = torch.Size((H2, W2))
        # xy_indices: [14, 14, 20, 3]
        xy_indices =  gen_all_indices(xy_shape, device=self.device)
        scale_H = H // H2
        scale_W = W // W2

        # Has to be exactly divided.
        if (scale_H * H2 != H) or (scale_W * W2 != W):
            breakpoint()

        if not self.scales_printed:
            print("\nImage scales: %dx%d. Feat: %s. Voxels: %s" %(scale_H, scale_W, list(xy_shape), list(vfeat_fpn.shape)))
            self.scales_printed = True

        scale = torch.tensor([[scale_H, scale_W]], device=self.device)
        # xy_indices: [1296, 2]
        # Rectify the scales on H, W.
        xy_indices = xy_indices.view([-1, 2]).float() * scale

        # voxels_pos: [B0, 1296, 2], "2" is coordinates.
        voxels_pos = xy_indices.unsqueeze(0).repeat((B0, 1, 1))

        # pos_embed = self.featemb(voxels_pos)
        # vfeat_fused: [2, 784, 1792]
        switch_to_vfeat_bias_after_first_iter = False
        if not self.use_global_bias:
            vfeat_fused = self.voxel_fusion(vfeat_fpn, voxels_pos, vmask.unsqueeze(2), xy_shape)
            for i in range(self.num_translayers):
                self.feature_maps.append(self.voxel_fusion.translayers[i].attention_scores)
            for i in range(self.num_translayers):
                layer_vfeat = self.voxel_fusion.layers_vfeat[i]
                layer_vfeat = layer_vfeat.view([B0, H2, W2, self.translayer_dims[i+1]])
                # layer_vfeat: [5, 32, 32, 1792] => [5, 1792, 32, 32]
                layer_vfeat = layer_vfeat.permute([0, 3, 1, 2])
                self.feature_maps.append(layer_vfeat)    
            if switch_to_vfeat_bias_after_first_iter:
                self.vfeat_bias = vfeat_fused.data.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True)
                self.use_global_bias = True
        else:
            vfeat_fused_expand_shape = list(vfeat_fpn.shape)
            vfeat_fused_expand_shape[2] = self.vfeat_bias.shape[2]
            vfeat_fused = self.vfeat_bias_norm_layer(self.vfeat_bias).expand(vfeat_fused_expand_shape)

        # vfeat_fused: [2, 32, 32, 1792]
        vfeat_fused = vfeat_fused.view([B0, H2, W2, self.trans_out_dim])
        # vfeat_fused: [5, 32, 32, 1792] => [5, 1792, 32, 32]
        vfeat_fused = vfeat_fused.permute([0, 3, 1, 2])

        if self.do_out_fpn:
            vfeat_fused = self.out_fpn_forward(batch_base_feats, vfeat_fused, B0)
            if self.posttrans_use_bn:
                vfeat_fused = batch_norm(vfeat_fused)
            trans_scores_small  = self.out_conv(vfeat_fused)
        else:
            # scores: [B0, 2, 36, 36]
            # if vfeat_fpn is already 28*28 (in_fpn_layers=='234'),
            # then out_conv does not do upsampling.
            if self.posttrans_use_bn:
                vfeat_fused = batch_norm(vfeat_fused)
            trans_scores_small  = self.out_conv(vfeat_fused)

        # full_scores: [B0, 2, 112, 112]
        trans_scores_up = F.interpolate(trans_scores_small, size=(H, W),
                                        mode='bilinear', align_corners=False)

        if self.out_features:
            return trans_scores_up, vfeat_fused
        else:
            return trans_scores_up
