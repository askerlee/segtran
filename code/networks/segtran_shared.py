import math
import random
from re import X
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from networks.segtran_ablation import RandPosEmbedder, SinuPosEmbedder, ZeroEmbedder, MultiHeadFeatTrans
torch.set_printoptions(sci_mode=False)


bb2feat_dims = { 'resnet34':  [64, 64,  128, 256,   512],
                 'resnet50':  [64, 256, 512, 1024, 2048],
                 'resnet101': [64, 256, 512, 1024, 2048],
                 'resibn101': [64, 256, 512, 1024, 2048],   # resibn: resnet + IBN layers
                 'eff-b0':    [16, 24,  40,  112,  1280],   # input: 224
                 'eff-b1':    [16, 24,  40,  112,  1280],   # input: 240
                 'eff-b2':    [16, 24,  48,  120,  1408],   # input: 260
                 'eff-b3':    [24, 32,  48,  136,  1536],   # input: 300
                 'eff-b4':    [24, 32,  56,  160,  1792],   # input: 380
                 'effv2m':    [24, 48,  80,  176,   512],   # input: 384
                 'i3d':       [64, 192, 480, 832,  1024]    # input: 224
               }
               
def gen_all_indices(shape, device):
    indices = torch.arange(shape.numel(), device=device).view(shape)

    out = []
    for dim_size in reversed(shape):
        out.append(indices % dim_size)
        # If using indices = indices // dim_size, pytorch will prompt a warning.
        indices = torch.div(indices, dim_size, rounding_mode='trunc')
    return torch.stack(tuple(reversed(out)), len(shape))

def multi_resize_shape(shape, scales):
    resized_shapes = []
    for scale in scales:
        resized_shape = [ int(s * scale) for s in shape ]
        resized_shapes.append(resized_shape)
    return resized_shapes

# First reshape flattened x into geoshape, then scale, then flatten.
def resize_flat_features(x, geoshape, scale):
    interp_modes = ('linear', 'bilinear', 'trilinear')
    x_shape0 = list(x.shape)
    # x.shape: [B, N0, C]. x_shape0: [B, -1, C].
    # New number of elements is N0 * (scale^d), where d is the number of geometric dimensions.
    x_shape0[-2] = -1
    x_shape = list(x.shape)
    x_shape[-2:-1] = geoshape
    x = x.reshape(x_shape)
    if len(geoshape) <= 3:
        interp_mode = interp_modes[len(geoshape) - 1]
        x = F.interpolate(x, scale_factor=scale, mode=interp_mode, align_corners=False)
        x = x.reshape(x_shape0)
    else:
        breakpoint()    # >3D features are not supported yet.

    return x

# Application-independent configurations.
class SegtranConfig:
    def __init__(self):
        # Architecture settings
        self.feat_dim       = -1
        self.in_feat_dim    = -1
        # Number of modes in the expansion attention block.
        # When doing ablation study of multi-head, num_modes means num_heads, 
        # to avoid introducing extra config parameters.
        self.num_modes = 4
        # Use AttractorAttFeatTrans instead of the vanilla CrossAttFeatTrans.
        self.use_squeezed_transformer = True
        self.num_attractors = 256
        self.tie_qk_scheme = 'shared'           # shared, loose, or none.
        self.mid_type      = 'shared'           # shared, private, or none.
        self.trans_output_type  = 'private'     # shared or private.
        self.act_fun = F.gelu
        self.has_FFN = True                 # Only used in SqueezedAttFeatTrans
        self.has_FFN_in_squeeze = False     # Seems to slightly improve accuracy, and reduces RAM and computation
        
        # Positional encoding settings.
        self.pos_code_type      = 'lsinu'
        self.pos_code_weight    = 1.
        self.pos_bias_radius    = 7
        self.max_pos_size       = (100, 100)
        self.pos_in_attn_only = False
        self.pos_code_every_layer = True

        # Removing biases from QK seems to slightly degrade performance.
        self.qk_have_bias = True
        # Removing biases from V seems to slightly improve performance.
        self.v_has_bias   = False
        
        self.attn_clip = 500
        self.base_initializer_range = 0.02
        # Add an identity matrix (*0.02*query_idbias_scale) to query/key weights
        # to make a bias towards identity mapping.
        # Set to 0 to disable the identity bias.
        self.query_idbias_scale = 10
        self.feattrans_lin1_idbias_scale = 10

        # Pooling settings
        self.pool_modes_feat  = 'softmax'   # softmax, max, mean, or none. 

        # Mince transformer settings
        self.use_mince_transformer  = False
        self.mince_scales           = None  # [4, 3, 2, 1]
        self.mince_channel_props    = None  # [1, 1, 1, 1] will be normalized to [0.25, 0.25, 0.25, 0.25]

        # Randomness settings
        self.hidden_dropout_prob    = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.out_fpn_do_dropout     = False
        self.eval_robustness        = False
        self.pos_code_type  = False
        self.ablate_multihead       = False

    # return True if any parameter is successfully set, and False if none is set.
    def try_assign(self, args, *keys):
        is_successful = False
        
        for key in keys:
            if key in args:
                if isinstance(args, dict):
                    self.__dict__[key] = args[key]
                else:
                    self.__dict__[key] = args.__dict__[key]
                is_successful = True
                
        return is_successful

    def set_fpn_layers(self, config_name, fpn_settings, do_print=True):
        in_fpn_layers, out_fpn_layers, in_fpn_scheme, \
        out_fpn_scheme, translayer_compress_ratios = \
            fpn_settings.in_fpn_layers, fpn_settings.out_fpn_layers, fpn_settings.in_fpn_scheme, \
            fpn_settings.out_fpn_scheme, fpn_settings.translayer_compress_ratios
                       
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

#====================================== Segtran Shared Modules ========================================#

class MMPrivateMid(nn.Module):
    def __init__(self, config):
        super(MMPrivateMid, self).__init__()
        # Use 1x1 convolution as a group linear layer.
        # Equivalent to each group going through a respective nn.Linear().
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.group_linear   = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.mid_act_fn     = config.act_fun
        # This dropout is not presented in huggingface transformers.
        # Added to conform with lucidrains and rwightman's implementations.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        x_trans = self.group_linear(x)      # [B0, 1792*4, U] -> [B0, 1792*4, U]
        x_act   = self.mid_act_fn(x_trans)  # [B0, 1792*4, U]
        x_drop  = self.dropout(x_act)
        return x_drop

class MMSharedMid(nn.Module):
    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.shared_linear  = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn     = config.act_fun
        # This dropout is not presented in huggingface transformers.
        # Added to conform with lucidrains and rwightman's implementations.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x: [B0, 1792*4, U] or [B0, 4, U, 1792]
    def forward(self, x):
        if len(x.shape) == 3:
            # shape_4d: [B0, 4, 1792, U].
            shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
            # x_4d: [B0, 4, U, 1792].
            x_4d        = x.view(shape_4d).permute([0, 1, 3, 2])
            reshaped    = True
        else:
            x_4d        = x
            reshaped    = False

        x_trans         = self.shared_linear(x_4d)
        x_act           = self.mid_act_fn(x_trans)
        x_drop          = self.dropout(x_act)

        if reshaped:
            # restore the original shape
            x_drop      = x_drop.permute([0, 1, 3, 2]).reshape(x.shape)

        return x_drop

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans <- SegtranFusionEncoder.
# MM***Output has a shortcut (residual) connection.
class MMPrivateOutput(nn.Module):
    def __init__(self, config):
        super(MMPrivateOutput, self).__init__()
        self.num_modes  = config.num_modes
        self.feat_dim   = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U]
    def forward(self, x, shortcut):
        x        = self.group_linear(x)
        # x_comb: [B0, 1792*4, U]. Residual connection.
        x_comb   = x + shortcut
        shape_4d = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        # x_comb_4d, x_drop_4d: [B0, 4, U, 1792].
        x_comb_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        x_drop_4d = self.dropout(x_comb_4d)
        x_normed = self.resout_norm_layer(x_drop_4d)
        return x_normed

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans <- SegtranFusionEncoder.
# MM***Output has a shortcut (residual) connection.
class MMSharedOutput(nn.Module):
    # feat_dim_allmode is not used. Just to keep the ctor arguments the same as MMPrivateOutput.
    def __init__(self, config):
        super(MMSharedOutput, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim  = config.feat_dim
        self.shared_linear = nn.Linear(self.feat_dim, self.feat_dim)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U] or [B0, 4, U, 1792]
    def forward(self, x, shortcut):
        # shape_4d: [B0, 4, 1792, U].
        shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        if len(x.shape) == 3:
            x_4d    = x.view(shape_4d).permute([0, 1, 3, 2])
        else:
            x_4d    = x
        if len(shortcut.shape) == 3:
            shortcut_4d = shortcut.view(shape_4d).permute([0, 1, 3, 2])
        else:
            shortcut_4d = shortcut

        # x_4d, shortcut_4d: [B0, 4, U, 1792].
        x_trans     = self.shared_linear(x_4d)
        # x_4d, x_comb: [B0, 4, U, 1792]. Residual connection.
        x_comb      = x_trans + shortcut_4d
        x_drop      = self.dropout(x_comb)
        x_normed    = self.resout_norm_layer(x_drop)
        return x_normed

# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        self.feat2score = nn.Linear(num_feat, 1)
        self.keepdim    = keepdim
        # self.aggr_norm_layer = nn.LayerNorm(num_feat, eps=1e-12, elementwise_affine=False)

    def forward(self, x, score_basis=None):
        # Assume the last dim of x is the feature dim.
        if score_basis is None:
            score_basis = x
        mode_scores     = self.feat2score(score_basis)
        mode_attn_probs = mode_scores.softmax(dim=self.group_dim)
        x_aggr          = (x * mode_attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        # x_aggr_normed = self.aggr_norm_layer(x_aggr)
        return x_aggr # x_aggr_normed

# ExpandedFeatTrans <- CrossAttFeatTrans.
# ExpandedFeatTrans has a residual connection.
class ExpandedFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(ExpandedFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_allmode = self.feat_dim * self.num_modes
        self.has_FFN        = config.has_FFN and not config.eval_robustness
        # has_input_skip should only used when not has_FFN.
        self.has_input_skip = getattr(config, 'has_input_skip', False)
        if self.has_input_skip:
            self.input_skip_coeff = nn.Parameter(torch.ones(1))

        if not config.use_mince_transformer or config.mince_scales is None:
            self.num_scales     = 0
            self.mince_scales   = None
            self.mince_channels = None
        else:
            # mince_scales: [1, 2, 3, 4...]
            self.mince_scales   = config.mince_scales
            self.num_scales     = len(self.mince_scales)
            # mince_channels could be fractions, or unnormalized integers,
            # e.g. [1, 1, 1, 1], which is equivalent to [0.25, 0.25, 0.25, 0.25].
            mince_channels_frac = np.array(config.mince_channel_props, dtype=float)
            mince_channels_frac /= mince_channels_frac.sum()
            # mince_channels_norm is a list of fractions of the feat_dim.
            # Convert them into a list of integers (starting and ending indices).
            self.mince_channels = [ 0 for _ in range(self.num_scales + 1) ]
            mince_channel_nums  = [ 0 for _ in range(self.num_scales) ]
            for i in range(self.num_scales - 1):
                mince_channel_num        = int(mince_channels_frac[i] * self.feat_dim)
                mince_channel_nums[i]    = mince_channel_num
                self.mince_channels[i+1] = mince_channel_num + self.mince_channels[i]
            # All the remaining channels belong to the last scale.
            # In case the sum of mince_channels = feat_dim - 1 due to loss of precision.
            self.mince_channels[-1] = self.feat_dim
            mince_channel_nums[-1]  = self.mince_channels[-1] - self.mince_channels[-2]

        # first_linear is the value/V projection in other transformer implementations.
        # The output of first_linear will be divided into num_modes groups.
        # first_linear is always 'private' for each group, i.e.,
        # parameters are not shared (parameter sharing makes no sense).
        self.first_linear   = nn.Linear(self.in_feat_dim, self.feat_dim_allmode, bias=config.v_has_bias)            
        self.first_norm_layer       = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.base_initializer_range = config.base_initializer_range
        
        print("{}: v_has_bias: {}, has_FFN: {}, has_input_skip: {}".format(
              self.name, config.v_has_bias, self.has_FFN, self.has_input_skip))
        if self.num_scales > 0:
            print("mince_scales: {}, mince_channels: {}".format(
                  self.num_scales, mince_channel_nums))

        self.pool_modes_keepdim = False
        self.pool_modes_feat = config.pool_modes_feat

        if self.pool_modes_feat == 'softmax':
            agg_basis_feat_dim = self.feat_dim
            # group_dim = 1, i.e., features will be aggregated across the modes.
            self.feat_softaggr = LearnedSoftAggregate(agg_basis_feat_dim, group_dim=1,
                                                      keepdim=self.pool_modes_keepdim)

        self.mid_type = config.mid_type
        if self.mid_type == 'shared':
            self.intermediate = MMSharedMid(self.config)
        elif self.mid_type == 'private':
            self.intermediate = MMPrivateMid(self.config)
        else:
            self.intermediate = config.act_fun

        if config.trans_output_type == 'shared':
            self.output = MMSharedOutput(config)
        elif config.trans_output_type == 'private':
            self.output = MMPrivateOutput(config)

    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            # first_linear dimension is num_modes * feat_dim.
            # If in_feat_dim == feat_dim, only add identity bias to the first mode.
            # If in_feat_dim > feat_dim, expand to more modes until all in_feat_dim dimensions are covered.
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.base_initializer_range \
                              * self.config.feattrans_lin1_idbias_scale
            # Only bias the weight of the first mode.
            # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
            self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] = \
                self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] * 0.5 + identity_weight

    def forward(self, input_feat, attention_probs):
        # input_feat: [B, U2, 1792], mm_first_feat: [B, Units, 1792*4]
        # B: batch size, U2: number of the 2nd group of units, 
        # IF: in_feat_dim, could be different from feat_dim, due to layer compression 
        # (different from squeezed attention).
        B, U2, IF = input_feat.shape
        U1 = attention_probs.shape[2]
        F = self.feat_dim
        M = self.num_modes
        # mm_first_feat: commonly known as the value features (expanded into M=4 modes)
        # No matter whether there are multiple mince_scales or one, the value projection is the same.
        mm_first_feat = self.first_linear(input_feat)
        # mm_first_feat after transpose: [B, 1792*4, U2]
        mm_first_feat = mm_first_feat.transpose(1, 2)
        
        # mm_first_feat_4d: [B, 4, U2, 1792]
        mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)

        if self.num_scales > 0:
            scales_first_feat_fusion = []
            for s in range(self.num_scales):
                scale = self.mince_scales[s]
                # L, R: left and right indices of the mince_channels.
                L, R = self.mince_channels[s], self.mince_channels[s+1]
                # mince_first_feat_4d: [B, 4, U2, R-L]
                mince_first_feat_4d = mm_first_feat_4d[:, :, :, L:R]
                mince_first_feat_4d = resize_flat_features(mince_first_feat_4d, 1./scale)
                # attention_probs[s]:  [B, 4, U1/scale^d, U2/scale^d], 
                # d is the geometric dimension (2 or 3).
                # mince_first_feat_4d:     [B, 4, U2/scale^2, R-L]
                # mince_first_feat_fusion: [B, 4, U1/scale^2, R-L]
                mince_first_feat_fusion = torch.matmul(attention_probs[s], mince_first_feat_4d)
                # mince_first_feat_fusion: [B, 4, U1, R-L]
                mince_first_feat_fusion = resize_flat_features(mince_first_feat_fusion, scale)
                scales_first_feat_fusion.append(mince_first_feat_fusion)
            
            # mm_first_feat_fusion: [B, 4, U1, 1792]
            mm_first_feat_fusion = torch.cat(scales_first_feat_fusion, dim=3)
        else:
            # attention_probs: [B, Modes, U1, U2], mm_first_feat_4d: [B, 4, U2, 1792]
            # mm_first_feat_fusion: [B, 4, U1, 1792]
            mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)

        mm_first_feat_fusion_3d = mm_first_feat_fusion.transpose(2, 3).reshape(B, M*F, U1)
        mm_first_feat = mm_first_feat_fusion_3d
        # Skip the transformation layers on fused value to simplify the analyzed pipeline.
        if not self.has_FFN:
            trans_feat = self.feat_softaggr(mm_first_feat_fusion)
            if self.has_input_skip:
                trans_feat = trans_feat + self.input_skip_coeff * input_feat
            trans_feat = self.first_norm_layer(trans_feat)
            return trans_feat

        # mm_mid_feat:   [B, 1792*4, U1]. Group linear & gelu of mm_first_feat.
        mm_mid_feat  = self.intermediate(mm_first_feat)
        # mm_last_feat:  [B, 4, U1, 1792]. Group/shared linear & residual & Layernorm
        mm_last_feat = self.output(mm_mid_feat, mm_first_feat)

        mm_trans_feat = mm_last_feat

        if self.pool_modes_feat == 'softmax':
            trans_feat = self.feat_softaggr(mm_trans_feat)
        elif self.pool_modes_feat == 'max':
            trans_feat = mm_trans_feat.max(dim=1)[0]
        elif self.pool_modes_feat == 'mean':
            trans_feat = mm_trans_feat.mean(dim=1)
        elif self.pool_modes_feat == 'none':
            trans_feat = mm_trans_feat

        # trans_feat: [B, U1, 1792]
        return trans_feat

class CrossAttFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(CrossAttFeatTrans, self).__init__()
        self.config         = config
        self.name           = name
        self.num_modes      = config.num_modes
        self.in_feat_dim    = config.in_feat_dim
        self.feat_dim       = config.feat_dim
        self.attention_mode_dim = self.in_feat_dim // self.num_modes   # 448
        # att_size_allmode: 512 * modes
        self.att_size_allmode = self.num_modes * self.attention_mode_dim
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.key   = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.base_initializer_range = config.base_initializer_range

        # if using SlidingPosBiases, then add positional embeddings here.
        if config.pos_code_type == 'bias':
            self.pos_code_weight = config.pos_code_weight
        else:
            self.pos_code_weight = 1
            
        if config.ablate_multihead:
            self.out_trans  = MultiHeadFeatTrans(config, name)
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name)

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.keep_attn_scores = False
        self.tie_qk_scheme = config.tie_qk_scheme
        print("{} in_feat_dim: {}, feat_dim: {}, qk_have_bias: {}".format(
              self.name, self.in_feat_dim, self.feat_dim, config.qk_have_bias))

        self.attn_clip          = config.attn_clip
        if 'attn_diag_cycles' in config.__dict__:
            self.attn_diag_cycles   = config.attn_diag_cycles
        else:
            self.attn_diag_cycles   = 500
            
        self.max_attn    = 0
        self.clamp_count = 0
        self.call_count  = 0

    # if tie_qk_scheme is not None, it overrides the initialized self.tie_qk_scheme
    def tie_qk(self, tie_qk_scheme=None):
        # override config.tie_qk_scheme
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

        print("Initialize QK scheme: {}".format(self.tie_qk_scheme))
        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias

        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def add_identity_bias(self):
        identity_weight = torch.diag(torch.ones(self.attention_mode_dim)) * self.base_initializer_range \
                          * self.config.query_idbias_scale
        repeat_count = self.in_feat_dim // self.attention_mode_dim
        identity_weight = identity_weight.repeat([1, repeat_count])
        # only bias the weight of the first mode
        # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
        self.key.weight.data[:self.attention_mode_dim] = \
            self.key.weight.data[:self.attention_mode_dim] * 0.5 + identity_weight

    def transpose_for_scores(self, x):
        x_new_shape = x.size()[:-1] + (self.num_modes, -1)
        x = x.view(*x_new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, in_query, in_key=None, pos_biases=None):
        # in_query: [B, U1, 1792]
        # if in_key == None: self attention.
        if in_key is None:
            in_key = in_query
        # mixed_query_feat, mixed_key_feat: [B, U1, 1792], [B, U2, 1792]
        mixed_query_feat = self.query(in_query)
        mixed_key_feat   = self.key(in_key)
        # query_feat, key_feat: [B, 4, U1, 448], [B, 4, U2, 448]
        query_feat = self.transpose_for_scores(mixed_query_feat)
        key_feat   = self.transpose_for_scores(mixed_key_feat)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_feat, key_feat.transpose(-1, -2)) # [B0, 4, U1, 448] [B0, 4, 448, U2]
        attention_scores = attention_scores / math.sqrt(self.attention_mode_dim)  # [B0, 4, U1, U2]

        with torch.no_grad():
            curr_max_attn = attention_scores.max().item()
            pos_count     = (attention_scores > 0).sum()
            curr_avg_attn = attention_scores.sum() / pos_count
            curr_avg_attn = curr_avg_attn.item()

        if curr_max_attn > self.max_attn:
            self.max_attn = curr_max_attn

        if curr_max_attn > self.attn_clip:
            attention_scores = torch.clamp(attention_scores, -self.attn_clip, self.attn_clip)
            self.clamp_count += 1

        if self.training:
            self.call_count += 1
            if self.call_count % self.attn_diag_cycles == 0:
                print("max-attn: {:.2f}, avg-attn: {:.2f}, clamp-count: {}".format(self.max_attn, curr_avg_attn, self.clamp_count))
                self.max_attn    = 0
                self.clamp_count = 0        

        # Apply the positional biases
        if pos_biases is not None:
            #[B0, 4, U1, U2] = [B0, 4, U1, U2]  + [U1, U2].
            attention_scores = attention_scores + self.pos_code_weight * pos_biases

        if self.keep_attn_scores:
            self.attention_scores = attention_scores
        else:
            self.attention_scores = None

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.att_dropout(attention_probs)     #[B0, 4, U1, U2]

        # out_feat: [B0, U1, 1792], in the same size as in_query.
        out_feat      = self.out_trans(in_key, attention_probs)

        return out_feat

class CrossMinceAttFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(CrossMinceAttFeatTrans, self).__init__()
        self.config         = config
        self.name           = name
        self.num_modes      = config.num_modes
        self.in_feat_dim    = config.in_feat_dim
        self.feat_dim       = config.feat_dim
        # mince_scales: [1, 2, 3, 4...]
        if config.use_mince_transformer:
            self.mince_scales     = config.mince_scales
            self.num_scales       = len(self.mince_scales)
        else:
            breakpoint()        # shouldn't reach here.

        self.attention_mode_dim = self.in_feat_dim // self.num_modes   # 448
        # att_size_allmode: each combination of (mode, scale) has 448 dims of features.
        self.att_size_allmode = self.num_modes * self.num_scales * self.attention_mode_dim

        # No need to allocate different queries/keys for different scales. 
        # Instead, split channels after the query/key projection.
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.key   = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.base_initializer_range = config.base_initializer_range

        # if using SlidingPosBiases, then add positional embeddings here.
        if config.pos_code_type == 'bias':
            self.pos_code_weight = config.pos_code_weight
        else:
            self.pos_code_weight = 1
            
        if config.ablate_multihead:
            breakpoint()    # Not implemented
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name)

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.keep_attn_scores = False
        self.tie_qk_scheme = config.tie_qk_scheme
        print("{} in_feat_dim: {}, feat_dim: {}, qk_have_bias: {}".format(
              self.name, self.in_feat_dim, self.feat_dim, config.qk_have_bias))

        self.attn_clip          = config.attn_clip
        if 'attn_diag_cycles' in config.__dict__:
            self.attn_diag_cycles   = config.attn_diag_cycles
        else:
            self.attn_diag_cycles   = 500
            
        self.max_attn    = [ 0 for i in range(self.num_scales) ]
        self.clamp_count = [ 0 for i in range(self.num_scales) ]
        self.call_count  = 0

    # if tie_qk_scheme is not None, it overrides the initialized self.tie_qk_scheme
    def tie_qk(self, tie_qk_scheme=None):
        # override config.tie_qk_scheme
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

        print("Initialize QK scheme: {}".format(self.tie_qk_scheme))
        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias

        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def add_identity_bias(self):
        identity_weight = torch.diag(torch.ones(self.attention_mode_dim)) * self.base_initializer_range \
                          * self.config.query_idbias_scale
        repeat_count = self.in_feat_dim // self.attention_mode_dim
        identity_weight = identity_weight.repeat([1, repeat_count])
        # only bias the weight of the first mode
        # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
        self.key.weight.data[:self.attention_mode_dim] = \
            self.key.weight.data[:self.attention_mode_dim] * 0.5 + identity_weight

    # x: [B, U*S, 1792] => [B, 4, S, U, 448]
    def transpose_for_scores(self, x):
        x_new_shape = list(x.shape)
        x_new_shape[1] = x_new_shape[1] // self.num_scales
        # x_new_shape: [B, U, S, 4, -1]
        x_new_shape[2:] = (self.num_scales, self.num_modes, -1)
        x = x.reshape(*x_new_shape)
        return x.permute(0, 3, 2, 1, 4)

    # query_geoshape, key_geoshape: the geometric shapes of query and key input features.
    # pos_biases: a list of positional biases, one for each scale.
    def forward(self, in_query, query_geoshape, in_key=None, key_geoshape=None, pos_biases=None):
        # in_query: [B, U1, 1792]
        # if in_key == None: self attention.
        if in_key is None:
            in_key     = in_query
            key_geoshape = query_geoshape
        # mixed_query_feat, mixed_key_feat: [B, U1, 1792], [B, U2, 1792]
        mixed_query_feat = self.query(in_query)
        mixed_key_feat   = self.key(in_key)
        # query_feat, key_feat: [B, 4, S, U1, 448], [B, 4, S, U2, 448]
        query_feat = self.transpose_for_scores(mixed_query_feat)
        key_feat   = self.transpose_for_scores(mixed_key_feat)
        attention_geoshape = (query_feat.shape[2], key_feat.shape[2])
        scales_attention_scores = []
        scales_attention_probs  = []
        self.call_count += 1

        for s in range(self.num_scales):
            scale = self.mince_scales[s]
            # scale_query_feat: features of this scale. [B, 4, U1, 448]
            scale_query_feat = query_feat[:, :, s]
            # scale_query_feat: [B, 4, U1/(scale^d), 448]
            scale_query_feat = resize_flat_features(scale_query_feat, query_geoshape, scale)
            # scale_key_feat:   [B, 4, U2, 448]
            scale_key_feat   = key_feat[  :, :, s]
            # scale_key_feat:   [B, 4, U2/(scale^d), 448]
            scale_key_feat   = resize_flat_features(scale_key_feat,   key_geoshape,   scale)
            
            # scale_attention_scores: attention score of this scale. [B, 4, U1/(scale^d), U2/(scale^d)]
            # Take the dot product between "query" and "key" to get the raw attention scores.
            scale_attention_scores = torch.matmul(scale_query_feat, scale_key_feat.transpose(-1, -2))
            scale_attention_scores = scale_attention_scores / math.sqrt(self.attention_mode_dim)

            with torch.no_grad():
                curr_max_attn = scale_attention_scores.max().item()
                pos_count     = (scale_attention_scores > 0).sum()
                curr_avg_attn = scale_attention_scores.sum() / pos_count
                curr_avg_attn = curr_avg_attn.item()

            if curr_max_attn > self.max_attn[s]:
                self.max_attn[s] = curr_max_attn

            if curr_max_attn > self.attn_clip:
                scale_attention_scores = torch.clamp(scale_attention_scores, -self.attn_clip, self.attn_clip)
                self.clamp_count[s] += 1

            if self.training and self.call_count % self.attn_diag_cycles == 0:
                print("{} attn max: {:.2f}, avg: {:.2f}, clamp-count: {}".format(
                        scale, self.max_attn[s], curr_avg_attn, self.clamp_count[s]))
                self.max_attn[s]    = 0
                self.clamp_count[s] = 0        

            # Apply the positional biases. 
            # pos_biases could be None for some of the scales, especially for smaller scales 
            # (the feature maps may be too small to be covered by positional biases).
            if pos_biases is not None and pos_biases[s] is not None:
                # [B, 4, U1/(scale^d), U2/(scale^d)] = [B, 4, U1/(scale^d), U2/(scale^d)] + 
                #                                      [U1/(scale^d), U2/(scale^d)]
                scale_attention_scores = scale_attention_scores + self.pos_code_weight * pos_biases[s]

            # scale_attention_scores: [B, 4, U1/(scale^d), U2/(scale^d)]
            scales_attention_scores.append(scale_attention_scores)

            # Normalize the attention scores to probabilities.
            scale_attention_probs = nn.functional.softmax(scale_attention_scores, dim=-1)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            scale_attention_probs = self.att_dropout(scale_attention_probs)     #[B0, 4, U1, U2]
            scales_attention_probs.append(scale_attention_probs)

        if self.keep_attn_scores:
            self.scales_attention_scores = scales_attention_scores
        else:
            self.scales_attention_scores = None

        # out_feat: [B0, U1, 1792], in the same size as in_query.
        out_feat      = self.out_trans(in_key, scales_attention_probs)

        return out_feat

class SqueezedAttFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(SqueezedAttFeatTrans, self).__init__()
        self.config    = config
        self.name      = name
        self.in_feat_dim    = config.in_feat_dim
        self.num_attractors = config.num_attractors
        
        # Disable channel compression and multi-mode expansion in in_ator_trans. 
        config1 = copy.copy(config)
        config1.feat_dim = config1.in_feat_dim        
        config1.num_modes = 1
        config1.has_FFN     = config.has_FFN_in_squeeze
        if config.use_mince_transformer:
            self.in_ator_trans  = CrossMinceAttFeatTrans(config1, name + '-in-squeeze')
            self.ator_out_trans = CrossMinceAttFeatTrans(config, name + '-squeeze-out')
        else:
            self.in_ator_trans  = CrossAttFeatTrans(config1, name + '-in-squeeze')
            self.ator_out_trans = CrossAttFeatTrans(config, name + '-squeeze-out')
        self.attractors     = Parameter(torch.randn(1, self.num_attractors, self.in_feat_dim))
    
    # pos_biases cannot be used in Squeezed Attention.
    def forward(self, in_feat, pos_biases=None):
        # in_feat: [B, 196, 1792]
        batch_size = in_feat.shape[0]
        batch_attractors = self.attractors.expand(batch_size, -1, -1)
        new_batch_attractors = self.in_ator_trans(batch_attractors, in_feat, pos_biases)
        out_feat             = self.ator_out_trans(in_feat, new_batch_attractors, pos_biases)
        self.attention_scores = self.ator_out_trans.attention_scores
        return out_feat

# A multi-layer self-attentive transformer.        
class SegtranFusionEncoder(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        self.name = name
        self.num_translayers            = config.num_translayers
        self.pos_code_type              = config.pos_code_type
        self.pos_code_every_layer       = config.pos_code_every_layer
        self.translayer_compress_ratios = config.translayer_compress_ratios
        self.translayer_dims            = config.translayer_dims
        self.dropout                    = nn.Dropout(config.hidden_dropout_prob)
        self.use_squeezed_transformer   = config.use_squeezed_transformer
        self.use_mince_transformer      = config.use_mince_transformer

        print(f'Segtran {self.name} Encoder with {self.num_translayers} trans-layers')

        self.pos_code_type          = config.pos_code_type
        # positional biases are only for ordinary self-attentive transformers.
        if self.use_squeezed_transformer and self.pos_code_type == 'bias':
            print("Squeezed transformer cannot use positional biases. Please specify '--nosqueeze'")
            exit(0)

        # if using SlidingPosBiases, do not add positional embeddings here.
        if config.pos_code_type != 'bias':
            self.pos_code_weight    = config.pos_code_weight
        else:
            self.pos_code_weight    = 0

        if self.use_mince_transformer:
            self.num_scales = len(config.mince_scales)
            self.mince_scales = config.mince_scales
            if self.pos_code_type != 'bias':
                breakpoint()        # Not supported yet.
            else:
                # A pos_coder_layer for each scale.
                self.pos_code_layers = [ SegtranPosEncoder(config) for _ in range(self.num_scales) ]
        else:
            self.num_scales = 0
            self.pos_code_layer = SegtranPosEncoder(config)

        # each vfeat_norm_layer has affine, to adjust the proportions of vfeat vs. pos embeddings
        vfeat_norm_layers = []
        translayers = []

        # if both use_squeezed_transformer and use_mince_transformer,
        # then SqueezedAttFeatTrans uses CrossMinceAttFeatTrans instead of CrossAttFeatTrans 
        # as the building blocks.
        if self.use_squeezed_transformer:
            TransformerClass = SqueezedAttFeatTrans
        elif self.use_mince_transformer:
            TransformerClass = CrossMinceAttFeatTrans
        else:
            TransformerClass = CrossAttFeatTrans
            
        for i in range(self.num_translayers):
            config2 = copy.copy(config)
            config2.in_feat_dim = self.translayer_dims[i]
            config2.feat_dim    = self.translayer_dims[i+1]
            translayers.append(TransformerClass(config2, '%s%d' %(name, i)))
        self.translayers = nn.ModuleList(translayers)

        # comb_norm_layers have no affine, to maintain the proportion of visual features.
        # Do not need the last translayer_dims, which is the output dim of the last layer.
        comb_norm_layers        = [ nn.LayerNorm(translayer_dim, eps=1e-12, elementwise_affine=False) \
                                    for translayer_dim in self.translayer_dims[:-1] ]
        # vfeat_norm_layers have affine, to have more flexibility.
        vfeat_norm_layers       = [ nn.LayerNorm(translayer_dim, eps=1e-12, elementwise_affine=True) \
                                    for translayer_dim in self.translayer_dims[:-1] ]
        self.comb_norm_layers   = nn.ModuleList(comb_norm_layers)
        self.vfeat_norm_layers  = nn.ModuleList(vfeat_norm_layers)

    # if pos_code_every_layer=True (default), then vfeat is vis_feat.
    # Otherwise, vfeat is combined feat.
    def forward(self, vfeat, voxels_pos, vmask, orig_feat_shape):
        self.layers_vfeat = []
        if self.pos_code_every_layer:
            MAX_POS_LAYER = self.num_translayers        # default behavior.
        else:
            MAX_POS_LAYER = 1

        for i, translayer in enumerate(self.translayers):
            if i < MAX_POS_LAYER:
                # Add pos embeddings to transformer input features in every layer,
                # to avoid loss of positional signals after transformations.
                vfeat_normed    = self.vfeat_norm_layers[i](vfeat)
                if self.num_scales > 0:
                    # orig_feat_shape only includes geometric shape (no batch, channel).
                    # multi_resize_shape() returns a list of shapes by resizing orig_feat_shape with different scales.
                    scale_feat_shapes = multi_resize_shape(orig_feat_shape, self.mince_scales)
                    # pos_code is actually a list of pos_code for different scales.
                    pos_code    = [ self.pos_code_layers[s](scale_feat_shapes[s], voxels_pos) for s in range(self.num_scales) ]
                else:
                    pos_code    = self.pos_code_layer(orig_feat_shape, voxels_pos)
 
                if self.pos_code_type != 'bias':
                    feat_comb       = vfeat_normed + \
                                      self.pos_code_weight * \
                                      pos_code[:, :, :self.translayer_dims[i]]
                                        
                    feat_normed     = self.comb_norm_layers[i](feat_comb)
                    # pos_code is already added to features. No need to pass pos_code to translayer.
                    pos_code        = None
                # For mince transformer, pos_code_type == 'bias'.
                else:
                    # pos_code will be added to attention scores in translayer().
                    feat_normed     = vfeat_normed
                    
                # Only do dropout in the first layer.
                # In later layers, dropout is already applied at the output end. Avoid doing it again.
                if i == 0:
                    feat_normed = self.dropout(feat_normed)
                feat_masked     = feat_normed * vmask
                # if pos_code_type != 'bias', positional embedding has been added to feat_comb, and pos_code is None. 
                # otherwise, pos_code is added to the attention score matrix computed in translayer().
                # If pos_code_type == 'bias', all layers share the same pos_biases.
                # Otherwise, at above, different subtensors (when different layers have different dimensions) 
                # of the same pos_code are added to different layers.
                vfeat = translayer(feat_masked, pos_biases=pos_code)
            else:
                feat_masked = vfeat * vmask
                vfeat = translayer(feat_masked, pos_biases=None)
            self.layers_vfeat.append(vfeat)

        return vfeat

# =================================== Segtran BackBone Components ==============================#

class LearnedSinuPosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, omega=1, affine=False):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_fc = nn.Linear(self.pos_dim, self.pos_embed_dim, bias=True)
        self.pos_mix_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        self.omega = omega
        print("Learnable Sinusoidal positional encoding")
        
    def forward(self, pos_normed):
        pos_embed_sum = 0
        pos_embed0 = self.pos_fc(pos_normed)
        pos_embed_sin = torch.sin(self.omega * pos_embed0[:, :, 0::2])
        pos_embed_cos = torch.cos(self.omega * pos_embed0[:, :, 1::2])
        # Interlace pos_embed_sin and pos_embed_cos.
        pos_embed_mix = torch.stack((pos_embed_sin, pos_embed_cos), dim=3).view(pos_embed0.shape)
        pos_embed_out = self.pos_mix_norm_layer(pos_embed_mix)

        return pos_embed_out

# For feature maps with a 2D spatial shape (i.e., 2D images).
# max_pos_size: maximum size of the feature maps that will be covered with positional biases.
class SlidingPosBiases2D(nn.Module):
    def __init__(self, pos_dim, pos_bias_radius=7, max_pos_size=(100, 100)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [15, 15]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim) ]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_size[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_size[1]):
                    # h1s, w1s, h2s, w2s are indices to the padded feature maps. 
                    # Therefore h2s, w2s are within [i, i+2*R] and [j, j+2*R].
                    # After removing padding, the actual indices are within [i-R, i+R] and [j-R, j+R].
                    # h1s, w1s, h2s, w2s: [1, 1, 2*R+1, 2*R+1]
                    h1s, w1s, h2s, w2s = torch.meshgrid(torch.tensor(i), torch.tensor(j), 
                                                        torch.arange(i, i+2*R+1), torch.arange(j, j+2*R+1))
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)

                # i_*: [1, W, 2*R+1, 2*R+1]                         
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            
            # all_*: [H, W, 2*R+1, 2*R+1]
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
        else:
            breakpoint()

        # Put indices on GPU to speed up.
        self.register_buffer('all_h1s', all_h1s)
        self.register_buffer('all_w1s', all_w1s)
        self.register_buffer('all_h2s', all_h2s)
        self.register_buffer('all_w2s', all_w2s)
        print(f"Sliding-window Positional Biases, r: {R}, max size: {max_pos_size}")
        
    def forward(self, feat_shape, device):
        R = self.R
        # spatial_shape: [H, W]
        spatial_shape = feat_shape[-self.pos_dim:]
        # [H, W, H, W] => [H, W, H+2R, W+2R].
        padded_pos_shape  = list(spatial_shape) + [ 2*R + spatial_shape[i] for i in range(self.pos_dim) ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[(all_h1s, all_w1s, all_h2s, all_w2s)] = self.biases
        else:
            breakpoint()

        # Remove padding. [H, W, H+2R, W+2R] => [H, W, H, W].
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
        pos_biases = pos_biases.reshape(feat_shape.numel(), feat_shape.numel())
        return pos_biases

# For feature maps with a 3D spatial shape (i.e., 3D images).
# max_pos_size: maximum size of the feature maps that will be covered with positional biases.
class SlidingPosBiases3D(nn.Module):
    def __init__(self, pos_dim, pos_bias_radius=7, max_pos_size=(20, 20, 20)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [15, 15, 15]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim) ]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        if self.pos_dim == 3:
            all_h1s, all_w1s, all_d1s, all_h2s, all_w2s, all_d2s = [], [], [], [], [], []
            for i in range(max_pos_size[0]):
                # i_*: indices *except* the i (0th) dimension.
                i_h1s, i_w1s, i_d1s, i_h2s, i_w2s, i_d2s = [], [], [], [], [], []
                for j in range(max_pos_size[1]):
                    # ij_*: indices *except* the i (0th) and j (1st) dimensions.
                    ij_h1s, ij_w1s, ij_d1s, ij_h2s, ij_w2s, ij_d2s = [], [], [], [], [], []
                    for k in range(max_pos_size[2]):
                        # h1s, w1s, h2s, w2s, h3s, w3s are indices to the padded feature maps. 
                        # Therefore h2s, w2s, h3s, w3s are within [i, i+2*R], [j, j+2*R], [k, k+2*R].
                        # After removing padding, the actual indices are within [i-R, i+R], [j-R, j+R], [k-R, k+R].
                        # h1s, w1s, d1s, h2s, w2s, d2s: [1, 1, 1, 2*R+1, 2*R+1, 2*R+1]
                        h1s, w1s, d1s, h2s, w2s, d2s = \
                                torch.meshgrid(torch.tensor(i), torch.tensor(j), torch.tensor(k), 
                                               torch.arange(i, i+2*R+1), torch.arange(j, j+2*R+1),
                                               torch.arange(k, k+2*R+1))
                        ij_h1s.append(h1s)
                        ij_w1s.append(w1s)
                        ij_d1s.append(d1s)
                        ij_h2s.append(h2s)
                        ij_w2s.append(w2s)
                        ij_d2s.append(d2s)
                    # ij_*: [1, 1, D, 2*R+1, 2*R+1, 2*R+1]
                    ij_h1s = torch.cat(ij_h1s, dim=2)
                    ij_w1s = torch.cat(ij_w1s, dim=2)
                    ij_d1s = torch.cat(ij_d1s, dim=2)
                    ij_h2s = torch.cat(ij_h2s, dim=2)
                    ij_w2s = torch.cat(ij_w2s, dim=2)
                    ij_d2s = torch.cat(ij_d2s, dim=2)
                    i_h1s.append(ij_h1s)
                    i_w1s.append(ij_w1s)
                    i_d1s.append(ij_d1s)
                    i_h2s.append(ij_h2s)
                    i_w2s.append(ij_w2s)
                    i_d2s.append(ij_d2s)
                # i_*: [1, W, D, 2*R+1, 2*R+1, 2*R+1]
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_d1s = torch.cat(i_d1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                i_d2s = torch.cat(i_d2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_d1s.append(i_d1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
                all_d2s.append(i_d2s)
            # all_*: [H, W, D, 2*R+1, 2*R+1, 2*R+1]
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_d1s = torch.cat(all_d1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
            all_d2s = torch.cat(all_d2s, dim=0)
        else:
            breakpoint()

        # Put indices on GPU to speed up.
        self.register_buffer('all_h1s', all_h1s)
        self.register_buffer('all_w1s', all_w1s)
        self.register_buffer('all_d1s', all_d1s)
        self.register_buffer('all_h2s', all_h2s)
        self.register_buffer('all_w2s', all_w2s)
        self.register_buffer('all_d2s', all_d2s)
        print(f"Sliding-window Positional Biases, r: {R}, max size: {max_pos_size}")
        
    def forward(self, feat_shape, device):
        R = self.R
        # spatial_shape: [H, W, D]
        spatial_shape = feat_shape[-self.pos_dim:]
        # [H, W, D, H, W, D] => [H, W, D, H+2R, W+2R, D+2R].
        padded_pos_shape  = list(spatial_shape) + [ 2*R + spatial_shape[i] for i in range(self.pos_dim) ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        
        if self.pos_dim == 3:
            H, W, D = spatial_shape
            all_h1s = self.all_h1s[:H, :W, :D]
            all_w1s = self.all_w1s[:H, :W, :D]
            all_d1s = self.all_d1s[:H, :W, :D]
            all_h2s = self.all_h2s[:H, :W, :D]
            all_w2s = self.all_w2s[:H, :W, :D]
            all_d2s = self.all_d2s[:H, :W, :D]
            padded_pos_biases[(all_h1s, all_w1s, all_d1s, all_h2s, all_w2s, all_d2s)] = self.biases
        else:
            breakpoint()

        # Remove padding. [H, W, H+2R, W+2R] => [H, W, H, W].
        pos_biases = padded_pos_biases[:, :, :, R:-R, R:-R, R:-R]
        pos_biases = pos_biases.reshape(feat_shape.numel(), feat_shape.numel())
        return pos_biases

class SegtranPosEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.trans_in_dim  # 1792
        self.pos_embed_dim = self.feat_dim
        self.pos_code_type = config.pos_code_type
        
        # self.feat_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)

        # Box position encoding. no affine, but could have bias.
        # 2 channels => 1792 channels
        if self.pos_code_type == 'lsinu':
            self.pos_coder = LearnedSinuPosEmbedder(config.pos_dim, self.pos_embed_dim, omega=1, affine=False)
        elif self.pos_code_type == 'rand':
            self.pos_coder = RandPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif self.pos_code_type == 'sinu':
            self.pos_coder = SinuPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif self.pos_code_type == 'zero':
            self.pos_coder = ZeroEmbedder(self.pos_embed_dim)
        elif self.pos_code_type == 'bias':
            if config.pos_dim == 2:
                self.pos_coder = SlidingPosBiases2D(config.pos_dim, config.pos_bias_radius, config.max_pos_size)
            elif config.pos_dim == 3:
                self.pos_coder = SlidingPosBiases3D(config.pos_dim, config.pos_bias_radius, config.max_pos_size)

        self.cached_pos_code   = None
        self.cached_feat_shape = None
        # comb_norm_layer has no affine, to maintain the proportion of visual features
        # self.comb_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.pos_code_every_layer = config.pos_code_every_layer

    # Cache the pos_code and feat_shape to avoid unnecessary generation time.    
    # This is only used during inference. During training, pos_code is always generated each time it's used.
    # Otherwise the cached pos_code cannot receive proper gradients.
    def pos_code_lookup_cache(self, vis_feat_shape, device, voxels_pos_normed):
        if self.pos_code_type == 'bias':
            # Cache miss for 'bias' type of positional codes.
            if self.training or self.cached_pos_code is None or self.cached_feat_shape != vis_feat_shape:
                # pos_coder is SlidingPosBiases2D or SlidingPosBiases3D.
                self.cached_pos_code    = self.pos_coder(vis_feat_shape, device)
                self.cached_feat_shape  = vis_feat_shape
            # else: self.cached_pos_code exists, and self.cached_feat_shape == vis_feat_shape.
            # Just return the cached pos_code.
        else:
            # Cache miss for all other type of positional codes.
            if self.training or self.cached_pos_code is None or self.cached_feat_shape != voxels_pos_normed.shape:
                # pos_coder is a positional embedder.
                self.cached_pos_code    = self.pos_coder(voxels_pos_normed)
                self.cached_feat_shape  = voxels_pos_normed.shape
            # else: self.cached_pos_code exists, and self.cached_feat_shape == voxels_pos_normed.shape.
            # Just return the cached pos_code.

        return self.cached_pos_code

    def forward(self, orig_feat_shape, voxels_pos):
        # orig_feat_shape:               [2, 1296, 1792]
        # voxels_pos, voxels_pos_normed: [2, 1296, 2]
        voxels_pos_normed = voxels_pos / voxels_pos.max()
        # if not bias, pos_code:         [2, 1296, 1792]
        pos_code = self.pos_code_lookup_cache(orig_feat_shape, voxels_pos.device, voxels_pos_normed)
        if self.pos_code_type == 'bias':
            ht, wd  = orig_feat_shape
            # pos_code:      [H*W, H*W] => [1, 1, H*W, H*W]
            pos_code = pos_code.reshape(1, 1, ht*wd, ht*wd)
        return pos_code

# =================================== Segtran Initialization ====================================#
class SegtranInitWeights(nn.Module):
    """ An abstract class to handle weights initialization """
    def __init__(self, config, *inputs, **kwargs):
        super(SegtranInitWeights, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
            type(module.weight)      # <class 'torch.nn.parameter.Parameter'>
            type(module.weight.data) # <class 'torch.Tensor'>
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if (np.array(module.weight.shape) < self.config.min_feat_dim).all():
                print("Skip init of Linear weight %s" %(list(module.weight.shape)))
            else:
                base_initializer_range  = self.config.base_initializer_range
                module.weight.data.normal_(mean=0.0, std=base_initializer_range)
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def tie_qk(self, module):
        if isinstance(module, CrossAttFeatTrans) and module.tie_qk_scheme != 'none':
            module.tie_qk()

    def add_identity_bias(self, module):
        if isinstance(module, CrossAttFeatTrans) or isinstance(module, ExpandedFeatTrans):
            module.add_identity_bias()
            
        '''
        if isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, a=1)
            module.bias.data.zero_()
            print("kaiming_uniform ConvTranspose2d %s" %list(module.weight.shape))
        '''
