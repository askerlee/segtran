import math
import random
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
        # If perturb_posw_range > 0, add random noise to pos_code_weight during training.
        # perturb_posw_range: the scale of the added random noise (relative to pos_code_weight)
        self.perturb_posw_range = 0.
        self.pos_in_attn_only = False
        self.pos_code_every_layer = True

        # Removing biases from QK seems to slightly degrade performance.
        self.qk_have_bias = True
        # Removing biases from V seems to slightly improve performance.
        self.v_has_bias   = False
        
        self.cross_attn_score_scale = 1.
        self.attn_clip = 500
        self.base_initializer_range = 0.02
        # Add an identity matrix (*0.02*query_idbias_scale) to query/key weights
        # to make a bias towards identity mapping.
        # Set to 0 to disable the identity bias.
        self.query_idbias_scale = 10
        self.feattrans_lin1_idbias_scale = 10

        # Pooling settings
        self.pool_modes_feat  = 'softmax'   # softmax, max, mean, or none. With [] means keepdim=True.

        # Randomness settings
        self.hidden_dropout_prob    = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.out_fpn_do_dropout     = False
        self.eval_robustness        = False
        self.pos_code_type  = False
        self.ablate_multihead       = False
                     
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
        # first_linear is the value projection in other transformer implementations.
        # The output of first_linear will be divided into num_modes groups.
        # first_linear is always 'private' for each group, i.e.,
        # parameters are not shared (parameter sharing makes no sense).
        
        self.first_linear   = nn.Linear(self.in_feat_dim, self.feat_dim_allmode, bias=config.v_has_bias)
        self.has_FFN        = config.has_FFN and not config.eval_robustness
        # has_input_skip should only used when not has_FFN.
        self.has_input_skip = getattr(config, 'has_input_skip', False)
        if self.has_input_skip:
            self.input_skip_coeff = nn.Parameter(torch.ones(1))
            
        self.first_norm_layer       = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.base_initializer_range = config.base_initializer_range
        
        print("{}: v_has_bias: {}, has_FFN: {}, has_input_skip: {}".format(
              self.name, config.v_has_bias, self.has_FFN, self.has_input_skip))

        if config.pool_modes_feat[0] == '[':
            self.pool_modes_keepdim = True
            self.pool_modes_feat = config.pool_modes_feat[1:-1]     # remove '[' and ']'
        else:
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
        mm_first_feat = self.first_linear(input_feat)
        # mm_first_feat after transpose: [B, 1792*4, U2]
        mm_first_feat = mm_first_feat.transpose(1, 2)
        
        # mm_first_feat_4d: [B, 4, U2, 1792]
        mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)

        # attention_probs: [B, Modes, U1, U2]
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
            # args.perturb_posw_range is the relative ratio. Get the absolute range here.
            self.perturb_posw_range = self.pos_code_weight * config.perturb_posw_range
            print("Positional biases weight perturbation: {:.3}/{:.3}".format(
                  self.perturb_posw_range, self.pos_code_weight))
        else:
            self.pos_code_weight = 1
            self.perturb_posw_range = 0
            
        if config.ablate_multihead:
            self.out_trans  = MultiHeadFeatTrans(config, name)
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name)

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.keep_attn_scores = True
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
        new_x_shape = x.size()[:-1] + (self.num_modes, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_feat, key_feat=None, pos_biases=None):
        # query_feat: [B, U1, 1792]
        # if key_feat == None: self attention.
        if key_feat is None:
            key_feat = query_feat
        # mixed_query_layer, mixed_key_layer: [B, U1, 1792], [B, U2, 1792]
        mixed_query_layer = self.query(query_feat)
        mixed_key_layer   = self.key(key_feat)
        # query_layer, key_layer: [B, 4, U1, 448], [B, 4, U2, 448]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B0, 4, U1, 448] [B0, 4, 448, U2]
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
            if self.perturb_posw_range > 0 and self.training:
                posw_noise = random.uniform(-self.perturb_posw_range, 
                                             self.perturb_posw_range)
            else:
                posw_noise = 0
            
            #[B0, 4, U1, U2] = [B0, 4, U1, U2]  + [U1, U2].
            attention_scores = attention_scores + (self.pos_code_weight + posw_noise) * pos_biases

        if self.keep_attn_scores:
            self.attention_scores = attention_scores
        else:
            self.attention_scores = None

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.att_dropout(attention_probs)     #[B0, 4, U1, U2]

        # out_feat: [B0, U1, 1792], in the same size as query_feat.
        out_feat      = self.out_trans(key_feat, attention_probs)

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
        self.in_ator_trans  = CrossAttFeatTrans(config1, name + '-in-squeeze')
        self.ator_out_trans = CrossAttFeatTrans(config, name + '-squeeze-out')
        self.attractors     = Parameter(torch.randn(1, self.num_attractors, self.in_feat_dim))
    
    # pos_biases cannot be used in Squeezed Attention.
    def forward(self, in_feat, pos_biases=None):
        # in_feat: [B, 196, 1792]
        batch_size = in_feat.shape[0]
        batch_attractors = self.attractors.expand(batch_size, -1, -1)
        new_batch_attractors = self.in_ator_trans(batch_attractors, in_feat)
        out_feat = self.ator_out_trans(in_feat, new_batch_attractors)
        self.attention_scores = self.ator_out_trans.attention_scores
        return out_feat
        
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
        print(f'Segtran {self.name} Encoder with {self.num_translayers} trans-layers')

        self.pos_code_type          = config.pos_code_type
        # if using SlidingPosBiases, do not add positional embeddings here.
        if config.pos_code_type != 'bias':
            self.pos_code_weight    = config.pos_code_weight
            # args.perturb_posw_range is the relative ratio. Get the absolute range here.
            self.perturb_posw_range = self.pos_code_weight * config.perturb_posw_range
            print("Positional embedding weight perturbation: {:.3}/{:.3}".format(
                  self.perturb_posw_range, self.pos_code_weight))
        else:
            self.pos_code_weight    = 0
            self.perturb_posw_range = 0
                    
        self.pos_code_layer = SegtranInputFeatEncoder(config)
        # each vfeat_norm_layer has affine, to adjust the proportions of vfeat vs. pos embeddings
        vfeat_norm_layers = []
        translayers = []

        if self.use_squeezed_transformer:
            TransformerClass = SqueezedAttFeatTrans
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

    # if pos_code_every_layer=True, then vfeat is vis_feat.
    # Otherwise, vfeat is combined feat.
    def forward(self, vfeat, voxels_pos, vmask, orig_feat_shape):
        self.layers_vfeat = []
        if self.pos_code_every_layer:
            MAX_POS_LAYER = self.num_translayers
        else:
            MAX_POS_LAYER = 1

        for i, translayer in enumerate(self.translayers):
            if i < MAX_POS_LAYER:
                # Add pos embeddings to transformer input features in every layer,
                # to avoid loss of positional signals after transformations.
                vfeat_normed    = self.vfeat_norm_layers[i](vfeat)
                pos_code        = self.pos_code_layer(orig_feat_shape, voxels_pos)
 
                if self.pos_code_type != 'bias':
                    if self.perturb_posw_range > 0 and self.training:
                        posw_noise = random.uniform(-self.perturb_posw_range, 
                                                     self.perturb_posw_range)
                    else:
                        posw_noise = 0

                    feat_comb       = vfeat_normed + \
                                      (self.pos_code_weight + posw_noise) * \
                                      pos_code[:, :, :self.translayer_dims[i]]
                                        
                    feat_normed     = self.comb_norm_layers[i](feat_comb)
                    pos_code        = None
                else:
                    # pos_code will be added to attention scores in translayer().
                    feat_normed     = vfeat_normed
                    
                # Only do dropout in the first layer.
                # In later layers, dropout is already applied at the output end. Avoid doing it again.
                if i == 0:
                    feat_normed = self.dropout(feat_normed)
                feat_masked     = feat_normed * vmask
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

class SlidingPosBiases(nn.Module):
    def __init__(self, pos_dim, pos_bias_radius=7, max_pos_size=(100, 100)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [15, 15]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim) ]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        # Currently only feature maps with a 2D spatial shape (i.e., 2D images) are supported.
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_size[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_size[1]):
                    h1s, w1s, h2s, w2s = torch.meshgrid(torch.tensor(i), torch.tensor(j), 
                                                        torch.arange(i, i+2*R+1), torch.arange(j, j+2*R+1))
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)
                                                  
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            
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
        spatial_shape = feat_shape[-self.pos_dim:]
        # [H, W, H, W] => [H+2R, W+2R, H+2R, W+2R].
        padded_pos_shape  = list(spatial_shape) + [ 2*R + spatial_shape[i] for i in range(self.pos_dim) ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=device)
        
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[(all_h1s, all_w1s, all_h2s, all_w2s)] = self.biases
                
        # Remove padding. [H+2R, W+2R, H+2R, W+2R] => [H, W, H, W].
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
        pos_biases = pos_biases.reshape(feat_shape.numel(), feat_shape.numel())
        return pos_biases
        
class SegtranInputFeatEncoder(nn.Module):
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
            self.pos_coder = SlidingPosBiases(config.pos_dim, config.pos_bias_radius, config.max_pos_size)

        # comb_norm_layer has no affine, to maintain the proportion of visual features
        # self.comb_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.pos_code_every_layer = config.pos_code_every_layer

    def forward(self, orig_feat_shape, voxels_pos):
        # vis_feat:  [2, 1296, 1792]
        # voxels_pos, voxels_pos_normed: [2, 1296, 2]
        voxels_pos_normed = voxels_pos / voxels_pos.max()
        # voxels_pos_normed: [B0, num_voxels, 2]
        if self.pos_code_type != 'bias':
            # pos_code:      [B0, num_voxels, 1792]
            pos_code = self.pos_coder(voxels_pos_normed)
        else:
            # pos_code:      [H*W, H*W]
            pos_code = self.pos_coder(orig_feat_shape, voxels_pos.device)
            # pos_code:      [1, 1, H*W, H*W]
            pos_code = pos_code.unsqueeze(0).unsqueeze(0)
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
