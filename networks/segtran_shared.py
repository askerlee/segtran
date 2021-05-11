import math
import numpy as np
import re
import copy

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import resnet
import resnet_ibn
from efficientnet.model import EfficientNet
from networks.segtran_ablation import RandPosEmbedder, SinuPosEmbedder, ZeroEmbedder, MultiHeadFeatTrans
torch.set_printoptions(sci_mode=False)


bb2feat_dims = { 'resnet34':  [64, 64,  128, 256,  512],
                 'resnet50':  [64, 256, 512, 1024, 2048],
                 'resnet101': [64, 256, 512, 1024, 2048],
                 'resibn101': [64, 256, 512, 1024, 2048],   # resibn: resnet + IBN layers
                 'eff-b0':    [16, 24,  40,  112,  1280],   # input: 224
                 'eff-b1':    [16, 24,  40,  112,  1280],   # input: 240
                 'eff-b2':    [16, 24,  48,  120,  1408],   # input: 260
                 'eff-b3':    [24, 32,  48,  136,  1536],   # input: 300
                 'eff-b4':    [24, 32,  56,  160,  1792],   # input: 380
                 'i3d':       [64, 192, 480, 832,  1024]    # input: 224
               }
               
max_attn = 0
avg_attn = 0
clamp_count = 0
call_count = 0

def swish(x):
    return x * torch.sigmoid(x)

def mish(x):
    return x *( torch.tanh(F.softplus(x)))

class Dropout(nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()
        self.kept_p = 1 - p
        self.mask = None
        
    def forward(self, x, use_old_mask=False):
        if not self.training:
            return x
    
        if not use_old_mask:
            self.mask = torch.bernoulli(torch.full_like(x, self.kept_p)) / self.kept_p
        return self.mask * x
            
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

    def forward(self, x):
        x_trans = self.group_linear(x)      # [B0, 1792*4, U] -> [B0, 1792*4, U]
        x_act   = self.mid_act_fn(x_trans)  # [B0, 1792*4, U]
        return x

class MMSharedMid(nn.Module):
    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.shared_linear  = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn     = config.act_fun

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

        if reshaped:
            # restore the original shape
            x_act       = x_act.permute([0, 1, 3, 2]).reshape(x.shape)

        return x_act

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
        mode_scores = self.feat2score(score_basis)
        attn_probs  = mode_scores.softmax(dim=self.group_dim)
        x_aggr      = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
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
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allmode)
        self.eval_robustness = config.eval_robustness
        self.base_initializer_range = config.base_initializer_range
        
        print("%s: pool_modes_feat=%s, mid_type=%s, trans_output_type=%s" % \
                (self.name, config.pool_modes_feat, config.mid_type,
                 config.trans_output_type))

        if config.pool_modes_feat[0] == '[':
            self.pool_modes_keepdim = True
            self.pool_modes_feat = config.pool_modes_feat[1:-1]     # remove '[' and ']'
        else:
            self.pool_modes_keepdim = False
            self.pool_modes_feat = config.pool_modes_feat

        self.pool_modes_basis = config.pool_modes_basis
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

        self.apply_attn_early = config.apply_attn_early

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

    def forward(self, input_feat, attention_probs, attention_scores):
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
        
        if self.apply_attn_early or self.eval_robustness:
            # mm_first_feat_4d: [B, 4, U2, 1792]
            mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)

            # attention_probs: [B, Modes, U1, U2]
            # mm_first_feat_fusion: [B, 4, U1, 1792]
            mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
            mm_first_feat_fusion_3d = mm_first_feat_fusion.transpose(2, 3).reshape(B, M*F, U1)
            mm_first_feat = mm_first_feat_fusion_3d
            # Skip the transformation layers on fused value to simplify the analyzed pipeline.
            if self.eval_robustness:
                trans_feat = self.feat_softaggr(mm_first_feat_fusion)
                return trans_feat

        # mm_mid_feat:   [B, 1792*4, U1]. Group linear & gelu of mm_first_feat.
        mm_mid_feat   = self.intermediate(mm_first_feat)
        # mm_last_feat:  [B, 4, U1, 1792]. Group/shared linear & residual & Layernorm
        mm_last_feat = self.output(mm_mid_feat, mm_first_feat)

        if (attention_probs is not None) and (not self.apply_attn_early):
            # matmul(t1, t2): (h1, w1), (w1, w2) => (h1, w2)
            # [B, 8, U1, U2][B, 4, U2, 1792] -> mm_trans_feat: [B, 4, U1, 1792]
            mm_trans_feat = torch.matmul(attention_probs, mm_last_feat)
        else:
            mm_trans_feat = mm_last_feat

        if self.pool_modes_feat == 'softmax':
            if self.pool_modes_basis == 'feat':
                trans_feat = self.feat_softaggr(mm_trans_feat)
            else:
                trans_feat = self.feat_softaggr(mm_trans_feat, attention_scores)

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
        self.config     = config
        self.name       = name
        self.num_modes  = config.num_modes
        self.in_feat_dim    = config.in_feat_dim
        self.feat_dim       = config.feat_dim
        self.attention_mode_dim = self.in_feat_dim // self.num_modes   # 448
        self.attn_clip    = config.attn_clip
        # att_size_allmode: 512 * modes
        self.att_size_allmode = self.num_modes * self.attention_mode_dim
        self.cross_attn_score_scale = config.cross_attn_score_scale
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode)
        self.key   = nn.Linear(self.in_feat_dim, self.att_size_allmode)
        self.base_initializer_range = config.base_initializer_range

        if config.ablate_multihead:
            self.out_trans  = MultiHeadFeatTrans(config, name)
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name)

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.keep_attn_scores = True
        self.tie_qk_scheme = config.tie_qk_scheme
        self.pos_in_attn_only = config.pos_in_attn_only
        print("{} in_feat_dim: {}, feat_dim: {}".format(self.name, self.in_feat_dim, self.feat_dim))

    # if tie_qk_scheme is not None, it overrides the initialized self.tie_qk_scheme
    def tie_qk(self, tie_qk_scheme=None):
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

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

    def forward(self, query_feat, query_vfeat, key_feat=None, key_vfeat=None, attention_mask=None):
        # query_feat: [B, U1, 1792]
        # if key_feat == None: self attention.
        if key_feat is None:
            key_feat  = query_feat
            key_vfeat = query_vfeat
            
        # mixed_query_layer, mixed_key_layer: [B, U1, 1792], [B, U2, 1792]
        mixed_query_layer = self.query(query_feat)
        mixed_key_layer   = self.key(key_feat)
        # query_layer, key_layer: [B, 4, U1, 448], [B, 4, U2, 448]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B0, 4, U1, 448] [B0, 4, 448, U2]
        attention_scores = attention_scores * self.cross_attn_score_scale / math.sqrt(self.attention_mode_dim)  # [B0, 4, U1, U2]

        with torch.no_grad():
            curr_max_attn = attention_scores.max().item()
            pos_count     = (attention_scores > 0).sum()
            curr_avg_attn = attention_scores.sum() / pos_count
            curr_avg_attn = curr_avg_attn.item()

        global max_attn, avg_attn, clamp_count, call_count
        if curr_max_attn > max_attn:
            max_attn = curr_max_attn
        avg_attn = curr_avg_attn

        verbose = False
        if curr_max_attn > self.attn_clip:
            if verbose:
                print("Warn: max attention {} > {}".format(curr_max_attn, self.attn_clip))
            attention_scores = torch.clamp(attention_scores, -self.attn_clip, self.attn_clip)
            clamp_count += 1
        call_count += 1
        if call_count % 500 == 0:
            print("max-attn: {:.2f}, avg-attn: {:.2f}, clamp-count: {}".format(max_attn, avg_attn, clamp_count))

        # attention_scores_premask: unmasked attention_scores
        attention_scores_premask = attention_scores
        # Apply the attention mask
        if attention_mask is not None:
            # [B0, 8, U1, U2] + [B0, 1, 1, U2] -> [B0, 8, U1, U2]
            attention_scores = attention_scores + attention_mask

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
        if self.pos_in_attn_only:
            trans_in_key_feat = key_vfeat
        else:
            trans_in_key_feat = key_feat
            
        out_feat    = self.out_trans(trans_in_key_feat, attention_probs, attention_scores_premask)

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
        self.in_ator_trans  = CrossAttFeatTrans(config1, name + '-in-squeeze')
        self.ator_out_trans = CrossAttFeatTrans(config, name + '-squeeze-out')
        self.attractors     = Parameter(torch.randn(1, self.num_attractors, self.in_feat_dim))
    
    def forward(self, in_feat, in_vfeat, attention_mask=None):
        # in_feat: [B, 196, 1792]
        batch_size = in_feat.shape[0]
        batch_attractors = self.attractors.expand(batch_size, -1, -1)
        new_batch_attractors = self.in_ator_trans(batch_attractors, batch_attractors, in_feat, in_vfeat)
        out_feat = self.ator_out_trans(in_feat, in_vfeat, new_batch_attractors, new_batch_attractors)
        self.attention_scores = self.ator_out_trans.attention_scores
        return out_feat
        
class SegtranFusionEncoder(nn.Module):
    def __init__(self, config, name):
        super().__init__()
        self.name = name
        self.num_translayers        = config.num_translayers
        self.pos_embed_every_layer  = config.pos_embed_every_layer
        self.translayer_compress_ratios = config.translayer_compress_ratios
        self.translayer_dims        = config.translayer_dims
        self.dropout                = nn.Dropout(config.hidden_dropout_prob)
        self.use_squeezed_transformer  = config.use_squeezed_transformer
        print(f'Segtran {self.name} Encoder with {self.num_translayers} trans-layers')

        self.pos_embed_layer = SegtranInputFeatEncoder(config)
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

    # if pos_embed_every_layer=True, then vfeat is vis_feat.
    # Otherwise, vfeat is combined feat.
    def forward(self, vfeat, voxels_pos, vmask):
        self.layers_vfeat = []
        if self.pos_embed_every_layer:
            MAX_POS_LAYER = self.num_translayers
        else:
            MAX_POS_LAYER = 1

        for i, translayer in enumerate(self.translayers):
            if i < MAX_POS_LAYER:
                # Add pos embeddings to transformer input features in every layer,
                # to avoid loss of positional signals after transformations.
                vfeat_normed    = self.vfeat_norm_layers[i](vfeat)
                pos_embed       = self.pos_embed_layer(voxels_pos) # self.pos_embed_layers[i](voxels_pos)
                feat_comb       = vfeat_normed + pos_embed[:, :, :self.translayer_dims[i]]
                feat_normed     = self.comb_norm_layers[i](feat_comb)
                # Only do dropout in the first layer.
                # In later layers, dropout is already applied at the output end. Avoid doing it again.
                if i == 0:
                    feat_normed  = self.dropout(feat_normed)
                    vfeat_normed = self.dropout(vfeat_normed)
                    
                feat_masked     = feat_normed * vmask
                vfeat_masked    = vfeat_normed * vmask
                vfeat           = translayer(feat_masked, vfeat_masked)
            else:
                feat_masked = vfeat * vmask
                vfeat = translayer(feat_masked, feat_masked)
            self.layers_vfeat.append(vfeat)

        return vfeat

# =================================== Segtran BackBone Components ==============================#

class PosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, omega=1, affine=True):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_fc = nn.Linear(self.pos_dim, self.pos_embed_dim, bias=True)
        self.pos_mix_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        self.omega = omega

    def forward(self, pos_normed):
        pos_embed_sum = 0
        pos_embed0 = self.pos_fc(pos_normed)
        pos_embed_sin = torch.sin(self.omega * pos_embed0[:, :, 0::2])
        pos_embed_cos = torch.cos(self.omega * pos_embed0[:, :, 1::2])
        # Interlace pos_embed_sin and pos_embed_cos.
        pos_embed_mix = torch.stack((pos_embed_sin, pos_embed_cos), dim=3).view(pos_embed0.shape)
        pos_embed_out = self.pos_mix_norm_layer(pos_embed_mix)

        return pos_embed_out

class SegtranInputFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim = config.trans_in_dim  # 1792
        self.pos_embed_dim = self.feat_dim

        # self.feat_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)

        # Box position encoding. no affine, but could have bias.
        # 2 channels => 1792 channels
        if config.ablate_pos_embed_type is None:
            self.pos_embedder = PosEmbedder(config.pos_dim, self.pos_embed_dim, omega=1, affine=False)
        elif config.ablate_pos_embed_type == 'rand':
            self.pos_embedder = RandPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.ablate_pos_embed_type == 'sinu':
            self.pos_embedder = SinuPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.ablate_pos_embed_type == 'zero':
            self.pos_embedder = ZeroEmbedder(self.pos_embed_dim)

        # comb_norm_layer has no affine, to maintain the proportion of visual features
        # self.comb_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.pos_embed_every_layer = config.pos_embed_every_layer

    def forward(self, voxels_pos):
        # batch_feat_fpn, vis_feat:  [2, 1296, 1792]
        # vis_feat = self.feat_norm_layer(batch_feat)

        # voxels_pos, voxels_pos_normed: [2, 1296, 2]
        voxels_pos_normed = voxels_pos / voxels_pos.max()
        # voxels_pos_normed: [B0, num_voxels, 2]
        # pos_embed:         [B0, num_voxels, 1792]
        pos_embed = self.pos_embedder(voxels_pos_normed)

        return pos_embed # vis_feat

def get_all_indices(shape, device):
    indices = torch.arange(shape.numel(), device=device).view(shape)

    out = []
    for dim_size in reversed(shape):
        out.append(indices % dim_size)
        indices = indices // dim_size
    return torch.stack(tuple(reversed(out)), len(shape))

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
