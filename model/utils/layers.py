import torch
import torch as th
from torch import nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import DropPath, Mlp, Attention

import math


class GatedEmbeddingUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super().__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class FusedGatedUnit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(FusedGatedUnit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = ContextGating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class ContextGating(nn.Module):
    def __init__(self, dimension):
        super(ContextGating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class SentenceMaxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(SentenceMaxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return torch.max(x, dim=1)[0]


class FusionBlock(nn.Module):
    """
        Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        Copyright 2020, Ross Wightman
    """
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_softmax=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.cross_attn = MultiHeadCrossAttention(d_model=dim, n_head=num_heads, use_softmax=use_softmax)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, k, q, attention_mask=None):
        output = q + self.drop_path(self.cross_attn(self.norm1(k), self.norm1(q)))#, attention_mask))  ##### 1) query만 residual
        output = output + self.drop_path(self.mlp(self.norm2(output)))
        return output

class ScaleDotProductAttention(nn.Module):
    def __init__(self, use_softmax):
        super(ScaleDotProductAttention, self).__init__()
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, e=1e-12):
        # input size: [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2,3)
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        if self.use_softmax:
            score = self.softmax(score)  #[0,1]
        v = score @ v

        return v, score 
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_head, use_softmax):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.use_softmax = use_softmax
        self.attention = ScaleDotProductAttention(use_softmax)

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    
    def forward(self, k, q):
        v = k
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v)
        out = self.concat(out)
        out = self.w_concat(out)
        return out 

    def split(self, tensor):
        # [batch_size, length, d_model] -> [batch_size, head, length, d_model]
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1,2)
        return tensor 
    
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        tensor = tensor.transpose(1,2).contiguous().view(batch_size, length, self.d_model)
        return tensor

# class FusionAttention(Attention):
#     """
#     Adopted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     Copyright 2020, Ross Wightman
#     """
#     def forward(self, x, y, attention_mask=None):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         attn = (q @ k.transpose(-2, -1)) * self.scale

#         if attention_mask is not None:
#             zero_attention_mask = (attention_mask == 0).view(B, 1, 1, N).expand_as(attn)  # (bs, n_heads, q_length, k_length)
#             attn.masked_fill_(zero_attention_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


def get_projection(input_dim, output_dim, projection_type):
    if projection_type == 'minimal':
        return nn.Linear(input_dim, output_dim)
    if projection_type == 'gated':
        return GatedEmbeddingUnit(input_dim, output_dim)
    elif projection_type == '':
        return nn.Identity()
    else:
        raise NotImplementedError