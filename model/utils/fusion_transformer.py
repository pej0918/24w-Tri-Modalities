import collections

from timm.models.vision_transformer import trunc_normal_
import torch.nn as nn
from functools import partial
import torch
from model.utils.layers import FusionBlock


class FusionTransformer(nn.Module):
    def __init__(self, embed_dim=4096, depth=1, num_heads=64, mlp_ratio=1, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 act_layer=None,
                 use_cls_token=True,
                 num_classes=20
                 ):
        super().__init__()

        self.embed_dim = embed_dim

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.masking_token = nn.Parameter(torch.zeros(embed_dim))

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            FusionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
            )
            for i in range(depth)])

        self.norm = norm_layer(embed_dim) # TODO: not needed, remove?
        self.init_weights()

        self.mlp_head = nn.Linear(embed_dim, num_classes)

    def init_weights(self):
        trunc_normal_(self.masking_token, std=.02)
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(_init_vit_weights)

    def forward(self, key, query, key_modal='', query_modal=''):
    # def forward(self, key, query):
        # concatenate tokens
        # tokens = {}
        # if text is not None:
        #     tokens['text'] = text
        # if video is not None:
        #     tokens['video'] = video
        # if audio is not None:
        #     tokens['audio'] = audio

        token_k = key.view(key.shape[0],1,self.embed_dim)
        token_q = query.view(query.shape[0],1,self.embed_dim)

        # tokens = [x for x in data if x is not None]
        # tokens = torch.cat(tokens, dim=1)

        # concatenate attention masks
        # tokens_mask = [x['attention_mask'] for x in data if x is not None]
        # tokens_mask = torch.cat(tokens_mask, dim=1)

        # print('original token_k:', token_k.shape)
        # concatenate cls token
        if self.cls_token is None:
            offset = 0
        else:
            cls_token = self.cls_token.expand(token_k.shape[0], -1, -1)
            # print('shape of cls_token:', cls_token.shape)
            token_k = torch.cat((cls_token, token_k), dim=1)

            cls_token = self.cls_token.expand(token_q.shape[0], -1, -1)
            token_q = torch.cat((cls_token, token_q), dim=1)

            # cls_token_mask = torch.ones((1, 1)).to(tokens_mask.device).expand(tokens_mask.shape[0], -1)
            # tokens_mask = torch.cat((cls_token_mask, tokens_mask), dim=1)
            offset = 1
        
        # print('cls + token_k:', token_k.shape)

        for block in self.blocks:
            tokens = block(token_k, token_q)
        
        # print('output:', tokens.shape)

        output = collections.OrderedDict()

        def _get_average(tokens, attention_mask):
            attention_mask = attention_mask.unsqueeze(2).expand_as(tokens)
            return (tokens * attention_mask).sum(1) / attention_mask.sum(1)

        # if text is not None:
        #     n_tokens = text['all_tokens'].size(1)
        #     attention_mask = text['attention_mask']
        #     all_tokens = tokens[:, offset:offset + n_tokens]

        #     offset += n_tokens
        #     output['text'] = {
        #         "all_tokens": all_tokens,
        #         "attention_mask": attention_mask,
        #     }

        # if video is not None:
        #     n_tokens = video['all_tokens'].size(1)
        #     attention_mask = video['attention_mask']
        #     all_tokens = tokens[:, offset:offset + n_tokens]

        #     offset += n_tokens
        #     output['video'] = {
        #         "all_tokens": all_tokens,
        #         "attention_mask": attention_mask,
        #     }

        # if audio is not None:
        #     n_tokens = audio['all_tokens'].size(1)
        #     attention_mask = audio['attention_mask']
        #     all_tokens = tokens[:, offset: offset + n_tokens]

        #     offset += n_tokens
        #     output['audio'] = {
        #         "all_tokens": all_tokens,
        #         "attention_mask": attention_mask,
        #     }


        # if self.cls_token is None:
        #     for key, value in output.items():
        #         output[key]['embed'] = _get_average(value["all_tokens"], value['attention_mask'])
        # else:
        #     modalities = list(output.keys())
        #     modalities = '_'.join(modalities)
        #     if modalities not in output:
        #         output[modalities] = {}
        #     output[modalities]['embed'] = tokens[:, 0]

        output = tokens[:,0,:].squeeze(1)
        output = self.mlp_head(output)

        return output



def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)