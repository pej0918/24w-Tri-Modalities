import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn as nn

from model.utils.utils import normalize_embeddings
from model.utils.layers import get_projection
from model.utils.fusion_transformer import FusionTransformer
from model.utils.davenet import load_DAVEnet
from model.utils.projection import projection_net

class EverythingAtOnceModel(nn.Module):
    def __init__(self,
                 args,
                 embed_dim=1024,
                 video_embed_dim=4096,
                 text_embed_dim=300,
                 video_max_tokens=None,
                 text_max_tokens=None,
                 audio_max_num_STFT_frames=None,
                 projection_dim=6144,
                 projection='gated',
                 strategy_audio_pooling='none',
                 davenet_v2=True,
                 individual_projections=True,
                 use_positional_emb=False,
                 use_softmax=False
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_softmax = use_softmax

        self.fusion = FusionTransformer(embed_dim=self.embed_dim, use_softmax=self.use_softmax)

        self.args = args
        self.token_projection = args.token_projection

        self.individual_projections = individual_projections
        self.use_positional_emb = use_positional_emb
        self.strategy_audio_pooling = strategy_audio_pooling

        self.video_norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.text_norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.audio_norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.norm_layer = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # audio token preprocess
        self.davenet = load_DAVEnet(v2=davenet_v2)

        if audio_max_num_STFT_frames is not None:
            if davenet_v2:
                audio_max_tokens = int(audio_max_num_STFT_frames / 64)
            else:
                audio_max_tokens = int(audio_max_num_STFT_frames / 16)
            self.audio_max_tokens = audio_max_tokens
        else:
            self.audio_max_tokens = None

        if self.use_positional_emb:
            assert video_max_tokens is not None
            assert text_max_tokens is not None
            assert audio_max_num_STFT_frames is not None
            self.video_pos_embed = nn.Parameter(torch.zeros(1, video_max_tokens, self.embed_dim))
            self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_tokens, self.embed_dim))
            self.audio_pos_embed = nn.Parameter(torch.zeros(1, self.audio_max_tokens, self.embed_dim))
        else:
            self.video_pos_embed = None
            self.text_pos_embed = None
            self.audio_pos_embed = None

        audio_embed_dim = 4096 if davenet_v2 else 1024
        if self.token_projection == 'projection_net':
            self.token_proj = projection_net(embed_dim=self.embed_dim)
        else:
            self.video_token_proj = get_projection(video_embed_dim, self.embed_dim, self.token_projection)
            self.text_token_proj = get_projection(text_embed_dim, self.embed_dim, self.token_projection)
            self.audio_token_proj = get_projection(audio_embed_dim, self.embed_dim, self.token_projection)
        
        # if not self.individual_projections:
        #     self.proj = get_projection(embed_dim, projection_dim, projection)
        # else:
        #     self.video_proj = get_projection(embed_dim, projection_dim, projection)
        #     self.text_proj = get_projection(embed_dim, projection_dim, projection)
        #     self.audio_proj = get_projection(embed_dim, projection_dim, projection)

        self.init_weights()

    def init_weights(self):
        for weights in [self.video_pos_embed, self.audio_pos_embed, self.text_pos_embed]:
            if weights is not None:
                trunc_normal_(weights, std=.02)

    def _check_and_fix_if_input_empty(self, x, attention_mask):
        nonempty_input_mask = attention_mask.sum(-1) != 0

        # if all tokens of modality is empty, add one masking token
        empty_input_mask = nonempty_input_mask == 0
        n_masking_tokens = 1
        x[empty_input_mask, :n_masking_tokens] = self.fusion.masking_token.type(x.dtype)
        attention_mask[empty_input_mask, :n_masking_tokens] = 1
        return x, attention_mask, nonempty_input_mask

    def extract_video_tokens(self, video):
        x = self.video_token_proj(video)
        x = self.video_norm_layer(x)

        # x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        # special_token_mask = attention_mask == 0

        return x

    def extract_audio_tokens(self, audio, audio_STFT_nframes):
        audio = self.davenet(audio)
        audio = audio.permute(0, 2, 1)

        # coef = int(np.ceil(attention_mask.shape[1] / audio.shape[1]))
        # attention_mask = torch.nn.functional.max_pool1d(attention_mask.unsqueeze(0), kernel_size=coef).squeeze(0)
        # audio_STFT_nframes = (audio_STFT_nframes / coef).int()

        # if (self.audio_max_tokens is not None) and (audio.shape[1] > self.audio_max_tokens):
        #     new_audio, new_audio_mask = [], []
        #     for i in range(len(audio)):
        #         cur_audio, cur_audio_mask = create_audio_tokens(
        #             audio[i], attention_mask[i], audio_STFT_nframes[i], self.audio_max_tokens, strategy=self.strategy_audio_pooling)
        #         new_audio.append(cur_audio)
        #         new_audio_mask.append(cur_audio_mask)
        # new_audio = [a for a in range(len(audio))]
        # audio = torch.stack(new_audio, dim=0)
        # # attention_mask = torch.stack(new_audio_mask, dim=0)

        audio = self.audio_token_proj(audio)
        audio = self.audio_norm_layer(audio)

        # audio, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(audio, attention_mask)
        # special_token_mask = attention_mask == 0
        return audio
    
    def extract_text_tokens(self, text):
        x = self.text_token_proj(text)
        x = self.text_norm_layer(x)

        # x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        # special_token_mask = attention_mask == 0
        return x
    
    def extract_tokens(self, video, audio, text, nframes):
        audio, text, video = self.token_proj(video, audio, nframes, text)
        audio = self.norm_layer(audio)
        text = self.norm_layer(text)
        video = self.norm_layer(video)
        return audio, text, video

    def forward(self, video, audio, nframes, text, category, force_cross_modal=False):
        output = {}

        if self.token_projection == 'projection_net':
            audio_raw_embed, text_raw_embed, video_raw_embed = self.extract_tokens(video, audio, text, nframes)
        else:
            text_raw_embed = self.extract_text_tokens(text) # [16, 30, 4096]
            video_raw_embed = self.extract_video_tokens(video) # [16, 4096]
            audio_raw_embed = self.extract_audio_tokens(audio, nframes) # [16, 80, 4096]

        va = self.fusion(key=video_raw_embed,
                            query=audio_raw_embed)
        at = self.fusion(key=audio_raw_embed,
                            query=text_raw_embed)
        tv = self.fusion(key=text_raw_embed,
                            query=video_raw_embed)
        # print('va:',va.shape,'at:',at.shape,'tv:',tv.shape)

        # output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        # output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        # output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        # # add positional embedding after masking
        # if self.use_positional_emb:
        #     text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
        #     video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed
        #     audio_raw_embed['all_tokens'] = audio_raw_embed['all_tokens'] + self.audio_pos_embed

        # text = self.fusion(text=text_raw_embed)['text']
        # video = self.fusion(video=video_raw_embed)['video']
        # audio = self.fusion(audio=audio_raw_embed)['audio']

        # if self.individual_projections:
        #     text_proj, video_proj, audio_proj = self.text_proj, self.video_proj, self.audio_proj
        # else:
        #     text_proj, video_proj, audio_proj = self.proj, self.proj, self.proj

        # output["text_embed"] = text_proj(text['embed'])
        # output["video_embed"] = video_proj(video['embed'])
        # output["audio_embed"] = audio_proj(audio['embed'])

        # if self.cross_modal or force_cross_modal:
        #     tv = self.fusion(text=text_raw_embed,
        #                      video=video_raw_embed)
        #     ta = self.fusion(text=text_raw_embed,
        #                      audio=audio_raw_embed)
        #     va = self.fusion(video=video_raw_embed,
        #                      audio=audio_raw_embed)

        #     if self.fusion.cls_token is not None:
        #         assert not self.individual_projections
        #         output["tv_embed"] = self.proj(tv['text_video']['embed'])
        #         output["ta_embed"] = self.proj(ta['text_audio']['embed'])
        #         output["va_embed"] = self.proj(va['video_audio']['embed'])
        #     else:
        #         output["tv_embed"] = (normalize_embeddings(text_proj(tv['text']['embed'])) +
        #                               normalize_embeddings(video_proj(tv['video']['embed']))) / 2

        #         output["ta_embed"] = (normalize_embeddings(text_proj(ta['text']['embed'])) +
        #                               normalize_embeddings(audio_proj(ta['audio']['embed']))) / 2

        #         output["va_embed"] = (normalize_embeddings(video_proj(va['video']['embed'])) +
        #                               normalize_embeddings(audio_proj(va['audio']['embed']))) / 2

        # if force_cross_modal:
        #     #  needed for ablation
        #     output["t+v_embed"] = (normalize_embeddings(output["text_embed"]) +
        #                            normalize_embeddings(output["video_embed"])) / 2
        #     output["t+a_embed"] = (normalize_embeddings(output["text_embed"]) +
        #                            normalize_embeddings(output["audio_embed"])) / 2
        #     output["v+a_embed"] = (normalize_embeddings(output["video_embed"]) +
        #                            normalize_embeddings(output["audio_embed"])) / 2

        return va, at, tv

