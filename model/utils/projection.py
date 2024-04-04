from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F

from model.utils.davenet import load_DAVEnet

class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)  # 차원 맞추기
        self.cg = Context_Gating(output_dimension)  # Context Gating

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return th.max(x, dim=1)[0]

class Fused_Gated_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Fused_Gated_Unit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class projection_net(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            video_dim=4096,
            we_dim=300,
            cross_attention=False
    ):
        super(projection_net, self).__init__()
        self.cross_attention = cross_attention

        # Fuse적용 X
        if not cross_attention:
            self.DAVEnet = load_DAVEnet(v2=True)
            self.GU_audio = Gated_Embedding_Unit(4096, embed_dim)
            self.GU_video = Gated_Embedding_Unit(video_dim, embed_dim)
            # self.text_pooling_caption = Sentence_Maxpool(we_dim, embed_dim)
            self.GU_text_captions = Gated_Embedding_Unit(we_dim, embed_dim)
        else:
            self.DAVEnet_projection = nn.Linear(1024, embed_dim // 2)
            self.video_projection = nn.Linear(video_dim, embed_dim // 2)
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embed_dim // 2)
            self.GU_fuse = Fused_Gated_Unit(embed_dim // 2, embed_dim)

    def forward(self, video, audio_input, nframes, text=None):
        audio = self.DAVEnet(audio_input)  # [16, 1024, 320]
        audio = audio.permute(0, 2, 1)


        if self.cross_attention:
            # 차원수 조절
            video = self.video_projection(video)
            audio = self.DAVEnet_projection(audio)
            text = self.text_pooling_caption(text)
            # GU FUSE
            audio_text = self.GU_fuse(audio, text)
            audio_video = self.GU_fuse(audio, video)
            text_video = self.GU_fuse(text, video)
            return audio_text, audio_video, text_video
        else:
            text = self.GU_text_captions(text)
            audio = self.GU_audio(audio)  # [16,5120] -> [16,4096]  [16, 1024, 320] ->
            video = self.GU_video(video)  # [16,40*4096] -> [16,4096]
            return audio, text, video