from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
from model_davenet import load_DAVEnet

# 입력과 출력의 차원수가 동일하다
class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)  

    def forward(self, x):
        x1 = self.fc(x)          
        x = th.cat((x, x1), 1)   # 차원 = 2 * dimension
        return F.glu(x, 1)       # 차원 = dimension , glu가 반만 이용
    
# 입력차원을 출력차원으로 맞춰주고 context_gating을 통과시켜준다
class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)  # 차원 맞추기
        self.cg = Context_Gating(output_dimension)              # Context Gating 

    def forward(self, x):
        x = self.fc(x)         
        x = self.cg(x)         
        return x               
    
# 차원수도 조절해주고 max pooling도 적용해준다. ex. (300,20) -> (1024)
class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim) #차원 맞추기

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return th.max(x, dim=1)[0]  # max pooling으로 (1024,20) -> (1024)
    
# 이때 input_dimension은 embed_dim/2로 들어가야해!
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
    
class Net(nn.Module):
    def __init__(
            self,
            embd_dim=1024,
            video_dim=2048,
            we_dim=300,
            cross_attention=False
    ):
        super(Net, self).__init__()
        self.DAVEnet = load_DAVEnet() 
        self.cross_attention = cross_attention

        # Fuse적용 X
        if cross_attention:
            self.GU_audio = Gated_Embedding_Unit(1024, embd_dim)
            self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
            self.GU_text_captions = Gated_Embedding_Unit(embd_dim, embd_dim)
        else:
            # 각각의 차원: 원하는 차원 // 2
            self.DAVEnet_projection = nn.Linear(1024, embd_dim // 2) 
            self.video_projection = nn.Linear(video_dim, embd_dim // 2)
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim // 2)
            # correlation반영
            self.GU_fuse= Fused_Gated_Unit(embd_dim // 2, embd_dim)

    def forward(self, video, audio_input, nframes, text=None):
        audio = self.DAVEnet(audio_input)
        if not self.training: # controlled by net.train() / net.eval() (use for downstream tasks) 
            pooling_ratio = round(audio_input.size(-1) / audio.size(-1))    # 입력 오디오 길이와 오디오 임베딩 길이 계산
            nframes.div_(pooling_ratio)                                     # 오디오 프레임 수를 풀링 비율로 나눈다.
            audioPoolfunc = th.nn.AdaptiveAvgPool2d((1, 1))                 # 입력 길이 맞추기
            audio_outputs = audio.unsqueeze(2)                              # 풀링을 위해 차원 추가
            pooled_audio_outputs_list = []
            for idx in range(audio.shape[0]):
                nF = max(1, nframes[idx])
                pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
            audio = th.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
        else:
            audio = audio.mean(dim=2) # this averages features from 0 padding too

        if not self.cross_attention:
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
            # 차원수 조절 + GU
            text = self.GU_text_captions(self.text_pooling_caption(text))
            audio = self.GU_audio(audio)
            video = self.GU_video(video)
            return audio, text, video