from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import json
import random

from gensim.models.keyedvectors import KeyedVectors

#지원이가 알려준 코드 그대로~ 주석만 추가
class MSRVTT_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            data_path,
            annotation_path,
            we, #word embewdding
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5, #오디에 데이터 길이 조절용
            training=True,
            tri_modal=False, #비디오 + 오디오 (+ 텍스트) -> 텍스트를 사용할지 유무
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data_path, 'rb')) #pkl파일을 바이트 스트림(이진 모드)
        self.annotation_path = annotation_path
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.max_video = 30
        self.num_frames_multiplier = num_frames_multiplier
        self.training = training
        self.tri_modal = tri_modal
        with open(self.annotation_path, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):  #입력 텐서의 크기를 고정된 크기로 만들기
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):  #텍스트를 단어 또는 부분 문자열로 분할
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):  #단어를 임베딩 벡터로 변환
        words = [word for word in words if word in self.we.vocab]
        #words = [word for word in words if word in self.we.key_to_index]
        if words: #해당 단어가 임베딩 모델에 존재할 때 벡터 추출(학습 한 것만)
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def _get_caption(self, idx):
        """Chooses random caption if training. Uses set caption if evaluating."""
        if self.training: #훈련중일 경우 무작위 caption 가져오기
            captions = self.data[idx]['caption']
            caption = self._words_to_we(self._tokenize_text(random.choice(captions)))
            return caption
        else:
            caption = self.data[idx]['eval_caption']
            return self._words_to_we(self._tokenize_text(caption))


    def __getitem__(self, idx):
        video_id = self.data[idx]['id']  #비디오의 고유 식별자 가져오기
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d_pooled']).float(), dim=0) #2D 데이터 정규화
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d_pooled']).float(), dim=0) #3D 데이터 정규화
        video = th.cat((feat_2d, feat_3d)) #2D와 3D 특징을 결합하여 하나의 비디오 특징 생성합니다.

        # load audio and zero pad/truncate if necessary
        audio = self.data[idx]['audio']  #오디오의 특징 가져오기
        target_length = 1024 * self.num_frames_multiplier
        nframes = audio.numpy().shape[1]
        p = target_length - nframes #오디오의 길이를 확인하고 부족한 경우 패딩을 추가.
        if p > 0:
            audio = np.pad(audio, ((0,0),(0,p)), 'constant', constant_values=(0,0))
        elif p < 0:
            audio = audio[:,0:p]
        audio = th.FloatTensor(audio)

        # choose a caption
        caption=''
        caption = self._get_caption(idx)

        # category 추가
        video_info = next((video for video in self.annotations['videos'] if video['video_id'] == video_id), None)
        if video_info is not None:
            category = video_info['category']
        else:
            category = None

        return {'video': video, 'text': caption, 'video_id': video_id,
                'audio': audio, 'nframes': nframes, 'category': category}