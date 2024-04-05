from argparse import Namespace

import numpy as np
import os
from pathlib import Path
import pickle
import re

from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from gensim.models.keyedvectors import KeyedVectors
from model.fusion_model import EverythingAtOnceModel

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

YMCA_ID = "./source/CA6-ATA_MLP3_16batch_200epochs.pth"
WORD_EMBEDDING_ID = "./source/GoogleNews-vectors-negative300.bin"
TEST_DATA_PATH = "./source/msrvtt_category_test.pkl"

CATEGORIES = ["music", "people", "gaming", "sports/actions", "news/events/politics", "education", "tv shows", "movie/comedy", "animation", "vehicles/autos", "howto", "travel", "science/technology", "animals/pets", "kids/family", "documentary", "food/drink", "cooking", "beauty/fashion", "advertisement"]

class YMCA:
    def __init__(self, device) -> None:
        self.device = device
        self.checkpoint = torch.load(YMCA_ID)
        self.token_projection = 'projection_net'
        self.use_softmax = False
        self.use_cls_token = False
        self.num_classes = 20

        self.word_embedding = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_ID, binary=True)
        self.we_dim = 300
        # self.dataset = MSRVTT_DataLoader(data_path=TEST_DATA_PATH, we=self.word_embedding)
        self.dataset = pickle.load(open(TEST_DATA_PATH, 'rb'))
        

        args = Namespace(
            token_projection=self.token_projection,
            use_softmax=self.use_softmax,
            use_cls_token=self.use_cls_token,
            num_classes=self.num_classes,
        )

        self.net = EverythingAtOnceModel(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr =0.001)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.net.eval()

        self.device_msg = 'Tested on GPU.' if 'cuda' in self.device else 'Tested on CPU.'

    def _tokenize_text(self, sentence):  #텍스트를 단어 또는 부분 문자열로 분할
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _zero_pad_tensor(self, tensor, size):  #입력 텐서의 크기를 고정된 크기로 만들기
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _words_to_we(self, words):
        max_words = 30
        words = [word for word in words if word in self.word_embedding.key_to_index]
        if words:
            we = self._zero_pad_tensor(self.word_embedding[words], max_words)
            return torch.from_numpy(we)
        else:
            return torch.zeros(max_words, self.we_dim)

    def _preprocessing(self, data):
        num_frames_multiplier = 5
        video_id = data['id']
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(torch.from_numpy(data['2d_pooled']).float(), dim=0)
        feat_3d = F.normalize(torch.from_numpy(data['3d_pooled']).float(), dim=0)
        video = torch.cat((feat_2d, feat_3d)) #2D와 3D 특징을 결합하여 하나의 비디오 특징 생성합니다.

        # load audio and zero pad/truncate if necessary
        audio = data['audio']  #오디오의 특징 가져오기
        target_length = 1024 * num_frames_multiplier
        nframes = audio.numpy().shape[1]
        p = target_length - nframes #오디오의 길이를 확인하고 부족한 경우 패딩을 추가.
        if p > 0:
            audio = np.pad(audio, ((0,0),(0,p)), 'constant', constant_values=(0,0))
        elif p < 0:
            audio = audio[:,0:p]
        audio = torch.FloatTensor(audio)

        # choose a caption
        caption = data['eval_caption']
        caption = self._words_to_we(self._tokenize_text(caption))

        # category 추가
        category = data['category']

        return {'video': video, 'text': caption, 'video_id': video_id,
                'audio': audio, 'nframes': nframes, 'category': category}

    def inference(self, video, text):
        caption = text.replace("[", "").replace("]", "").replace("'", "")
        # Get the index number from matching the caption with the dataset
        examples = self.get_example_list()
        video_file_name=""
        for video_path, target_caption in examples:
            if caption == target_caption[0]:
                video_file_name = video_path
                break
        # split the video file name to get the video number
        data_id = video_file_name.split('/')[-1][:-4]

        data = None
        for get_data in self.dataset:
            if get_data['id'] == data_id:
                data = get_data
                break
        data = self._preprocessing(data)
        video_f = data['video'].to(self.device)
        audio_f = data['audio'].to(self.device)
        text_f = data['text'].to(self.device)
        nframes = torch.tensor(data['nframes'], dtype=torch.int64).to(self.device)
        category = torch.tensor(data['category'], dtype=torch.int64).to(self.device)
        
        # add dimension for batch size
        video_f = video_f.unsqueeze(0)
        audio_f = audio_f.unsqueeze(0)
        text_f = text_f.unsqueeze(0)

        video_f = video_f.view(-1, video_f.shape[-1])
        audio_f = audio_f.view(-1, audio_f.shape[-2], audio_f.shape[-1])
        text_f = text_f.view(-1, text_f.shape[-2], text_f.shape[-1])

        va, at, tv = self.net(video_f, audio_f, nframes, text_f, category)
        soft_vote = (va + at + tv) / 3    #각 클래스별 확률 값
        print(f"** Soft Vote: {len(soft_vote)}")
        _, pred_category = torch.max(soft_vote, 1)   #예측한 class 값
        accuracy = (pred_category == category).sum().item()
        print(f"** Accuracy: {accuracy}")

        return dict(zip(CATEGORIES, map(float, soft_vote)))

    def get_example_list(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return [[str(Path(current_dir, "source/video7061.mp4")), ["goldfish chase each other around a blue tank to music"]],
                [str(Path(current_dir, "source/video7118.mp4")), ["a young girl in a horror movie is haunted"]]
]

    def get_css(self):
        return """
        img {
            margin: 0 auto;
            display:block;
        }
        h1 {
            text-align: center;
            display:block;
        }
        h3 {
            text-align: center;
            display:block;
        }
        """