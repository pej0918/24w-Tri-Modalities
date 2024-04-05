import os
from pathlib import Path

from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from gensim.models.keyedvectors import KeyedVectors
from model.fusion_model import EverythingAtOnceModel

import torch
from torch.utils.data import DataLoader

YMCA_ID = "./source/CA6-ATA_MLP3_16batch_200epochs.pth"
WORD_EMBEDDING_ID = "./source/GoogleNews-vectors-negative300.bin"
TEST_DATA_PATH = "./source/msrvtt_category_test.pkl"

class YMCA:
    def __init__(self, device) -> None:
        self.device = device
        self.checkpoint = torch.load(YMCA_ID)
        self.token_projection = 'projection_net'
        self.use_softmax = False
        self.use_cls_token = False
        self.num_classes = 20

        self.word_embedding = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_ID, binary=True)
        self.dataset = MSRVTT_DataLoader(data_path=TEST_DATA_PATH, we=self.word_embedding)

        self.args = {
            'token_projection': self.token_projection,
            'use_softmax': self.use_softmax,
            'use_cls_token': self.use_cls_token,
            'num_classes': self.num_classes
        }

        self.net = EverythingAtOnceModel(args=self.args).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr =0.001)
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.net.eval()

        self.device_msg = 'Tested on GPU.' if 'cuda' in self.device else 'Tested on CPU.'


    def inference(self, video, text):
        caption = text[0]
        # Get the index number from matching the caption with the dataset
        example_captions = self.get_example_list()
        for video_path, caption in example_captions:
            if caption == text:
                video_file_name = video_path
                break
        # split the video file name to get the video number
        index = int(video_file_name.split('/')[-1][5:-4])

        data = self.dataset.__getitem__(index)
        video_f = data['video'].to(self.device)
        audio_f = data['audio'].to(self.device)
        text_f = data['text'].to(self.device)
        nframes = data['nframes'].to(self.device)
        category = data['category'].to(self.device)

        video_f = video_f.view(-1, video_f.shape[-1])
        audio_f = audio_f.view(-1, audio_f.shape[-2], audio_f.shape[-1])
        text_f = text_f.view(-1, text_f.shape[-2], text_f.shape[-1])

        pred = self.net(video_f, audio_f, nframes, text_f, category)
        pred_category = torch.argmax(pred, dim=1)
        accuracy = (category == pred_category).sum().item() / category.shape[0]

        return pred_category, accuracy

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