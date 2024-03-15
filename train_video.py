import random
import os
import re
import time
import pickle
import numpy as np
import shutil
import pandas as pd
# from args import get_args
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from gensim.models.keyedvectors import KeyedVectors

import torch
import argparse

torch.backends.cudnn.benchmark = True
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import json

#triplenet-network
from tripletnet import Tripletnet, Net, AverageMeter
from visdom import Visdom
from torchvision import datasets, transforms
from torch.autograd import Variable

data_path = "data/msrvtt_category_train-002.pkl"
anchor_path = "data/video_anchor.pkl"
seed = 42
margin = 0.2
lr = 0.01
momentum = 0.5
epochs = 100
batch_size = 16
best_acc = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(seed)




class Video_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            data_path,
            anchor_path
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data_path, 'rb'))
        self.anchor_dict = pickle.load(open(anchor_path, 'rb'))
        self.num_videos = len(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_id = self.data[idx]['id']

        # Positive
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(torch.from_numpy(self.data[idx]['2d_pooled']).float(), dim=0) #2D 데이터 정규화
        feat_3d = F.normalize(torch.from_numpy(self.data[idx]['3d_pooled']).float(), dim=0) #3D 데이터 정규화
        positive = torch.cat((feat_2d, feat_3d))

        # category
        category = self.data[idx]['category']

        # Anchor
        anchor = self.anchor_dict[category]

        # Negative
        negative_category = category
        while negative_category == category:  # 선택된 negative가 positive와 같은 category가 아닐 때까지 다시 선택
            random_idx = random.randint(0, self.num_videos - 1)
            negative_category = self.data[random_idx]['category']
        feat_2d_neg = F.normalize(torch.from_numpy(self.data[random_idx]['2d_pooled']).float(), dim=0)
        feat_3d_neg = F.normalize(torch.from_numpy(self.data[random_idx]['3d_pooled']).float(), dim=0)
        negative = torch.cat((feat_2d_neg, feat_3d_neg))

        return {'anchor': anchor, 'positive': positive, 'negative': negative, 'video_id': video_id, 'category': category, 'random_idx': random_idx, 'negative_category': negative_category
                }


def accuracy(dista, distb):
    margin = 0
    pred = (distb-dista-margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%('TripletNet')
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/'%('TripletNet') + 'model_best.pth.tar')


def train(video_dataloader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    tnet.train()
    for batch_idx, data_dict in enumerate(video_dataloader):
        anchor, positive, negative, category = data_dict['anchor'].cuda(), data_dict['positive'].cuda(), data_dict['negative'].cuda(),data_dict['category']
        anchor, positive, negative = torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)
        distp, distn = tnet(anchor, positive, negative)
        target = torch.FloatTensor(distp.size()).fill_(-1).cuda()
        # target.cuda()
        target = torch.tensor(target)

        loss = loss_triplet = criterion(distp, distn, target)
        acc = accuracy(distp, distn)
        losses.update(loss_triplet.data, anchor.size(0))
        accs.update(acc, anchor.size(0))

        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'.format(
                epoch, batch_idx * len(anchor), len(video_dataloader.dataset),
                losses.val, losses.avg,
                       100. * accs.val, 100. * accs.avg))


def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, data_dict in enumerate(test_loader):
        anchor = data_dict['anchor'].cuda()
        positive = data_dict['positive'].cuda()
        negative = data_dict['negative'].cuda()
        category = data_dict['category']

        anchor, positive, negative = torch.tensor(anchor), torch.tensor(positive), torch.tensor(negative)

        # compute output
        distp, distn = tnet(anchor, positive, negative)
        target = torch.FloatTensor(distp.size()).fill_(-1)
        target = target.cuda()
        target = torch.tensor(target)
        test_loss = criterion(distp, distn, target).item()

        # measure accuracy and record loss
        acc = accuracy(distp, distn)
        accs.update(acc, anchor.size(0))
        losses.update(test_loss, anchor.size(0))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    return accs.avg



video_dataset = Video_DataLoader(data_path = data_path, anchor_path = anchor_path)
dataset_size = len(video_dataset)
train_size = int(0.8 * dataset_size)
train_dataset, test_dataset = random_split(video_dataset, [train_size, dataset_size - train_size])
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=True,
    drop_last=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False,
    drop_last=False
)

model = Net()
tnet = Tripletnet(model)
tnet.cuda()
criterion = torch.nn.MarginRankingLoss(margin=margin)
optimizer = optim.SGD(tnet.parameters(), lr=lr, momentum=momentum)

for epoch in range(1, epochs + 1):
    # train for one epoch
    train(train_loader, tnet, criterion, optimizer, epoch)
    # evaluate on validation set
    acc = test(test_loader, tnet, criterion, epoch)

    # remember best acc and save checkpoint
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': tnet.state_dict(),
        'best_prec1': best_acc,
    }, is_best)
