import argparse
import collections
# from sacred import Experiment
# from neptunecontrib.monitoring.sacred import NeptuneObserver

import os
# import torch.optim as module_optim
# import torch.optim.lr_scheduler as module_lr_scheduler

# from everything_at_once import data_loader as module_data
# from everything_at_once import model as module_arch
# from everything_at_once import loss as module_loss

# from everything_at_once.trainer import Trainer
# from everything_at_once.metric import RetrievalMetric

# from parse_config import ConfigParser

from data_loader.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader

import time
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from functools import partial

import math
import tqdm

import matplotlib.pyplot as plt

def TrainOneBatch(model, opt, data, loss_fun, apex=False):
    video = data['video'].cuda()
    audio = data['audio'].cuda()
    text = data['text'].cuda()
    nframes = data['nframes'].cuda()
    category = data['category'].cuda()
    # print('video:', video.shape, 'audio:', audio.shape, 'text:', text.shape)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    nframes = nframes.view(-1)
    # print('video:', video.shape, 'audio:', audio.shape, 'text:', text.shape)

    opt.zero_grad()
    
    # AVLnet-Text independent audio and text branches
    va, at, tv = model(video, audio, nframes, text, category)
    loss_va = loss_fun(va, category)
    loss_at = loss_fun(at, category)
    loss_tv = loss_fun(tv, category)
    loss = loss_va + loss_at + loss_tv
    loss.backward()
    opt.step()
    return loss.item()

def get_soft_voting(va, at, tv):
    # Soft voting by averaging the probabilities
    soft_vote = (va + at + tv) / 3
    _, soft_vote_preds = torch.max(soft_vote, 1)
    return soft_vote_preds

def get_hard_voting(va_preds, at_preds, tv_preds):
    # Hard voting by selecting the most frequent prediction
    combined_preds = torch.stack((va_preds, at_preds, tv_preds), dim=1)
    hard_vote, _ = torch.mode(combined_preds, dim=1)
    return hard_vote

def get_predictions(va, at, tv):
    _, va_preds = torch.max(va, 1)
    _, at_preds = torch.max(at, 1)
    _, tv_preds = torch.max(tv, 1)
    return va_preds, at_preds, tv_preds

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path', default='C:/Users/heeryung/code/24w-Tri-Modalities/data/GoogleNews-vectors-negative300.bin', type=str)
    parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w-Tri-Modalities/data/msrvtt_category_train.pkl', type=str)
    parser.add_argument('--val_data_path', default='C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl', type=str)
    parser.add_argument('--save_path', default='C:/Users/heeryung/code/24w_deep_daiv/ckpt/trial2_audio_flatten', type=str)
    parser.add_argument('--token_projection', default='projection_net', type=str) # 한결이가 만든 projection_net 쓸건지
    parser.add_argument('--batch_size', default=16, type=int) # 한결이가 만든 projection_net 쓸건지
    args = parser.parse_args()

    # setup data_loader instances
    we=None
    we=KeyedVectors.load_word2vec_format(args.we_path, binary=True)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    dataset = MSRVTT_DataLoader(
            data_path=args.data_path,
            we=we
            )
    
    val_dataset = MSRVTT_DataLoader(
            data_path=args.val_data_path,
            we=we
            )
    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    loss = torch.nn.CrossEntropyLoss()
    net = EverythingAtOnceModel(args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

    total_video_correct = 0
    total_audio_correct = 0
    total_text_correct = 0
    total_hard_vote_correct = 0
    total_soft_vote_correct = 0

   

    for epoch in range(0,1001):
        net.train()
        running_loss = 0.0
        print('Epoch: %d' % epoch)
        for i_batch, sample_batch in enumerate(data_loader):
            batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss)
            running_loss += batch_loss
        print('Total loss: ', running_loss / len(data_loader))

        if epoch % 10 == 0:
            checkpoint = {'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict()
                        }
            torch.save(checkpoint, os.path.join(save_path, 'epoch{}.pth'.format(epoch)))

            net.eval()
            with torch.no_grad():
                for val_batch in val_data_loader:
                    video = val_batch['video'].cuda()
                    audio = val_batch['audio'].cuda()
                    text = val_batch['text'].cuda()
                    nframes = val_batch['nframes'].cuda()
                    category = val_batch['category'].cuda()

                    video = video.view(-1, video.shape[-1])
                    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
                    text = text.view(-1, text.shape[-2], text.shape[-1])

                    va, at, tv = net(video, audio, nframes, text, category)
                    va_preds, at_preds, tv_preds = get_predictions(va, at, tv)

                    # Soft voting
                    soft_vote_preds = get_soft_voting(va, at, tv)
                    soft_vote_correct = (soft_vote_preds == category).sum().item()
                    total_soft_vote_correct += soft_vote_correct

                    # Hard voting
                    hard_vote_preds = get_hard_voting(va_preds, at_preds, tv_preds)
                    hard_vote_correct = (hard_vote_preds == category).sum().item()
                    total_hard_vote_correct += hard_vote_correct

                    # Calculate accuracy for each modality
                    video_correct = (va_preds == category).sum().item()
                    audio_correct = (at_preds == category).sum().item()
                    text_correct = (tv_preds == category).sum().item()

                    total_video_correct += video_correct
                    total_audio_correct += audio_correct
                    total_text_correct += text_correct
                
                # Calculate final accuracies
                video_accuracy = total_video_correct / (len(val_data_loader) * batch_size)
                audio_accuracy = total_audio_correct / (len(val_data_loader) * batch_size)
                text_accuracy = total_text_correct / (len(val_data_loader) * batch_size)
                hard_vote_accuracy = total_hard_vote_correct / (len(val_data_loader) * batch_size)
                soft_vote_accuracy = total_soft_vote_correct / (len(val_data_loader) * batch_size)

                print("Video accuracy:", video_accuracy)
                print("Audio accuracy:", audio_accuracy)
                print("Text accuracy:", text_accuracy)
                print("Hard voting accuracy:", hard_vote_accuracy)
                print("Soft voting accuracy:", soft_vote_accuracy)

            fig, ax1 = plt.subplots()
            ax1.plot(epoch, hard_vote_accuracy, color = 'red', alpha = 0.5)
            ax2 = ax1.twinx()
            ax2.plot(epoch, soft_vote_accuracy, color = 'blue', alpha = 0.5)
            ax3 = ax1.twinx()
            ax3.plot(epoch, audio_accuracy, color='yellow', alpha = 0.5)

            plt.title("Hard voting/soft voting/audio accuracy")
            plt.show()
            plt.savefig(save_path + '/' + f'eval_graph.png')
            plt.close(fig)