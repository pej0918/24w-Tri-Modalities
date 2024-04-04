import argparse
import collections
import os

from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import time
import torch 
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from functools import partial

import math
import tqdm

import matplotlib.pyplot as plt

def TrainOneBatch(model, opt, data, loss_fun, use_cls_token=False):
    video = data['video'].to(device)
    audio = data['audio'].to(device)
    text = data['text'].to(device)
    nframes = data['nframes'].to(device)
    category = data['category'].to(device)
    # print('video:', video.shape, 'audio:', audio.shape, 'text:', text.shape)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])
    nframes = nframes.view(-1)
    # print('video:', video.shape, 'audio:', audio.shape, 'text:', text.shape)
    opt.zero_grad()

    # loss
    if use_cls_token:
        v, a, t = model(video, audio, nframes, text, category)
        loss_v = loss_fun(v, category)
        loss_a = loss_fun(a, category)
        loss_t = loss_fun(t, category)
        loss = loss_v + loss_a + loss_t
    else:
        # pred = model(video, audio, nframes, text, category)
        # loss = loss_fun(pred, category)
        v, a, t = model(video, audio, nframes, text, category)
        loss_v = loss_fun(v, category)
        loss_a = loss_fun(a, category)
        loss_t = loss_fun(t, category)
        loss = loss_v + loss_a + loss_t

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


def EvalUseClsToken(val_batch, net):
    video = val_batch['video'].to(device)
    audio = val_batch['audio'].to(device)
    text = val_batch['text'].to(device)
    nframes = val_batch['nframes'].to(device)
    category = val_batch['category'].to(device)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])

    va, at, tv = net(video, audio, nframes, text, category)
    va_preds, at_preds, tv_preds = get_predictions(va, at, tv)

    # Soft voting
    soft_vote_preds = get_soft_voting(va, at, tv)
    soft_vote_correct = (soft_vote_preds == category).sum().item()

    # Hard voting
    hard_vote_preds = get_hard_voting(va_preds, at_preds, tv_preds)
    hard_vote_correct = (hard_vote_preds == category).sum().item()


    # Calculate accuracy for each modality
    video_correct = (va_preds == category).sum().item()
    audio_correct = (at_preds == category).sum().item()
    text_correct = (tv_preds == category).sum().item()

    return video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct


def EvalEmbedvec(val_batch, net):
    video = val_batch['video'].to(device)
    audio = val_batch['audio'].to(device)
    text = val_batch['text'].to(device)
    nframes = val_batch['nframes'].to(device)
    category = val_batch['category'].to(device)

    video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    text = text.view(-1, text.shape[-2], text.shape[-1])

    pred = net(video, audio, nframes, text, category)
    _, pred = torch.max(pred.data, 1)
    correct = (pred == category).sum().item()
    return correct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path',
                        default='data/GoogleNews-vectors-negative300.bin',
                        type=str)
    parser.add_argument('--data_path', default='data/msrvtt_category_train.pkl',
                        type=str)
    parser.add_argument('--val_data_path', default='data/msrvtt_category_test.pkl',
                        type=str)
    parser.add_argument('--save_path', default='ckpt', type=str)
    parser.add_argument('--exp', default='trial2', type=str)
    parser.add_argument('--use_softmax', default=True, type=bool)
    parser.add_argument('--use_cls_token', default=False, type=bool)
    parser.add_argument('--token_projection', default='projection_net', type=str)
    parser.add_argument('--num_classes', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--ckpt_path', default='', type=str)
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--device', default="0", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    # setup data_loader instances
    we = None
    we = KeyedVectors.load_word2vec_format(args.we_path, binary=True)

    save_path = args.save_path + '/' + args.exp
    os.makedirs(save_path, exist_ok=True)

    dataset = MSRVTT_DataLoader(data_path=args.data_path, we=we)
    val_dataset = MSRVTT_DataLoader(data_path=args.val_data_path, we=we)

    batch_size = args.batch_size
    data_loader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

    loss = torch.nn.CrossEntropyLoss()
    net = EverythingAtOnceModel(args).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    epoch = args.epoch

    total_video_correct = 0
    total_audio_correct = 0
    total_text_correct = 0
    total_hard_vote_correct = 0
    total_soft_vote_correct = 0
    total_correct = 0
    total_num = 0

    epochs_list = []
    accuracy_list = []
    hard_accuracy_list = []
    soft_accuracy_list = []

    for epoch in range(0, epoch + 1):
        net.train()
        running_loss = 0.0

        for i_batch, sample_batch in enumerate(data_loader):
            batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss, use_cls_token=args.use_cls_token)
            running_loss += batch_loss

        print('Epoch: {} / Total loss: {}'.format(epoch, running_loss / len(data_loader)))
        scheduler.step()

        if epoch % 10 == 0:
            # Save checkpoint
            checkpoint = {'epoch': epoch,
                          'model_state_dict': net.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict()}
            torch.save(checkpoint, os.path.join(save_path, 'epoch{}.pth'.format(epoch)))

            # validation
            net.eval()
            with torch.no_grad():
                for val_batch in val_data_loader:
                    category = val_batch['category'].to(device)

                    if args.use_cls_token:
                        video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct = EvalUseClsToken(
                            val_batch, net)

                        total_soft_vote_correct += soft_vote_correct
                        total_hard_vote_correct += hard_vote_correct
                        total_video_correct += video_correct
                        total_audio_correct += audio_correct
                        total_text_correct += text_correct

                    else:
                        # correct = EvalUseEmbedvec(val_batch, net)
                        # total_correct += correct
                        video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct = EvalUseClsToken(
                            val_batch, net)
                        total_soft_vote_correct += soft_vote_correct
                        total_hard_vote_correct += hard_vote_correct
                        total_video_correct += video_correct
                        total_audio_correct += audio_correct
                        total_text_correct += text_correct

                    total_num += category.size(0)

                # Calculate final accuracies
                if args.use_cls_token:
                    print("Video accuracy:", total_video_correct / total_num)
                    print("Audio accuracy:", total_audio_correct / total_num)
                    print("Text accuracy:", total_text_correct / total_num)
                    print("Hard voting accuracy:", total_hard_vote_correct / total_num)
                    print("Soft voting accuracy:", total_soft_vote_correct / total_num)

                else:

                    hard_vote_accuracy = total_hard_vote_correct / total_num
                    soft_vote_accuracy = total_soft_vote_correct / total_num

                    print("Video accuracy:", total_video_correct / total_num)
                    print("Audio accuracy:", total_audio_correct / total_num)
                    print("Text accuracy:", total_text_correct / total_num)
                    # print("Hard voting accuracy:", hard_vote_accuracy)
                    print("Soft voting accuracy:", soft_vote_accuracy)

                    epochs_list.append(epoch)
                    hard_accuracy_list.append(hard_vote_accuracy)
                    soft_accuracy_list.append(soft_vote_accuracy)

                    plt.figure(figsize=(10, 6))
                    # plt.plot(epochs_list, hard_accuracy_list, marker='o', linestyle='-', color='b', label='Hard Voting')
                    plt.plot(epochs_list, soft_accuracy_list, marker='o', linestyle='-', color='r', label='Soft Voting')
                    plt.title('Accuracy over Epochs')
                    plt.xlabel('Epoch')
                    plt.ylabel('Accuracy')
                    plt.grid(True)
                    plt.savefig(save_path + '/accuracy_epoch{}.png'.format(epoch))
                    plt.close()