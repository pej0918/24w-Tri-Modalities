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

    # ##### 수정해야함
    # # Cross Attention 적용
    # sim_audio_video = torch.matmul(audio, video.t())
    # sim_audio_text = torch.matmul(audio, text.t())
    # sim_text_video = torch.matmul(text, video.t())
    # #loss = loss_fun(sim_audio_video) + loss_fun(sim_audio_text) + loss_fun(sim_text_video)
    # va, at, tv = model(image) #3개
    # loss(output,category) #3개
    # loss = a + b + c
    # loss.backward()
    # opt.step()
    # return loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path', default='C:/Users/heeryung/code/24w-Tri-Modalities/data/GoogleNews-vectors-negative300.bin', type=str)
    parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w-Tri-Modalities/data/msrvtt_category_train.pkl', type=str)
    parser.add_argument('--save_path', default='C:/Users/heeryung/code/24w_deep_daiv\chpt', type=str)
    parser.add_argument('--token_projection', default='projection_net', type=str) # 한결이가 만든 projection_net 쓸건지
    args = parser.parse_args()

    # setup data_loader instances
    we=None
    we=KeyedVectors.load_word2vec_format(args.we_path, binary=True)

    save_path = args.save_path

    dataset = MSRVTT_DataLoader(
            data_path=args.data_path,
            we=we
            )

    data_loader = DataLoader(dataset, batch_size=16)

    loss = torch.nn.CrossEntropyLoss()
    net = EverythingAtOnceModel(args).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

    for epoch in range(100):
        running_loss = 0.0
        print('Epoch: %d' % epoch)

        for i_batch, sample_batch in enumerate(data_loader):
            batch_loss = TrainOneBatch(net, optimizer, sample_batch, loss)
            running_loss += batch_loss
        print(running_loss / len(data_loader))

        if epoch % 10 == 0:
            checkpoint = {'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'scheduler_state_dict': scheduler.state_dict()
                        }
            torch.save(checkpoint, os.path.join(save_path, 'epoch{}.path'.format(epoch)))



            ################ 뒷부분 수정해야함 loss 추가
            
        #     if (i_batch + 1) % args.n_display == 0 and args.verbose:
        #         print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
        #         (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
        #         running_loss / args.n_display))
        #         print('Batch load time avg: %.4f, Batch process time avg: %.4f' %
        #         (data_time.avg, batch_time.avg))
        #         running_loss = 0.0
        #         # reset the load meters
        #         batch_time = AverageMeter()
        #         data_time = AverageMeter()
        # save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
        #             else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] *= args.lr_decay
        # if args.checkpoint_dir != '':
        #     path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(save_epoch))
        #     net.module.save_checkpoint(path)
        #     if args.apex_level == 1:
        #         amp_checkpoint = {'net': net.module.state_dict(),
        #                         'optimizer': optimizer.state_dict(),
        #                         'amp': amp.state_dict()}
        #         torch.save(amp_checkpoint, os.path.join(args.checkpoint_dir, 'amp_checkpoint.pt'))
