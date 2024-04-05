import torch
import torch.nn as nn
from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--we_path', default='C:/Users/heeryung/code/24w_deep_daiv/GoogleNews-vectors-negative300.bin', type=str)
    parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl', type=str)
    parser.add_argument('--checkpoint_path', default='D:/download/epoch200.pth', type=str)
    parser.add_argument('--token_projection', default='projection_net', type=str) 
    parser.add_argument('--use_softmax', default=True, type=bool) 
    parser.add_argument('--use_cls_token', default=False, type=bool) 
    parser.add_argument('--num_classes', default=20, type=int) 
    parser.add_argument('--batch_size', default=16, type=int) 
    args = parser.parse_args()

    device = torch.device("cuda")

    checkpoint = torch.load(args.checkpoint_path)

    we = None 
    we = KeyedVectors.load_word2vec_format(args.we_path, binary=True)

    dataset = MSRVTT_DataLoader(data_path=args.data_path, we=we)
    data_loader = DataLoader(dataset, batch_size=args.batch_size)

    net = EverythingAtOnceModel(args).to(device)
    # optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

    net.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    net.eval()

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
    
    def get_topk_predictions(va, at, tv, k):
        _, va_preds = torch.max(va, k)
        _, at_preds = torch.max(at, k)
        _, tv_preds = torch.max(tv, k)
        return va_preds, at_preds, tv_preds 

    def calculate_accuracy(predictions, labels):
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy
    
    def calculate_f1_score(predictions, labels):
        f1 = f1_score(labels.cpu(), predictions.cpu(), average='weighted')
        return f1
    
    def Eval(val_batch, net):
        video = val_batch['video'].to(device)
        audio = val_batch['audio'].to(device)
        text = val_batch['text'].to(device)
        nframes = val_batch['nframes'].to(device)
        category = val_batch['category'].to(device)

        video = video.view(-1, video.shape[-1])
        audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
        text = text.view(-1, text.shape[-2], text.shape[-1])

        va, at, tv = net(video, audio, nframes, text, category)
        print('va', va, 'at', at, 'tv', tv)

        va_preds, at_preds, tv_preds = get_predictions(va, at, tv)
        # va_top5_preds, at_top5_preds, tv_top5_preds = get_topk_predictions(va, at, tv, k=5)

        # Soft voting
        soft_vote_preds = get_soft_voting(va, at, tv)
        soft_vote_correct = (soft_vote_preds == category).sum().item()

        # Hard voting
        hard_vote_preds = get_hard_voting(va_preds, at_preds, tv_preds)
        hard_vote_correct = (hard_vote_preds == category).sum().item()

        # F1 Score
        f1 = calculate_f1_score(soft_vote_preds, category)  # 소프트 보팅의 예측 결과로 f1 score 계산
        
        # Calculate accuracy for each modality
        video_correct = (va_preds == category).sum().item()
        audio_correct = (at_preds == category).sum().item()
        text_correct = (tv_preds == category).sum().item()

        return video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct, f1

    total_num = 0
    total_accuracy = 0
    total_video_correct = 0
    total_audio_correct = 0
    total_text_correct = 0
    total_hard_vote_correct = 0
    total_soft_vote_correct = 0

    for test_batch in data_loader:
        category = test_batch['category'].to(device)

        video_correct, audio_correct, text_correct, soft_vote_correct, hard_vote_correct, _ = Eval(test_batch, net)
        total_soft_vote_correct += soft_vote_correct
        total_hard_vote_correct += hard_vote_correct
        total_video_correct += video_correct
        total_audio_correct += audio_correct
        total_text_correct += text_correct

        total_num += category.size(0)

    # Calculate final accuracies
    hard_vote_accuracy = total_hard_vote_correct / total_num
    soft_vote_accuracy = total_soft_vote_correct / total_num

    print("Video accuracy:", total_video_correct / total_num)
    print("Audio accuracy:", total_audio_correct / total_num)
    print("Text accuracy:", total_text_correct / total_num)
    print("Hard voting accuracy:", hard_vote_accuracy)
    print("Soft voting accuracy:", soft_vote_accuracy)
    #print("F1 Score:", f1_accuracy)