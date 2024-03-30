import torch
import torch.nn as nn
from dataset.msrvtt_dataloader import MSRVTT_DataLoader
from model.fusion_model import EverythingAtOnceModel
from gensim.models.keyedvectors import KeyedVectors
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--we_path', default='./data/GoogleNews-vectors-negative300.bin', type=str)
parser.add_argument('--data_path', default='C:/Users/heeryung/code/24w_deep_daiv/msrvtt_category_test.pkl', type=str)
parser.add_argument('--checkpoint_path', default='C:/Users/heeryung/code/24w_deep_daiv/ckpt/trial3_audio_davenet/epoch100.pth', type=str)
parser.add_argument('--token_projection', default='projection_net', type=str) 
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_path)

we = None 
we = KeyedVectors.load_word2vec_format(args.we_path, binary=True)

dataset = MSRVTT_DataLoader(
        data_path=args.data_path,
        we=we
        )

data_loader = DataLoader(dataset, batch_size=16)

net = EverythingAtOnceModel(args).cuda()
optimizer = torch.optim.Adam(net.parameters(), lr =0.001)

net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

total_samples = 0
total_video_correct = 0
total_audio_correct = 0
total_text_correct = 0
total_hard_vote_correct = 0
total_soft_vote_correct = 0

for data in data_loader:
    video = data['video'].cuda()
    audio = data['audio'].cuda()
    text = data['text'].cuda()
    nframes = data['nframes'].cuda()
    category = data['category'].cuda()

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

    total_samples += category.size(0)
    total_video_correct += video_correct
    total_audio_correct += audio_correct
    total_text_correct += text_correct

# Calculate final accuracies
video_accuracy = total_video_correct / total_samples
audio_accuracy = total_audio_correct / total_samples
text_accuracy = total_text_correct / total_samples
hard_vote_accuracy = total_hard_vote_correct / total_samples
soft_vote_accuracy = total_soft_vote_correct / total_samples

print("Video accuracy:", video_accuracy)
print("Audio accuracy:", audio_accuracy)
print("Text accuracy:", text_accuracy)
print("Hard voting accuracy:", hard_vote_accuracy)
print("Soft voting accuracy:", soft_vote_accuracy)
