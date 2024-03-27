import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(embed_dim*3, embed_dim)
        self.fc2 = nn.Linear(embed_dim, num_classes)
    
    def forward(self, video_embed, audio_embed, text_embed):
        v = video_embed.mean(dim=1) # [16, 1054, 1024] -> [16, 1024]
        a = audio_embed.mean(dim=1) # [16, 31, 1024] -> [16, 1024]
        t = text_embed.mean(dim=1) # [16, 1025, 1024] -> [16, 1024]

        x = torch.cat((v, a, t), dim=1) # [16, 1024] -> [16, 1024*3]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x