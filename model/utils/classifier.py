import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim):
        super(Classifier, self).__init__()
        self.video_layer = nn.Linear(1054*embed_dim, 512) #512로 해보기
        self.audio_layer = nn.Linear(31*embed_dim, 512)
        self.text_layer = nn.Linear(1025*embed_dim, 512)
        self.fc_layer = nn.Linear(512*3, num_classes)
    
    def forward(self, video_embed, audio_embed, text_embed):
        v = video_embed.view(video_embed.size(0),-1) # [16, 1054, 1024] -> [16, 1024]
        a = audio_embed.view(audio_embed.size(0),-1) # [16, 31, 1024] -> [16, 1024]
        t = text_embed.view(text_embed.size(0),-1) # [16, 1025, 1024] -> [16, 1024]

        v = torch.relu(self.video_layer(v)) 
        a = torch.relu(self.audio_layer(a)) 
        t = torch.relu(self.text_layer(t)) 

        x = torch.cat((v, a, t), dim=1) # [16, 1024] -> [16, 1024*3]
        x = self.fc_layer(x)
        return x