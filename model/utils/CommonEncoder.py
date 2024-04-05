import torch.nn as nn
import torch 
import torch.nn.functional as F


class CommonEncoder(nn.Module):
    def __init__(self, common_dim, latent_dim):
        super(CommonEncoder, self).__init__()
        self.common_dim = common_dim
        self.latent_dim = latent_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(common_dim, 512),
            Swish(),
            nn.Linear(512, 512),
            Swish(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.feature_extractor(x), dim=-1)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)